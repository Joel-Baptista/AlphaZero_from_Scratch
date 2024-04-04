from chess_mcts_alpha import MCTSParallel
from chess_mcts_alpha import MCTS as MCTSAlpha
from chess_game import ChessGame
from model import ResNetChess
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import wandb
import time
import os
from stockfish import Stockfish
import copy
import argparse

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game: ChessGame, args, log_mode = False,  save_models=False) -> None:
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, self.args, model)
        self.log_mode = log_mode
        self.save_models = save_models
        self.init_time = None
        self.log_iteration = 0
        print(self.args)



    def selfPlay(self):
        return_memory = []
        player = 1

        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        j = 0
        while len(spGames) > 0:
            j += 1
            print(f"Episode Lenght: {j}")
            # neutral_states = self.game.change_prespective(states, player)
            st = time.time()
            print("Starting Search")
            self.mcts.search(spGames)
            print(f"Search Time: {time.time() - st}")
            st = time.time()
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                valid_actions = self.game.get_uci_valid_moves(spg.state)

                action_probs = np.zeros((1, 8, 8, 73))
                for child in spg.root.children:
                    action_probs += child.visit_count * self.game.encode_action(spg.state, child.action_taken)

                # print(action_probs)
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                # TODO Implement temperature again
                # temperature_action_probs = action_probs ** (1 / self.args["temperature"])
                # temperature_action_probs /= np.sum(temperature_action_probs)

                # action_idx = np.random.choice(len(valid_actions), p=temperature_action_probs)
                action = self.game.decode_action(action_probs, spg.state)[0]

                spg.board_state = self.game.get_next_state(spg.board_state, action)
                spg.state = spg.board_state.fen()

                if (j % 10) == 0:
                    self.game.show(spg.state)
                    print("-------------------")

                value, is_terminal = self.game.get_value_and_terminated(spg.board_state, action)

                if is_terminal:   
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append(
                            (self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome)
                        )
                    del spGames[i]
            print(f"Sorting data {time.time() - st}")
            print(f"Memory size: {len(return_memory)}")
            print(f"Games Running: {len(spGames)}")
            player = self.game.get_opponent(player)
        
        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx:min(batchIdx+self.args["batch_size"], len(memory)-1)]
            if len(sample)<2:
                continue

            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            # print(policy_targets)
            # print(policy_targets[0].shape)
            # print(policy_targets[0][0])
            # print(type(policy_targets[0][0]))

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
    
            out_policy, out_value = self.model(state.squeeze(1))
    
            # print(policy_targets.shape)

            policy_loss = F.cross_entropy(out_policy, policy_targets.squeeze(1))
            value_loss = F.mse_loss(out_value, value_targets)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.log_mode:
                self.log_iteration += 1
                wandb.log({
                    "loss": loss.detach().cpu().item(),
                    "policy_loss": policy_loss.detach().cpu().item(),
                    "value_loss": policy_loss.detach().cpu().item(),
                    "time": (time.time() - self.init_time) / 60
                        }, step=self.log_iteration)
    
    def eval(self):
        player = 1

        mtcs_args = {
            "C": 1.41,
            "num_searches": 300,
            "dirichlet_epsilon": 0.2,
            "dirichlet_alpha": 1.3,
        }
        mcts_alpha = MCTSAlpha(self.game, mtcs_args, self.model)
        alpha_wins = 0
        alpa_draws = 0
        st = time.time()
        for i in range(1, 21):
            print(f"Start Game number {i}")
            state = self.game.get_initial_state()
            board_state = game.get_board_state(state)
            # stockfish = Stockfish(path="./stockfish/stockfish-ubuntu-x86-64-avx2", depth=18, parameters={"Threads": 2, "Minimum Thinking Time": 30})
            r = random.random()
            if r > 0.5:
                alpha_player = 1
            else:
                alpha_player = -1

            while True:
                state = board_state.fen()
                # game.show(state)

                if player == alpha_player:
                    # print("Alpha Turn")
                    valid_moves, mcts_probs = mcts_alpha.search(state)
                    action = valid_moves[np.argmax(mcts_probs)]
                else:
                    # print("MTCS Turn")
                    valid_moves = self.game.get_uci_valid_moves(state)
                    action = random.choice(valid_moves)
                    # action = stockfish.get_best_move()

                board_state = game.get_next_state(board_state, action)

                value, is_terminal = game.get_value_and_terminated(board_state, action)
                
                if is_terminal:
                    # game.show(state)
                    if value == 1:
                        winner = "MTCS"
                        if player == alpha_player: 
                            winner = "Alpha"
                            alpha_wins += 1
                    elif value == 0:
                        alpa_draws += 1
                        # print(winner, " won")
                    break

                player = self.game.get_opponent(player)

        alpha_wins_ratio = alpha_wins / i
        if self.log_mode:         
            wandb.log({
                "win_rate": alpha_wins_ratio
            })

    def learn(self):
        self.init_time = time.time()
        checkpoint_counter = 4
        if checkpoint_counter < 1: checkpoint_counter = 1

        for iteration in range(self.args["num_iterations"]):
            print(f"Iteration number {iteration}")
            print(self.args)
            memory = []

            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args["num_selfPlay_iterations"] // self.args["num_parallel_games"])):
                memory += self.selfPlay()

            self.model.train()
            for epoch in tqdm(range(self.args["num_epochs"])):
                self.train(memory)


            if self.save_models and iteration % checkpoint_counter == 0:
                if not os.path.exists("models"):
                    os.mkdir("models")

                torch.save(self.model.state_dict(), f"models/model_{iteration}_{self.game}.pt") 
                torch.save(self.optimizer.state_dict(), f"models/optimizer_{iteration}_{self.game}.pt") 

            self.model.eval()
            self.eval()
        # val_memory = []
        # validation_iterations = self.args["num_selfPlay_iterations"] // (5 * self.args["num_parallel_games"])
        # if validation_iterations < 1: validation_iterations = 1
        # for selfPlay_iteration in tqdm(range(validation_iterations)):
        #     val_memory += self.selfPlay()

class SPG:
    def __init__(self, game: ChessGame) -> None:
        self.state = game.get_initial_state()
        self.board_state = game.get_board_state(self.state)
        self.memory = []
        self.root = None
        self.node = None

if __name__=="__main__":
    debug = True
    if "DEVICE" in os.environ:
        device = os.getenv("DEVICE")
    else:
        device = "cuda:0"
    
    # parser = argparse.ArgumentParser(
    #                 prog='ProgramName',
    #                 description='What the program does',
    #                 epilog='Text at the bottom of help')

    # parser.add_argument('--device', '-c', type=str, default="cuda:0")  
    # parser.add_argument('--debug', '-d', action="store_true", default=False)  
    # args_parsed = parser.parse_args()
    # debug = args_parsed.debug
    # device = args_parsed.device

    args = {
        "C": 3.5,
        "num_searches": 100,
        "num_iterations": 8,
        "num_selfPlay_iterations": 10,
        "num_parallel_games": 10,
        "num_epochs": 8,
        "batch_size": 64,
        "temperature": 3,
        "dirichlet_epsilon": 0.8,
        "dirichlet_alpha": 1.3,
        "lr": 0.0001,
        "weight_decay": 0.001,
        "num_resblocks": 7,
        "num_hidden": 256,  
    }
    # args = {
    #     'C': 2.9904504116582817, 
    #     'batch_size': 113, 
    #     'dirichlet_alpha': 0.5686682396595849, 
    #     'dirichlet_epsilon': 0.9196408364077916, 
    #     'num_epochs': 19, 
    #     'num_iterations': 17, 
    #     'num_parallel_games': 250, 
    #     'num_searches': 3182, 
    #     'num_selfPlay_iterations': 500, 
    #     'temperature': 1.1318118266763582}

    game = ChessGame()

    if not debug:
        run = wandb.init(
        # Set the project where this run will be logged
            project="AlphaZero",
            name=str(game),
            # Track hyperparameters and run metadata
            config=args
            )
        print(run.config)
        args = run.config


    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ResNetChess(game, args["num_resblocks"], args["num_hidden"], device)
    #TODO make a first inference to verifying the full ocupancy of the model in the GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    print(os.path.exists("models"))

    log_mode = not debug
    alphaZero = AlphaZeroParallel(model, optimizer, game, args, log_mode=log_mode, save_models=False)

    alphaZero.learn()

