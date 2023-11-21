from mcts_alpha import MCTS, MCTSParallel
from tictactoe import TicTacToe
from connect_four import ConnectFour
from model import ResNet
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import wandb
import time
import os

torch.manual_seed(0)

class AlphaZero:
    def __init__(self, model, optimizer, game, args, log_mode = False) -> None:
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.init_time = None

        if log_mode:
            run = wandb.init(
                # Set the project where this run will be logged
                project="AlphaZero",
                name=str(game),
                # Track hyperparameters and run metadata
                config=args
                )
            print(run.config)


    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_prespective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args["temperature"])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append(
                        (self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome)
                    )
                return returnMemory
            
            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx:min(batchIdx+self.args["batch_size"], len(memory)-1)]
            if len(sample)<2:
                continue

            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def learn(self):
        self.init_time = time.time()
        for iteration in range(self.args["num_iterations"]):
            print(f"Iteration number {iteration}")
            memory = []

            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args["num_selfPlay_iterations"])):
                memory += self.selfPlay()

            self.model.train()
            for epoch in tqdm(range(self.args["num_epochs"])):
                self.train(memory)

            torch.save(self.model.state_dict(), f"models/model_{iteration}_{self.game}.pt") 
            torch.save(self.optimizer.state_dict(), f"models/optimizer_{iteration}_{self.game}.pt") 

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, log_mode = False,  save_models=False) -> None:
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self.log_mode = log_mode
        self.save_models = save_models
        self.init_time = None
        self.log_iteration = 0

        if log_mode:
            run = wandb.init(
                # Set the project where this run will be logged
                project="AlphaZero",
                name=str(game),
                # Track hyperparameters and run metadata
                config=self.args
                )
            self.args = run.config
        
        print(self.args)


    def selfPlay(self):
        return_memory = []
        player = 1

        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])

            neutral_states = self.game.change_prespective(states, player)
            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count

                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args["temperature"])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:   
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append(
                            (self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome)
                        )
                    del spGames[i]
            

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

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
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
    
    def eval(self, memory):
        random.shuffle(memory)
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        iterations = 0
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx:min(batchIdx+self.args["batch_size"], len(memory)-1)]
            if len(sample)<2:
                continue

            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)

            loss = policy_loss + value_loss

            total_loss += loss.detach().cpu().item()
            total_policy_loss += policy_loss.detach().cpu().item()
            total_value_loss += value_loss.detach().cpu().item()
            iterations += 1
        
        val_loss = total_loss / iterations
        val_policy_loss = total_policy_loss / iterations
        val_value_loss = total_value_loss / iterations

        print(f"Validation loss: {val_loss}, Policy Validation Loss: {val_policy_loss}, Value Validation Loss: {val_value_loss}")
        if self.log_mode:
            wandb.log({
                "val_loss": val_loss,
                "val_policy_loss": val_policy_loss,
                "val_value_loss": val_value_loss }
                , step=self.log_iteration)

    def learn(self):
        self.init_time = time.time()
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

            self.model.eval()
            val_memory = []
            validation_iterations = self.args["num_selfPlay_iterations"] // (5 * self.args["num_parallel_games"])
            if validation_iterations < 1: validation_iterations = 1
            for selfPlay_iteration in tqdm(range(validation_iterations)):
                val_memory += self.selfPlay()
            self.eval(val_memory)

            if self.save_models:
                if not os.path.exists("models"):
                    os.mkdir("models")

                torch.save(self.model.state_dict(), f"models/model_{iteration}_{self.game}.pt") 
                torch.save(self.optimizer.state_dict(), f"models/optimizer_{iteration}_{self.game}.pt") 

class SPG:
    def __init__(self, game) -> None:
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

if __name__=="__main__":
    game = ConnectFour()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 9, 128, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    print(os.path.exists("models"))
    args = {
        'C': 3,
        'num_searches': 100,
        'num_iterations': 20,
        'num_selfPlay_iterations': 6,
        'num_parallel_games': 2,
        'num_epochs': 8,
        'batch_size': 2,
        'temperature': 2,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.4  
    }

    alphaZero = AlphaZeroParallel(model, optimizer, game, args, log_mode=True, save_models=False)

    alphaZero.learn()
