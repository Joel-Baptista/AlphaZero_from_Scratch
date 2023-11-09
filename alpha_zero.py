from mcts_alpha import MCTS
from tictactoe import TicTacToe
from connect_four import ConnectFour
from model import ResNet
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random

torch.manual_seed(0)

class AlphaZero:
    def __init__(self, model, optimizer, game, args) -> None:
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

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

if __name__=="__main__":
    game = ConnectFour()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 9, 128, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 2,
        'num_searches': 600,
        'num_iterations': 8,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3  
    }

    alphaZero = AlphaZero(model, optimizer, game, args)

    alphaZero.learn()
