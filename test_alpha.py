from tictactoe import TicTacToe
from connect_four import ConnectFour
from mcts_alpha import MCTS
from model import ResNet
import numpy as np
import torch

args = {
    "C": 5,
    "num_searches": 600,
    "num_iterations": 40,
    "num_selfPlay_iterations": 500,
    "num_parallel_games": 250,
    "num_epochs": 6,
    "batch_size": 64,
    "temperature": 0.2,
    "dirichlet_epsilon": 0.5,
    "dirichlet_alpha": 0.8,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "num_resblocks": 15,
    "num_hidden": 1024,  
    }

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    game = ConnectFour()
    player = 1
    model = ResNet(game, args["num_resblocks"], args["num_hidden"], device)
    model.load_state_dict(torch.load('model/model_10_ConnectFour.pt', map_location=device))
    model.eval() 

    mcts = MCTS(game, args, model)

    state = game.get_initial_state()

    while True:
        game.show(state)

        if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("valid_moves:", [i for i in range(game.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("action not valid")
                continue
        else:
            neutral_state = game.change_prespective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        state = game.get_next_state(state, action, player)

        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            game.show(state)
            if value == 1:
                print(player, " won")

            else:
                print("draw")

            break

        player = game.get_opponent(player)


if __name__=="__main__":
    main()