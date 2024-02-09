from connect_four import ConnectFour
from mcts import MCTS
import numpy as np
import random

from connect_four import ConnectFour
from mcts_alpha import MCTS as MCTSAlpha
from model import ResNet
import numpy as np
import torch
import time

args = {
    "C": 1.41,
    "num_searches": 100,
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
    model.load_state_dict(torch.load('models/model_10_ConnectFour.pt', map_location=device))
    model.eval() 

    mcts_aplha = MCTSAlpha(game, args, model)

    game = ConnectFour()
    player = 1
    mcts = MCTS(game, {'C': 1.41, 'num_searches': 10})

    r = random.random()
    if r > 0.5:
        alpha_player = 1
    else:
        alpha_player = -1
       
    alpha_wins = 0
    st = time.time()
    for i in range(1, 11):
        print(f"Start Game number {i}")
        state = game.get_initial_state()

        r = random.random()
        if r > 0.5:
            alpha_player = 1
        else:
            alpha_player = -1

        print(alpha_player)
        while True:
            # game.show(state)

            if player == alpha_player:
                # print("Alpha Turn")
                neutral_state = game.change_prespective(state, player)
                mcts_probs = mcts_aplha.search(neutral_state)
                action = np.argmax(mcts_probs)
            else:
                # print("MTCS Turn")
                neutral_state = game.change_prespective(state, player)
                mcts_probs = mcts.search(neutral_state)
                action = np.argmax(mcts_probs)

            state = game.get_next_state(state, action, player)

            value, is_terminal = game.get_value_and_terminated(state, action)
            
            if is_terminal:
                # game.show(state)
                if value == 1:
                    winner = "MTCS"
                    if player == alpha_player: 
                        winner = "Alpha"
                        alpha_wins += 1

                    print(winner, " won")

                else:
                    # print("draw")
                    pass

                break

            player = game.get_opponent(player)

    print(f"Alpha won {alpha_wins} games")
    print(f"Average game time: {(time.time() - st) / i}")
if __name__=="__main__":
    main()