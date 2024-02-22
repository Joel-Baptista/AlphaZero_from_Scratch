import numpy as np
import random
from chess_game import ChessGame

from chess_mcts_alpha import MCTS as MCTSAlpha
from model import ResNetChess
import numpy as np
import torch
import time
from stockfish import Stockfish

args = {
    "C": 1.41,
    "num_searches": 2,
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
    game = ChessGame()
    player = 1
    model = ResNetChess(game, args["num_resblocks"], args["num_hidden"], device)
    # model.load_state_dict(torch.load('models/model_10_ConnectFour.pt', map_location=device))
    model.eval() 

    mcts_aplha = MCTSAlpha(game, args, model)

    stockfish = Stockfish(path="./stockfish/stockfish-ubuntu-x86-64-avx2", depth=18, parameters={"Threads": 2, "Minimum Thinking Time": 30})
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
                print("Alpha Turn")
                valid_moves, mcts_probs = mcts_aplha.search(state)
                action = valid_moves[np.argmax(mcts_probs)]
            else:
                print("Stockfish Turn")
                stockfish.set_fen_position(state)
                action = stockfish.get_best_move()

            value, is_terminal = game.get_value_and_terminated(state, action, True)
            state = game.get_next_state(state, action)
            
            if is_terminal:
                # game.show(state)
                if value == 1:
                    winner = "StockFish"
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