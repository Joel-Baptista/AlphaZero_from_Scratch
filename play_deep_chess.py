from chess_game import ChessGame
from chess_mcts_alpha import MCTS
import numpy as np
from model import ResNetChess

args = {
    "C": 5,
    "num_searches": 10,
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

    game = ChessGame()
    player = 1

    model = ResNetChess(game, args["num_resblocks"], args["num_hidden"], "cpu")
    mcts = MCTS(game, args, model)

    state = game.get_initial_state()

    player_moves = ["e2e4", "f1c4", "g1f3", "f3g5", "c4f7"]
    i = 0
    while True:
        print("GAME STATE")
        game.show(state)

        if player == 1:
            is_p_not_valid = True
            valid_moves = game.get_uci_valid_moves(state)
            while is_p_not_valid:
                print(valid_moves)

                action = str(input(f"{player}:"))

                if action in valid_moves:
                    is_p_not_valid = False
                else:
                    print("Invalid Move!")
            # action = player_moves[i]
            # i += 1
        else:

            valid_moves, mcts_probs = mcts.search(state)
            action = valid_moves[np.argmax(mcts_probs)]

       
        value, is_terminal = game.get_value_and_terminated(state, action, True)
        
        state = game.get_next_state(state, action)
        
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