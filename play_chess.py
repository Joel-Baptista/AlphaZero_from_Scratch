from chess_game import ChessGame
from chess_mcts import MCTS
import numpy as np

args = {
    'C': 1.41,
    'num_searches': 10
}

def main():

    game = ChessGame()
    player = 1
    mcts = MCTS(game, args)

    state = game.get_initial_state()

    while True:
        game.show(state)

        if player == 1:
            valid_moves = game.get_uci_valid_moves(state)
            print(valid_moves)

            action = str(input(f"{player}:"))
        else:

            valid_moves, mcts_probs = mcts.search(state)
            action = valid_moves[np.argmax(mcts_probs)]

       
        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            game.show(state)
            if value == 1:
                print(player, " won")

            else:
                print("draw")

            break

        state = game.get_next_state(state, action)
        player = game.get_opponent(player)

if __name__=="__main__":
    main()