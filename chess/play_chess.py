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
    board_state = game.get_board_state(state)

    player_moves = ["e2e4", "f1c4", "g1f3", "f3g5", "c4f7"]
    player_moves_w = ["g1f3", "f3g1","g1f3", "f3g1","g1f3", "f3g1", "g1f3", "f3g1","g1f3", "f3g1","g1f3", "f3g1","g1f3", "f3g1"]
    player_moves_b = ["g8f6", "f6g8","g8f6", "f6g8","g8f6", "f6g8", "g8f6", "f6g8","g8f6", "f6g8","g8f6", "f6g8","g8f6", "f6g8"]
    i = 0
    j = 0
    while True:
        print("GAME STATE")
        state = board_state.fen()
        game.show(state)

        if player == 1:
            is_p_not_valid = True
            valid_moves = game.get_uci_valid_moves(state)
            # while is_p_not_valid:
            #     print(valid_moves)

            #     action = str(input(f"{player}:"))

            #     if action in valid_moves:
            #         is_p_not_valid = False
            #     else:
            #         print("Invalid Move!")
            action = player_moves_w[i]
            i += 1
        else:
            is_p_not_valid = True
            valid_moves = game.get_uci_valid_moves(state)
            # while is_p_not_valid:
            #     print(valid_moves)

            #     action = str(input(f"{player}:"))

            #     if action in valid_moves:
            #         is_p_not_valid = False
            #     else:
            #         print("Invalid Move!")
            action = player_moves_b[j]
            j += 1

            # valid_moves, mcts_probs = mcts.search(state)
            # action = valid_moves[np.argmax(mcts_probs)]

       
        value, is_terminal = game.get_value_and_terminated(board_state, action, False)
        # print(is_terminal)
        board_state = game.get_next_state(board_state, action)

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