from tictactoe import TicTacToe
from connect_four import ConnectFour
from mcts import MCTS
import numpy as np

args = {
    'C': 1.41,
    'num_searches': 10_000
}

def main():

    game = ConnectFour()
    player = 1
    mcts = MCTS(game, args)

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
            print(state)
            if value == 1:
                print(player, " won")

            else:
                print("draw")

            break

        player = game.get_opponent(player)

if __name__=="__main__":
    main()