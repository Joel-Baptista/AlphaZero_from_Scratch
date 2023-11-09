from tictactoe import TicTacToe
from mcts import MCTS
import numpy as np

args = {
    'C': 1.41,
    'num_searches': 1_000
}

def main():

    tictactoe = TicTacToe()
    player = 1
    mcts = MCTS(tictactoe, args)

    state = tictactoe.get_initial_state()

    while True:
        print(state)

        if player == 1:
            valid_moves = tictactoe.get_valid_moves(state)
            print("valid_moves:", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("action not valid")
                continue
        else:
            neutral_state = tictactoe.change_prespective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        state = tictactoe.get_next_state(state, action, player)

        value, is_terminal = tictactoe.get_value_and_terminated(state, action)

        if is_terminal:
            print(state)
            if value == 1:
                print(player, " won")

            else:
                print("draw")

            break

        player = tictactoe.get_opponent(player)

if __name__=="__main__":
    main()