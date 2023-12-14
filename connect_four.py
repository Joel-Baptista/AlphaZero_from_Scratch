import numpy as np
from colorama import Fore

class ConnectFour:
    def __init__(self) -> None:
        self.row_count = 6
        self.collumn_count = 7
        self.action_size = self.collumn_count 
        self.in_a_row = 4

    def __repr__(self) -> str:
        return "ConnectFour"

    def show(self, state) -> None:

        for row in state:
            str_row = "| "
            for item in row:
                if item == 1:
                    str_row += Fore.RED
                elif item == -1:
                    str_row += Fore.YELLOW
                str_row +=  "O "
                str_row += Fore.RESET
            print(f"{str_row} |")
            

    def get_initial_state(self):
        return np.zeros((self.row_count, self.collumn_count))

    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state
    
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action is None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        collumn = action

        player = state[row][collumn]

        def count(offset_row, offset_collumn):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_collumn * i
                if (
                    r < 0
                    or r>= self.row_count
                    or c < 0
                    or c>= self.collumn_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical check
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal check
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal check
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal check
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        
        return 0, False

    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value

    def change_prespective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state




