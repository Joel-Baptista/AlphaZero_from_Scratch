import numpy as np

class TicTacToe:
    def __init__(self) -> None:
        self.row_count = 3
        self.collumn_count = 3
        self.action_size = self.collumn_count * self.row_count

    def __repr__(self) -> str:
        return "TicTacToe"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.collumn_count))

    def get_next_state(self, state, action, player):
        row = action // self.collumn_count
        collumn = action % self.collumn_count

        state[row, collumn] = player

        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1)==0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action is None:
            return False

        row = action // self.collumn_count
        collumn = action % self.collumn_count

        player = state[row, collumn]        

        return(
            (np.sum(state[row,:]) == player * self.collumn_count) or \
            (np.sum(state[:, collumn]) == player * self.row_count) or \
            (np.sum(np.diag(state)) == player * self.row_count) or \
            (np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count)
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

        return encoded_state




