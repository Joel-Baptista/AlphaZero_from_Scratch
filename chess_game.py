import numpy as np
import chess
import random
import math
import cv2 as cv

class ChessGame:
    def __init__(self) -> None:
        self.collumn_mapping = {
                                "a": 1,
                                "b": 2,
                                "c": 3,
                                "d": 4,
                                "e": 5,
                                "f": 6,
                                "g": 7,
                                "h": 8,
                                }
        self.knight_move_mapping = { # (x, y) where x is the horizontal squares and y the vertical sqaures
            "1": (1, 2),
            "2": (-1, 2),
            "3": (2, 1),
            "4": (-2, 1),
            "5": (-2, 1),
            "6": (-2, -1),
            "7": (1, -2),
            "8": (2, -1),
        }

    def __repr__(self) -> str:
        return "ChessGame"

    def get_initial_state(self):
        return chess.Board().fen()

    def get_next_state(self, state: str, action: str | np.array):
        
        if isinstance(action, np.array):
            print("Numpy!!!")

        board_state = chess.Board(state)
        board_state.push_uci(action)
        return board_state.fen()
    
    def get_valid_moves(self, state: str):
        board_state = chess.Board(state)
        valid_moves = board_state.legal_moves
        print(valid_moves)
        valid_moves = [str(move) for move in valid_moves]
        return valid_moves
    
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

    def get_encoded_state(self, state: str | list):
        """
        Convert board to a state that is interpretable by the model
        """

        board_state = chess.Board(state)

        # 1. is it white's turn? (1x8x8)
        is_white_turn = np.ones((8, 8)) if board_state.turn else np.zeros((8, 8))
        print(board_state.turn)
        # 2. castling rights (4x8x8)
        castling = np.asarray([
            np.ones((8, 8)) if board_state.has_queenside_castling_rights(
                chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board_state.has_kingside_castling_rights(
                chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board_state.has_queenside_castling_rights(
                chess.BLACK) else np.zeros((8, 8)),
            np.ones((8, 8)) if board_state.has_kingside_castling_rights(
                chess.BLACK) else np.zeros((8, 8)),
        ])

        # 3. repitition counter
        counter = np.ones(
            (8, 8)) if board_state.can_claim_fifty_moves() else np.zeros((8, 8))

        # create new np array
        arrays = []
        
        colors = chess.COLORS.copy()
        if not board_state.turn:
            colors.reverse()

        for color in colors:
            # 4. player 1's pieces (6x8x8)
            # 5. player 2's pieces (6x8x8)
            for piece_type in chess.PIECE_TYPES:
                # 6 arrays of 8x8 booleans
                array = np.zeros((8, 8))
                for index in list(board_state.pieces(piece_type, color)):
                    # row calculation: 7 - index/8 because we want to count from bottom left, not top left
                    array[7 - int(index/8)][index % 8] = True
                if not board_state.turn: array = np.flip(array, axis=0) # Always the prespective of the turn player
                arrays.append(array)

        arrays = np.asarray(arrays)

        # 6. en passant square (8x8)
        en_passant = np.zeros((8, 8))
        if board_state.has_legal_en_passant():
            en_passant[7 - int(board_state.ep_square/8)][board_state.ep_square % 8] = True
            if not board_state.turn: array = np.flip(array, axis=0) # Always the prespective of the turn player

        r = np.array([[is_white_turn, *castling,
                     counter, *arrays, en_passant]])
        r = np.moveaxis(r, 1, 3)
        print(np.shape(r))
        # memory management
        del board_state
        return r.astype(bool)

    def relative_movement(self, state: str, move: str):
        board_state = chess.Board(state)
        pos = move[0:2]
        relative_move = {"piece": str.lower(str(board_state.piece_at(chess.parse_square(pos)))),
                         "position": move[0:2], 
                         "direction": None,
                         "lenght": None}
        print(move)
        piece = relative_move["piece"]
        if move[0] == move[2]: # if collumn is the same
            if move[1] < move[3]:
                relative_move["direction"] = "N"
            else: 
                relative_move["direction"] = "S"
            
            relative_move["lenght"] = abs(int(move[1]) - int(move[3]))
            print("Vertical movement")
        elif move[1] == move[3]: # if line is the same 
            if self.collumn_mapping[move[0]] < self.collumn_mapping[move[2]]:
                relative_move["direction"] = "W"
            else: 
                relative_move["direction"] = "E"
            
            relative_move["lenght"] = abs(int(self.collumn_mapping[move[0]]) - int(self.collumn_mapping[move[2]]))
            print("Horizontal movement")

        elif str.lower(str(piece)) == "n":
            direction = (self.collumn_mapping[move[2]] - self.collumn_mapping[pos[0]], int(move[3]) - int(pos[1]))
            relative_move["direction"] = direction
            print("Knight movement")
        else:
            if move[1] < move[3]:
                if self.collumn_mapping[move[0]] < self.collumn_mapping[move[2]]:
                    relative_move["direction"] = "NW"
                else:
                    relative_move["direction"] = "NE"
            else: 
                if self.collumn_mapping[move[0]] < self.collumn_mapping[move[2]]:
                    relative_move["direction"] = "SW"
                else:
                    relative_move["direction"] = "SE"
            
            relative_move["lenght"] = abs(int(move[1]) - int(move[3]))
            print("Diagonal movement")

        return relative_move

if __name__=="__main__":
    game = ChessGame()

    state = game.get_initial_state()
    print(state)
    for i in range(0, 100):
        print(state)
        valid_moves = game.get_valid_moves(state)
        print(valid_moves)
        print(chess.Board(state))
        encoded_state = game.get_encoded_state(state)
        for i in range(0, 19):
            image = encoded_state[0,:,:, i]
            print(np.shape(image))
            print(image.astype(np.uint8))
            cv.imshow("state", 255 * image.astype(np.uint8))
            key = cv.waitKey()
            if key == ord("q"):
                break
        if key == ord("q"):
                break
        print("--------------------------------------")
        move = random.choice(valid_moves)
        relative_move = game.relative_movement(state, move)
        # if chess.Board(state).has_legal_en_passant():
        #     for i in range(0, 19):
        #         image = encoded_state[0,:,:, i]
        #         print(np.shape(image))
        #         print(image.astype(np.uint8))
        #         cv.imshow("state", 255 * image.astype(np.uint8))
        #         key = cv.waitKey()
        #         if key == ord("q"):
        #             break
        #     if key == ord("q"):
        #             break

        state = game.get_next_state(state, np.zeros((1,1)))
