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
        self.reverse_collumn_mapping = {
                                "1": "a",
                                "2": "b" ,
                                "3": "c",
                                "4": "d",
                                "5": "e",
                                "6": "f" ,
                                "7": "g",
                                "8": "h",
                                }
        self.knight_move_mapping = { # (x, y) where x is the horizontal squares and y the vertical sqaures
            "(1, 2)": 1,
            "(-1, 2)": 2,
            "(2, 1)": 3,
            "(-2, 1)": 4,
            "(-1, -2)": 5,
            "(-2, -1)": 6,
            "(1, -2)": 7,
            "(2, -1)": 8,
        }
        self.reverse_knight_move_mapping = {
            "1": (1, 2) ,
            "2": (-1, 2),
            "3": (2, 1),
            "4": (-2, 1) ,
            "5": (-1, -2),
            "6": (-2, -1),
            "7": (1, -2),
            "8": (2, -1),
        } 



        self.encoded_action = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Knights", "Under"]

    def __repr__(self) -> str:
        return "ChessGame"

    def get_initial_state(self):
        return chess.Board().fen()

    def get_next_state(self, state: str, action: str | np.ndarray):
        
        if isinstance(action, np.ndarray):
            print("Numpy!!!")

        board_state = chess.Board(state)
        board_state.push_uci(action)
        return board_state.fen()
    
    def decode_action(self, state: str, action: np.ndarray):
        plane = np.argmax(action, axis=2)
        print(f"plane: {plane}")
        possible_moves = np.nonzero(plane)
        print(f"possible_moves: {possible_moves}")
        if len(possible_moves[0]) > 1:
            idx_r = random.randint(0, len(np.nonzero(plane))-1)
            move_idx = plane[np.nonzero(plane)[0][idx_r]][np.nonzero(plane)[1][idx_r]]
        elif len(possible_moves[0]) == 0: # Special case, the first plane is the best move, indexed with zero
            move_idx = 0
        else: 
            move_idx = plane[np.nonzero(plane)[0][0]][np.nonzero(plane)[1][0]]

        # print(move_idx // 7)
        print(f"Direction: {self.encoded_action[move_idx // 7]}")
        index_dir = move_idx // 7
        index_len = move_idx - index_dir * 7 + 1
        print(f"Len: {index_len}")
        move = self.encoded_action[index_dir]
        print(f"Max actions: {action[:,:, move_idx]}")
        possible_positions = np.argmax(action[:,:, move_idx])
        print(f"possible_positions: {possible_positions}")
        c = possible_positions % 8 + 1
        l = 8 - possible_positions // 8

        print(f"c: {c}")
        print(f"l:{l}")

        pos = f"{self.reverse_collumn_mapping[str(c)]}{str(l)}"
        print(f"pos: {pos}")

        # board_state = chess.Board(state)
        # print(board_state)
        # piece = str.lower(str(board_state.piece_at(chess.parse_square(pos))))
        # print(piece)
        # print(move)
        if move == "Knights":
            knight_jump = self.reverse_knight_move_mapping[str(index_len)]
            print(knight_jump)
            new_pos = f"{self.reverse_collumn_mapping[str(c + knight_jump[0])]}{l + knight_jump[1]}"
            # print(f"New Pos: {new_pos}")
        elif move == "N":
            new_pos = f"{self.reverse_collumn_mapping[str(c)]}{l + index_len}"
        elif move == "S":
            new_pos = f"{self.reverse_collumn_mapping[str(c)]}{l - index_len}"
        elif move == "W":
            new_pos = f"{self.reverse_collumn_mapping[str(c - index_len)]}{l}"
        elif move == "E":
            new_pos = f"{self.reverse_collumn_mapping[str(c + index_len)]}{l}"
        elif move == "NE":
            new_pos = f"{self.reverse_collumn_mapping[str(c + index_len)]}{l + index_len}"
        elif move == "NW":
            new_pos = f"{self.reverse_collumn_mapping[str(c - index_len)]}{l + index_len}"
        elif move == "SW":
            new_pos = f"{self.reverse_collumn_mapping[str(c - index_len)]}{l - index_len}"
        elif move == "SE":
            new_pos = f"{self.reverse_collumn_mapping[str(c + index_len)]}{l - index_len}"

        print(f"{pos}{new_pos}")
        return f"{pos}{new_pos}"


    def get_uci_valid_moves(self, state: str):
        board_state = chess.Board(state)
        valid_moves = board_state.legal_moves
        print(valid_moves)
        valid_moves = [str(move) for move in valid_moves]
        return valid_moves
    
    def get_valid_moves(self, state: str):
        board_state = chess.Board(state)
        valid_moves = board_state.legal_moves
        # print(valid_moves)
        valid_moves = [str(move) for move in valid_moves]
        relative_valid_moves = [self.relative_movement(state, move) for move in valid_moves]
        # print(relative_valid_moves)
        encoded_valid_moves = np.zeros((1, 8, 8, 73))
        #TODO Add special move of underpromoting
        for move in relative_valid_moves:
            # print(move["direction"])
            # print(move)
            if isinstance(move["direction"], tuple):
                index_dir = self.encoded_action.index("Knights")
                index_len = self.knight_move_mapping[str(move["direction"])]
            else:
                index_dir = self.encoded_action.index(move["direction"])
                index_len = move["lenght"]

            index_frame = index_dir * 7 + index_len - 1
            l = int(move["position"][1])
            c = move["position"][0]
            # print(index_frame)
            # print(l)
            # print(c)
            # print(self.collumn_mapping[c])
            # print(encoded_valid_moves.shape)
            # print(encoded_valid_moves[0, 1, 7, 56])
            encoded_valid_moves[0, 7 - (l - 1), 7 - (self.collumn_mapping[c] - 1), index_frame] = 1

        return encoded_valid_moves

    def check_win(self, state, action):
        if action is None:
            return False

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
        # print(board_state.turn)
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
        # print(np.shape(r))
        # memory management
        del board_state
        return r.astype(bool)

    def relative_movement(self, state: str, move: str):
        #TODO Add special move of underpromoting
        board_state = chess.Board(state)
        pos = move[0:2]
        relative_move = {"piece": str.lower(str(board_state.piece_at(chess.parse_square(pos)))),
                         "position": move[0:2], 
                         "direction": None,
                         "lenght": None}
        # print(move)
        piece = relative_move["piece"]
        if move[0] == move[2]: # if collumn is the same
            if move[1] < move[3]:
                relative_move["direction"] = "N"
            else: 
                relative_move["direction"] = "S"
            
            relative_move["lenght"] = abs(int(move[1]) - int(move[3]))
            # print("Vertical movement")
        elif move[1] == move[3]: # if line is the same 
            if self.collumn_mapping[move[0]] < self.collumn_mapping[move[2]]:
                relative_move["direction"] = "W"
            else: 
                relative_move["direction"] = "E"
            
            relative_move["lenght"] = abs(int(self.collumn_mapping[move[0]]) - int(self.collumn_mapping[move[2]]))
            # print("Horizontal movement")

        elif str.lower(str(piece)) == "n":
            direction = (self.collumn_mapping[move[2]] - self.collumn_mapping[pos[0]], int(move[3]) - int(pos[1]))
            relative_move["direction"] = direction
            # print("Knight movement")
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
            # print("Diagonal movement")

        return relative_move

if __name__=="__main__":
    game = ChessGame()

    state = game.get_initial_state()
    print(state)
    for i in range(0, 100):
        print(state)
        valid_moves = game.get_uci_valid_moves(state)
        print(state)
        # print(valid_moves)
        # print(chess.Board(state))
        encoded_state = game.get_encoded_state(state)
        encoded_actions = game.get_valid_moves(state)
        decoded_action = game.decode_action(state, encoded_actions[0])
        # for i in range(0, 73):
        #     image = encoded_actions[0,:,:, i]
        #     print(np.shape(image))
        #     print(image.astype(np.uint8))
        #     cv.imshow("state", 255 * image.astype(np.uint8))
        #     key = cv.waitKey()
        #     if key == ord("q"):
        #         break
        # if key == ord("q"):
        #         break
        print("--------------------------------------")
        move = random.choice(valid_moves)
        relative_move = game.relative_movement(state, move)
        game.get_valid_moves(state)
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

        state = game.get_next_state(state, move)
