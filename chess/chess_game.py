import numpy as np
import chess
import random
from colorama import Fore
import functools
import time

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
    
    def get_board_state(self, state: str | chess.Board):
        
        if isinstance(state, chess.Board):
            return state

        return chess.Board(state)
    
    def get_fen(self, state: chess.Board):
        
        return state.fen()
    
    def copy_board(self, state: chess.Board):
        return chess.Board(fen=state.fen())


    def get_initial_state(self):
        return chess.Board().fen()
    
    def show(self, state):
        print(chess.Board(state))

    def get_next_state(self, state: str | chess.Board, action: str | np.ndarray):
        
        if isinstance(action, np.ndarray):
            print("Numpy!!!")

        if isinstance(state, chess.Board):
            state.push_uci(action)
            return state
            
        board_state = chess.Board(state)
        board_state.push_uci(action)

        return board_state.fen()
    
    # @functools.lru_cache(maxsize=10_000)
    def decode_all_actions(self, action: np.ndarray, state: list[str]):

        st = time.time()
        non_zero_idx = np.nonzero(action)
        non_zero_values = action[non_zero_idx]
        non_zero_coordinates = np.transpose(non_zero_idx)

        # print(time.time()-st)
        # print(non_zero_values)
        # print(non_zero_coordinates)

        decoded_action = []
        for idx, coordiantes in enumerate(non_zero_coordinates):
            i, j ,k = coordiantes
            move_idx = k

            index_dir = move_idx // 7
            if move_idx <= 55:
                direction = self.encoded_action[move_idx // 7]
            elif 56 <= move_idx <= 63:
                direction = "Knights"
                index_dir = 8
            else:
                direction = "Under"
            
            index_len = move_idx - index_dir * 7 + 1

            c = j + 1
            l = 8 - i

            pos = f"{self.reverse_collumn_mapping[str(c)]}{str(l)}"

            if direction == "Knights":
                knight_jump = self.reverse_knight_move_mapping[str(index_len)]
                new_pos = f"{self.reverse_collumn_mapping[str(c + knight_jump[0])]}{l + knight_jump[1]}"
            elif direction == "N":
                new_pos = f"{self.reverse_collumn_mapping[str(c)]}{l + index_len}"
            elif direction == "S":
                new_pos = f"{self.reverse_collumn_mapping[str(c)]}{l - index_len}"
            elif direction == "W":
                new_pos = f"{self.reverse_collumn_mapping[str(c - index_len)]}{l}"
            elif direction == "E":
                new_pos = f"{self.reverse_collumn_mapping[str(c + index_len)]}{l}"
            elif direction == "NE":
                new_pos = f"{self.reverse_collumn_mapping[str(c + index_len)]}{l + index_len}"
            elif direction == "NW":
                new_pos = f"{self.reverse_collumn_mapping[str(c - index_len)]}{l + index_len}"
            elif direction == "SW":
                new_pos = f"{self.reverse_collumn_mapping[str(c - index_len)]}{l - index_len}"
            elif direction == "SE":
                new_pos = f"{self.reverse_collumn_mapping[str(c + index_len)]}{l - index_len}"
            elif direction == "Under":
                print(f"{Fore.YELLOW}Under Promotion is not yet implemented: move_idx is {move_idx}{Fore.RESET}")

            promotion = ""
            board_state = chess.Board(state)
            piece = str.lower(str(board_state.piece_at(chess.parse_square(pos))))
            if (int(new_pos[1]) == 8 or int(new_pos[1]) == 1) and piece == "p":
                promotion = "q" #TODO Introduce underpromotions

            decoded_action.append((f"{pos}{new_pos}{promotion}", non_zero_values[idx]))

        # decoded_action = []
        # for i in range(action.shape[0]):
        #     for j in range(action.shape[1]):
        #         for k in range(action.shape[2]):
        #             probability = action[i, j, k]
        #             if probability > 0:
        #                 move_idx = k


        #                 index_dir = move_idx // 7
        #                 if move_idx <= 55:
        #                     direction = self.encoded_action[move_idx // 7]
        #                 elif 56 <= move_idx <= 63:
        #                     direction = "Knights"
        #                     index_dir = 8
        #                 else:
        #                     direction = "Under"
                        
        #                 index_len = move_idx - index_dir * 7 + 1

        #                 c = j + 1
        #                 l = 8 - i

        #                 pos = f"{self.reverse_collumn_mapping[str(c)]}{str(l)}"

        #                 if direction == "Knights":
        #                     knight_jump = self.reverse_knight_move_mapping[str(index_len)]
        #                     new_pos = f"{self.reverse_collumn_mapping[str(c + knight_jump[0])]}{l + knight_jump[1]}"
        #                 elif direction == "N":
        #                     new_pos = f"{self.reverse_collumn_mapping[str(c)]}{l + index_len}"
        #                 elif direction == "S":
        #                     new_pos = f"{self.reverse_collumn_mapping[str(c)]}{l - index_len}"
        #                 elif direction == "W":
        #                     new_pos = f"{self.reverse_collumn_mapping[str(c - index_len)]}{l}"
        #                 elif direction == "E":
        #                     new_pos = f"{self.reverse_collumn_mapping[str(c + index_len)]}{l}"
        #                 elif direction == "NE":
        #                     new_pos = f"{self.reverse_collumn_mapping[str(c + index_len)]}{l + index_len}"
        #                 elif direction == "NW":
        #                     new_pos = f"{self.reverse_collumn_mapping[str(c - index_len)]}{l + index_len}"
        #                 elif direction == "SW":
        #                     new_pos = f"{self.reverse_collumn_mapping[str(c - index_len)]}{l - index_len}"
        #                 elif direction == "SE":
        #                     new_pos = f"{self.reverse_collumn_mapping[str(c + index_len)]}{l - index_len}"
        #                 elif direction == "Under":
        #                     print(f"{Fore.YELLOW}Under Promotion is not yet implemented: move_idx is {move_idx}{Fore.RESET}")

        #                 promotion = ""
        #                 board_state = chess.Board(state)
        #                 piece = str.lower(str(board_state.piece_at(chess.parse_square(pos))))
        #                 if (int(new_pos[1]) == 8 or int(new_pos[1]) == 1) and piece == "p":
        #                     promotion = "q" #TODO Introduce underpromotions

        #                 decoded_action.append((f"{pos}{new_pos}{promotion}", probability))

        return decoded_action

    def decode_all_actions_template(self, action: np.ndarray, state: list[str]):

        reverse_collumn_mapping = {
                                "1": "a",
                                "2": "b" ,
                                "3": "c",
                                "4": "d",
                                "5": "e",
                                "6": "f" ,
                                "7": "g",
                                "8": "h",
                                }

        reverse_knight_move_mapping = {
            "1": (1, 2) ,
            "2": (-1, 2),
            "3": (2, 1),
            "4": (-2, 1) ,
            "5": (-1, -2),
            "6": (-2, -1),
            "7": (1, -2),
            "8": (2, -1),
        } 
        
        encoded_action = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Knights", "Under"]

        st = time.time()
        non_zero_idx = np.nonzero(action)
        non_zero_values = action[non_zero_idx]
        non_zero_coordinates = np.transpose(non_zero_idx)

        decoded_action = []
        for idx, coordiantes in enumerate(non_zero_coordinates):
            i, j ,k = coordiantes
            move_idx = k

            index_dir = move_idx // 7
            if move_idx <= 55:
                direction = encoded_action[move_idx // 7]
            elif 56 <= move_idx <= 63:
                direction = "Knights"
                index_dir = 8
            else:
                direction = "Under"
            
            index_len = move_idx - index_dir * 7 + 1

            c = j + 1
            l = 8 - i

            pos = f"{reverse_collumn_mapping[str(c)]}{str(l)}"

            if direction == "Knights":
                knight_jump = reverse_knight_move_mapping[str(index_len)]
                new_pos = f"{reverse_collumn_mapping[str(c + knight_jump[0])]}{l + knight_jump[1]}"
            elif direction == "N":
                new_pos = f"{reverse_collumn_mapping[str(c)]}{l + index_len}"
            elif direction == "S":
                new_pos = f"{reverse_collumn_mapping[str(c)]}{l - index_len}"
            elif direction == "W":
                new_pos = f"{reverse_collumn_mapping[str(c - index_len)]}{l}"
            elif direction == "E":
                new_pos = f"{reverse_collumn_mapping[str(c + index_len)]}{l}"
            elif direction == "NE":
                new_pos = f"{reverse_collumn_mapping[str(c + index_len)]}{l + index_len}"
            elif direction == "NW":
                new_pos = f"{reverse_collumn_mapping[str(c - index_len)]}{l + index_len}"
            elif direction == "SW":
                new_pos = f"{reverse_collumn_mapping[str(c - index_len)]}{l - index_len}"
            elif direction == "SE":
                new_pos = f"{reverse_collumn_mapping[str(c + index_len)]}{l - index_len}"
            elif direction == "Under":
                print(f"{Fore.YELLOW}Under Promotion is not yet implemented: move_idx is {move_idx}{Fore.RESET}")

            promotion = ""
            board_state = chess.Board(state)
            piece = str.lower(str(board_state.piece_at(chess.parse_square(pos))))
            if (int(new_pos[1]) == 8 or int(new_pos[1]) == 1) and piece == "p":
                promotion = "q" #TODO Introduce underpromotions

            decoded_action.append((f"{pos}{new_pos}{promotion}", non_zero_values[idx]))
        return decoded_action


    def decode_action(self, actions: np.ndarray, state: str | chess.Board):
        decoded_actions = []

        for action in actions:
            # print(action.shape)
            plane = np.argmax(action, axis=2)
            # print(f"plane: {plane}")
            possible_moves = np.nonzero(plane)
            # print(f"possible_moves: {possible_moves}")
            if len(possible_moves[0]) > 1:
                idx_r = random.randint(0, len(np.nonzero(plane))-1)
                move_idx = plane[np.nonzero(plane)[0][idx_r]][np.nonzero(plane)[1][idx_r]]
            elif len(possible_moves[0]) == 0: # Special case, the first plane is the best move, indexed with zero
                move_idx = 0
            else: 
                move_idx = plane[np.nonzero(plane)[0][0]][np.nonzero(plane)[1][0]]

            index_dir = move_idx // 7
            if move_idx <= 55:
                move = self.encoded_action[move_idx // 7]
            elif 56 <= move_idx <= 63:
                move = "Knights"
                index_dir = 8
            else:
                move = "Under"
            

            # print(move_idx // 7)
            # print(f"Direction: {self.encoded_action[move_idx // 7]}")
            index_len = move_idx - index_dir * 7 + 1
            # print(f"Len: {index_len}")
            move = self.encoded_action[index_dir]
            # print(f"Max actions: {action[:,:, move_idx]}")
            possible_positions = np.argmax(action[:,:, move_idx])
            # print(f"possible_positions: {possible_positions}")
            c = possible_positions % 8 + 1
            l = 8 - possible_positions // 8

            # print(f"c: {c}")
            # print(f"l:{l}")

            pos = f"{self.reverse_collumn_mapping[str(c)]}{str(l)}"
            # print(f"pos: {pos}")

            if move == "Knights":
                knight_jump = self.reverse_knight_move_mapping[str(index_len)]
                # print(knight_jump)
                new_pos = f"{self.reverse_collumn_mapping[str(c + knight_jump[0])]}{l + knight_jump[1]}"

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
            elif move == "Under":
                        print(f"{Fore.YELLOW}Under Promotion is not yet implemented: move_idx is {move_idx}{Fore.RESET}")

            promotion = ""
            board_state = chess.Board(state)
            piece = str.lower(str(board_state.piece_at(chess.parse_square(pos))))
            if (int(new_pos[1]) == 8 or int(new_pos[1]) == 1) and piece == "p":
                promotion = "q" #TODO Introduce underpromotions


            decoded_actions.append(f"{pos}{new_pos}{promotion}")
            # print(f"{pos}{new_pos}")
        return decoded_actions


    def get_uci_valid_moves(self, state: str | chess.Board):

        if isinstance(state, str):
            board_state = chess.Board(state)
        elif isinstance(state, chess.Board):
            board_state = state

        valid_moves = board_state.legal_moves

        valid_moves = [str(move) for move in valid_moves]
        return valid_moves
    
    @functools.lru_cache(maxsize=10_000)
    def get_valid_moves(self, state: str):
        valid_moves = self.get_uci_valid_moves(state)
        relative_valid_moves = [self.relative_movement(state, move) for move in valid_moves]

        encoded_valid_moves = np.zeros((1, 8, 8, 73))
        #TODO Add special move of underpromoting
        for move in relative_valid_moves:

            if isinstance(move["direction"], tuple):
                index_dir = self.encoded_action.index("Knights")
                index_len = self.knight_move_mapping[str(move["direction"])]
            else:
                index_dir = self.encoded_action.index(move["direction"])
                index_len = move["lenght"]

            index_frame = index_dir * 7 + index_len - 1
            l = int(move["position"][1])
            c = move["position"][0]

            encoded_valid_moves[0, 7 - (l - 1), self.collumn_mapping[c] - 1, index_frame] = 1

        return encoded_valid_moves
    
    def encode_action(self, state: str, action: str):

        relative_move = self.relative_movement(state, action)
        encoded_action = np.zeros((1, 8, 8, 73))

        if isinstance(relative_move["direction"], tuple):
                index_dir = self.encoded_action.index("Knights")
                index_len = self.knight_move_mapping[str(relative_move["direction"])]
        else:
            index_dir = self.encoded_action.index(relative_move["direction"])
            index_len = relative_move["lenght"]

        index_frame = index_dir * 7 + index_len - 1
        l = int(relative_move["position"][1])
        c = relative_move["position"][0]

        encoded_action[0, 7 - (l - 1), self.collumn_mapping[c] - 1, index_frame] = 1

        return encoded_action

    def check_win(self, state, action):
        state = self.get_next_state(state, action)
        return chess.Board(state).is_checkmate()
    

        

    def get_value_and_terminated(self, state, action, previous_state=False):
    
        if previous_state and action is not None:
            state = self.get_next_state(state, action)
        
        if isinstance(state, chess.Board):
            board_state = state
        else:
            board_state = chess.Board(state)

        if board_state.is_checkmate():
            return 1, True
        
        if board_state.is_game_over(claim_draw=True):
            return 0, True

        return 0, False

    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value

    # def change_prespective(self, state, player):
    #     return state * player

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
        # r = np.moveaxis(r, 1, 3)
        # print(np.shape(r))
        # memory management
        del board_state
        return r.astype(float)

    def relative_movement(self, state: str, move: str):

        #TODO Add special move of underpromoting
        board_state = chess.Board(state)
        pos = move[0:2]
        relative_move = {"piece": str.lower(str(board_state.piece_at(chess.parse_square(pos)))),
                         "position": move[0:2], 
                         "direction": None,
                         "lenght": None,
                         "promote": None}
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
                relative_move["direction"] = "E"
            else: 
                relative_move["direction"] = "W"
            
            relative_move["lenght"] = abs(int(self.collumn_mapping[move[0]]) - int(self.collumn_mapping[move[2]]))
            # print("Horizontal movement")

        elif str.lower(str(piece)) == "n":
            direction = (self.collumn_mapping[move[2]] - self.collumn_mapping[pos[0]], int(move[3]) - int(pos[1]))
            relative_move["direction"] = direction
            # print("Knight movement")
        else:
            if move[1] < move[3]:
                if self.collumn_mapping[move[0]] < self.collumn_mapping[move[2]]:
                    relative_move["direction"] = "NE"
                else:
                    relative_move["direction"] = "NW"
            else: 
                if self.collumn_mapping[move[0]] < self.collumn_mapping[move[2]]:
                    relative_move["direction"] = "SE"
                else:
                    relative_move["direction"] = "SW"
 
            relative_move["lenght"] = abs(int(move[1]) - int(move[3]))
            # print("Diagonal movement")

        if str.lower(str(piece)) == "p" and (int(move[3]) == 8 or int(move[3]) == 1):
            relative_move["promote"] = move[4]

        return relative_move

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

# if __name__=="__main__":
#     game = ChessGame()

#     model = ResNetChess(game, args["num_resblocks"], args["num_hidden"], "cpu")
#     state = game.get_initial_state()
#     print(state)
#     for i in range(0, 100):
#         print(state)
#         valid_moves = game.get_uci_valid_moves(state)
#         print(state)
#         # print(valid_moves)
#         # print(chess.Board(state))
#         encoded_state = game.get_encoded_state(state)

#         policy, value = model(torch.Tensor(encoded_state))
#         policy = torch.softmax(policy, axis=1).detach().cpu().numpy()
#         valid_moves = game.get_valid_moves(state)
#         policy *= valid_moves
#         policy /= np.sum(policy)
        
#         print(game.decode_action(policy))
        # print(encoded_state.shape)
        # encoded_actions = game.get_valid_moves(state)
        # decoded_action = game.decode_action(state, encoded_actions[0])
        # print(encoded_actions.shape)
        # break
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
        # print("--------------------------------------")
        # move = random.choice(valid_moves)
        # relative_move = game.relative_movement(state, move)
        # game.get_valid_moves(state)
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

        # state = game.get_next_state(state, move)
