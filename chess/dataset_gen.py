from stockfish import Stockfish
import pandas as pd
from chess_game import ChessGame
import numpy as np 
import time
import random


def main():
    stockfish = Stockfish(path="./stockfish_old/stockfish-ubuntu-x86-64-avx2", depth=10, parameters={"Threads": 2})
    game = ChessGame()

    data = []
    white = True
    n_games = 0
    st = time.time()
    for i in range(1, 11):
        n_moves = 0
        state = game.get_initial_state()
        
        init_random_moves = int(random.random() * 20)
        is_terminal = False

        for j in range(0, init_random_moves):
            valid_moves = game.get_uci_valid_moves(state)

            action = random.choice(valid_moves)
            
            value, is_terminal = game.get_value_and_terminated(state, action, True)
            state = game.get_next_state(state, action)

            if is_terminal:
                break
        
    
        if is_terminal:
            continue

        while True:
            # game.show(state)

            stockfish.set_fen_position(state)
            action = stockfish.get_best_move()
            stock_value = stockfish.get_evaluation()

            data.append([state, action, stock_value["value"]])

            n_moves += 1

            value, is_terminal = game.get_value_and_terminated(state, action, True)
            state = game.get_next_state(state, action)
            
            # if n_moves % 10 == 0: 
            #     print(n_moves)

            if is_terminal:
                # if value == 0:
                #     print(f"Game Ended {int(n_moves / 2)} moves in a draw")
                # else:
                #     print(f"Game Ended with {int(n_moves / 2)} moves. White Won equals {white}")
                break
            
            white = not white
        n_games += 1
        print(f"Played {n_games} games")
        if n_games % 10 == 0:
            print("------------Saving checkpoint games----------")
            df = pd.DataFrame(data, columns=["state", "action", "value"])
            df.to_csv('dataset.csv', index=False)

    print(time.time() - st)

if __name__=="__main__":
    main()