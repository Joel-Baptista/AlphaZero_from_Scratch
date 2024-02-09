import numpy as np
import math
import random
from chess_game import ChessGame

class MCTS:
    def __init__(self, game: ChessGame, args) -> None:
        self.game = game 
        self.args = args

    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminated = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminated:
                node = node.expand()
                value = node.simulate()
                
                node.backpropagate(value)
            
        valid_actions = self.game.get_uci_valid_moves(state)
        action_probs = np.zeros(len(valid_actions))
        for child in root.children:
            action_probs[valid_actions.index(child.action_taken)] = child.visit_count

        action_probs /= np.sum(action_probs)
        return valid_actions, action_probs


class Node:
    def __init__(self, game: ChessGame, args, state, parent=None, action_taken=None) -> None:
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_moves = game.get_uci_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.expandable_moves) == 0 and len(self.children) > 0 
        # return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child
    
    def get_ucb(self, child):
        # The next state is for our opponent, so we want to minimize the value of the next state
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args["C"] * math.sqrt(math.log(self.visit_count) / child.visit_count)


    def expand(self):

        action = random.choice(self.expandable_moves)
        self.expandable_moves.pop(self.expandable_moves.index(action))

        child_state = self.state
        child_state = self.game.get_next_state(child_state, action)
        
        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)

        return child
    
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)

        if is_terminal:
            return value
        
        rollout_state = self.state
        rollout_player = 1
        i = 0
        while True:
            valid_moves = self.game.get_uci_valid_moves(rollout_state)
            action = random.choice(valid_moves)
            # print(action)
            rollout_state = self.game.get_next_state(rollout_state, action)
        
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            i += 1
            if is_terminal or i > 20:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)

                return value
            
            rollout_player = self.game.get_opponent(rollout_player)
        
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)



