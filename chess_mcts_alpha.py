import numpy as np
import math
import torch
import time
from chess_game import ChessGame
from model import ResNet
import copy
from multiprocessing import Process
import multiprocessing


class MCTSParallel:
    def __init__(self, game: ChessGame, args, model: ResNet) -> None:
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        st = time.time()
        policy, value = self.model(
                    torch.tensor(
                        np.array([self.game.get_encoded_state(state).squeeze(0) for state in states]), # I love python :) 
                        device=self.model.device,
                        dtype=torch.float32
                                 )
                )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        print(f"Inference time: {time.time() - st}")
        st = time.time()
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args["dirichlet_epsilon"] \
            * np.random.dirichlet(alpha=self.args["dirichlet_alpha"] * np.ones(8*8*73)).reshape((-1, 8, 8, 73))

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            
            valid_moves = self.game.get_valid_moves(states[i]).squeeze(0)
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count = 1)

            spg.root.expand(spg_policy)
        # print("Roots Expanded!")
        print(f"Expand roots time: {time.time() - st}")

        for search in range(self.args['num_searches']):
            st_total = time.time()
            st = time.time()
            for spg in spGames:
                spg.node = None
                node = spg.root 
                node: Node

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminated = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminated:    
                    node.backpropagate(value)
                else:
                    spg.node = node

            print(f"Expand all games time: {time.time() - st}")
            st = time.time()
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
            
            print(f"Check expandable games time: {time.time() - st}")
            st = time.time()

            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                policy, value = self.model(
                    torch.tensor(
                        np.array([self.game.get_encoded_state(state).squeeze(0) for state in states]), 
                        device=self.model.device,
                        dtype=torch.float32
                                 )
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.detach().cpu().numpy()

            print(f"Inference time: {time.time() - st}")
            st = time.time()
            manager = multiprocessing.Manager()
            manager_dict = manager.dict()

            processes = []
            num_processess = 10
            delta_idx = len(expandable_spGames) // num_processess
            st_idx = 0
            for i in range(1, num_processess + 1):
                end_idx = st_idx + delta_idx
                if end_idx > len(expandable_spGames): end_idx == len(expandable_spGames)
                
                nodes = [spGames[mappingIdx].node for mappingIdx in expandable_spGames[st_idx:end_idx]]
                manager_dict[f"proc_{i}"] = nodes
                processes.append(Process(
                    target=self.parallel_expand_games, 
                    args=(manager_dict, policy[st_idx:end_idx], value[st_idx:end_idx], i)))
                processes[-1].start()
                st_idx += delta_idx

            for thread in processes:
                thread.join()

            st_sort = time.time()
            st_idx = 0
            for i in range(1, num_processess + 1):
                end_idx = st_idx + delta_idx
                if end_idx > len(expandable_spGames): end_idx == len(expandable_spGames)
                for j, mappingIdx in enumerate(expandable_spGames[st_idx:end_idx]):
                    spGames[mappingIdx].node = manager_dict[f"proc_{i}"][j]

                st_idx += delta_idx

            print(f"Sort threads time: {time.time() - st_sort}")
            # self.parallel_expand_games(expandable_spGames, spGames,policy, value)
            # for i, mappingIdx in enumerate(expandable_spGames):
            #     st2 = time.time()
            #     node = spGames[mappingIdx].node
            #     spg_policy = policy[i]
            #     spg_value = value[i]

            #     # print(f"Mapping time: {time.time() - st2}")
            #     # st2 = time.time()

            #     valid_moves = self.game.get_valid_moves(node.state).squeeze(0)
                
            #     # print(f"Get Valid moves time: {time.time() - st2}")
            #     # st2 = time.time()
                
            #     spg_policy *= valid_moves
            #     spg_policy /= np.sum(spg_policy)

            #     # print(f"Filter Valid Moves time: {time.time() - st2}")
            #     # st2 = time.time()

            #     spg_value = spg_value.item()
                
            #     node.expand(spg_policy)

            #     # print(f"Expand time: {time.time() - st2}")
            #     # st2 = time.time()

            #     node.backpropagate(spg_value)

            #     # print(f"Backpropagate time: {time.time() - st2}")
            #     # st2 = time.time()

            #     # print("==========================================================")

            print(f"Backpropagate all games time: {time.time() - st}")
            st = time.time()
            print(f"Total Time: {time.time() - st_total}")
            print("-----------------------------------------------------------------")


    def parallel_expand_games(self, manager_dict, policy, value, process_num):
        for i, node in enumerate(manager_dict[f"proc_{process_num}"]):
            st2 = time.time()

            spg_policy = policy[i]
            spg_value = value[i]

            # print(f"Mapping time: {time.time() - st2}")
            # st2 = time.time()

            valid_moves = self.game.get_valid_moves(node.state).squeeze(0)
            
            # print(f"Get Valid moves time: {time.time() - st2}")
            # st2 = time.time()
            
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            # print(f"Filter Valid Moves time: {time.time() - st2}")
            # st2 = time.time()
            
            node.expand(spg_policy)

            # print(f"Expand time: {time.time() - st2}")
            # st2 = time.time()

            node.backpropagate(spg_value)

            manager_dict[f"proc_{process_num}"][i] = node
            # print(f"Backpropagate time: {time.time() - st2}")
            # st2 = time.time()

            # print("==========================================================")

class MCTS:
    def __init__(self, game: ChessGame, args, model) -> None:
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count = 1)

        policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(state), 
                        device=self.model.device,
                        dtype=torch.float32
                                 )
                )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args["dirichlet_epsilon"] \
            * np.random.dirichlet(alpha=self.args["dirichlet_alpha"] * np.ones(8*8*73)).reshape((8, 8, 73))
        
        valid_moves = self.game.get_valid_moves(state).squeeze(0)
    
        policy *= valid_moves
        policy /= np.sum(policy)

        root.expand(policy)

        for search in range(self.args['num_searches']):
            print(f"search: {search}")
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminated = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminated:
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(node.state), 
                        device=self.model.device,
                        dtype=torch.float32
                                 )
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state).squeeze(0)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()
                
                node.expand(policy)

            node.backpropagate(value)
            
            
        valid_actions = self.game.get_uci_valid_moves(state)
        action_probs = np.zeros(len(valid_actions))
        for child in root.children:
            action_probs[valid_actions.index(child.action_taken)] = child.visit_count

        action_probs /= np.sum(action_probs)
        return valid_actions, action_probs

class Node:
    def __init__(self, game: ChessGame, args, state, parent=None, action_taken=None, prior=0, visit_count = 0) -> None:
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior 

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

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
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        
        return q_value +  child.prior * self.args["C"] * math.sqrt(self.visit_count / (1 + child.visit_count))


    def expand(self, policy): 

        all_policy = self.game.decode_all_actions(policy, self.state)

        for i, policy in enumerate(all_policy):
            
            action = policy[0]
            prob = policy[1]
            
            child_state = copy.deepcopy(self.state)
            
            child_state = self.game.get_next_state(child_state, action)
            
            child = Node(self.game, self.args, child_state, self, action, prob)
            self.children.append(child)


    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


