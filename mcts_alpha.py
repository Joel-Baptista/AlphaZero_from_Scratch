import numpy as np
import math
import torch
import time


class MCTSParallel:
    def __init__(self, game, args, model) -> None:
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        st = time.time()
        policy, _ = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args["dirichlet_epsilon"] \
            * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size, size=policy.shape[0])
        
        # print(f"Inference time: {time.time() - st}")

        # policy = np.random.random((250, 7))
        # print("Root Random Policy!")

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count = 1)

            spg.root.expand(spg_policy)
        # print("Roots Expanded!")
        total_inference_time = 0
        total_mapping_time = 0
        
        for search in range(self.args['num_searches']):
            st = time.time()
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminated = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)
                
                if is_terminated:    
                    node.backpropagate(value)
                else:
                    spg.node = node
            a = time.time() - st
            # print(f"Fully expanand in {a}")
            st = time.time()
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]

            st = time.time()
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                # policy = np.random.random((250, 7))
                # value = np.random.random(250)
                # print("Expanded Random Policy!")
            # b = time.time() - st
            # print(f"Fully infered in {b}")
            total_inference_time += time.time() - st
            st = time.time()
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy = policy[i]
                spg_value = value[i]

                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                spg_value = spg_value.item()
                
                node.expand(spg_policy)

                node.backpropagate(spg_value)
            total_mapping_time += time.time() - st
            # print(f"Search {search} Complet")
            # c = time.time() - st
            # print(f"Fully mapped in {c}")
            # print(f"Total search time: {a+b+c}")
            # print("---------------------------------------")

        # print(f"Average inference time: {total_inference_time / search}")
        # print(f"Average mapping time: {total_mapping_time / search}")
        # print("-------------------------------------------")
        # action_probs = np.zeros(self.game.action_size)
        # for child in root.children:
        #     action_probs[child.action_taken] = child.visit_count

        # action_probs /= np.sum(action_probs)
        # return action_probs


class MCTS:
    def __init__(self, game, args, model) -> None:
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count = 1)

        policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
                )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args["dirichlet_epsilon"] \
            * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)

        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminated = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminated:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()
                
                node.expand(policy)

            node.backpropagate(value)
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count = 0) -> None:
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
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_prespective(child_state, -1)
                
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
        
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)



