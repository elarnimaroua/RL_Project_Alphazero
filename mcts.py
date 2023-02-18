import math

import numpy as np

from copy import deepcopy


class TreeNode(object):
    """for each node we keep track of its value Q, 
    prior proba P and the visit count adjusted prior score u"""
    

    def __init__(self, parent=None, action=None, psa=0.0, child_psas=[]):
        """Initializes TreeNode with the initial statistics and data."""
        self.Nsa = 0
        self.Wsa = 0.0
        self.Qsa = 0.0
        self.Psa = psa
        self.action = action
        self.children = []
        self.child_psas = child_psas
        self.parent = parent
        self.c_puct = 1
        self.dirichlet_alpha = 0.5
        self.num_mcts_sims = 30
        self.epsilon = 0.25

    def is_not_leaf(self):
        """Check if the node is a leaf"""
        if len(self.children) > 0:
            return True
        return False

    def select_child(self):
        """Select child using upper confidence bound"""
        c_puct = self.c_puct

        highest_uct = 0
        highest_index = 0

        for idx, child in enumerate(self.children):
            uct = child.Qsa + child.Psa * c_puct * (
                    math.sqrt(self.Nsa) / (1 + child.Nsa))
            if uct > highest_uct:
                highest_uct = uct
                highest_index = idx

        return self.children[highest_index]

    def expand_node(self, game, psa_vector):
        """Expands the current node by adding valid moves as children."""
        self.child_psas = deepcopy(psa_vector)
        valid_moves = game.get_valid_moves(game.current_player)
        for idx, move in enumerate(valid_moves):
            if move[0] is not 0:
                action = deepcopy(move)
                self.add_child_node(parent=self, action=action,
                                    psa=psa_vector[idx])

    def add_child_node(self, parent, action, psa=0.0):
        """Create a new child"""

        child_node = TreeNode(parent=parent, action=action, psa=psa)
        self.children.append(child_node)
        return child_node

    def back_prop(self, wsa, v):
        """Updating node values """
        self.Nsa += 1
        self.Wsa = wsa + v
        self.Qsa = self.Wsa / self.Nsa


class MonteCarloTreeSearch(object):
    """Monte Carlo Tree Search Algorithm."""

    def __init__(self, net):
        """Initializes TreeNode, board and neural network."""
        self.root = None
        self.game = None
        self.net = net
        self.c_puct = 1
        self.num_mcts_sims = 100
        self.epsilon = 0.25
        self.dirichlet_alpha = 0.5

    def search(self, game, node, temperature):
        """Return the best move to be played"""
        self.root = node
        self.game = game

        for i in range(self.num_mcts_sims):
            node = self.root
            game = self.game.clone()  # clone game state for each loop 

            while node.is_not_leaf(): #not a leaf select child using upper confidence bound
                node = node.select_child()
                game.play_action(node.action)

            psa_vector, v = self.net.predict(game.state) #proba and value of current state

            # Add Dirichlet noise to the psa_vector of the root node.
            if node.parent is None:
                psa_vector = self.add_dirichlet_noise(game, psa_vector)

            valid_moves = game.get_valid_moves(game.current_player)
            for idx, move in enumerate(valid_moves):
                if move[0] is 0:
                    psa_vector[idx] = 0

            psa_vector_sum = sum(psa_vector)

            # Renormalize psa vector
            if psa_vector_sum > 0:
                psa_vector /= psa_vector_sum

           
            node.expand_node(game=game, psa_vector=psa_vector) #next node

            game_over, wsa = game.check_game_over(game.current_player)

            
            while node is not None: #back propagate to root
                wsa = -wsa
                v = -v
                node.back_prop(wsa, v)
                node = node.parent

        highest_nsa = 0
        highest_index = 0

        # select child's move using temperature
        for idx, child in enumerate(self.root.children):
            temperature_exponent = int(1 / temperature)

            if child.Nsa ** temperature_exponent > highest_nsa:
                highest_nsa = child.Nsa ** temperature_exponent
                highest_index = idx

        return self.root.children[highest_index]

    def add_dirichlet_noise(self, game, psa_vector):
        """Add Dirichlet noise to the psa_vector of the root node."""
        dirichlet_input = [self.dirichlet_alpha for x in range(game.row * game.column)]

        dirichlet_list = np.random.dirichlet(dirichlet_input)
        noisy_psa_vector = []

        for idx, psa in enumerate(psa_vector):
            noisy_psa_vector.append(
                (1 - self.epsilon) * psa + self.epsilon * dirichlet_list[idx])

        return noisy_psa_vector
