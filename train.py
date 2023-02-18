from mcts import MonteCarloTreeSearch, TreeNode
from neural_net_wrapper import NeuralNetworkWrapper
from evaluate import Evaluate
from copy import deepcopy

class Train(object):
    """Class to train the neural network"""

    def __init__(self, game, net):
        """Initializes the class Train"""
        self.game = game
        self.net = net
        self.eval_net = NeuralNetworkWrapper(game)
        self.num_iterations = 4
        self.num_games = 50
        self.temp_thresh = 10
        self.eval_win_rate = 0.55
        self.temp_init = 1
        self.temp_final = 0.001
        
    def start(self):
        """Main loop for training"""
        for i in range(self.num_iterations):
            print("Iteration", i + 1)
            training_data = [] 
            for j in range(self.num_games):
                print("Start Training Self-Play Game", j + 1)
                game = self.game.clone()  # Create a clone of the game
                self.play_game(game, training_data)

            self.net.save_model() # Save the model
            self.eval_net.load_model() # Load the= saved model into the evaluator network
            self.net.train(training_data) # Train the network
            current_mcts = MonteCarloTreeSearch(self.net) # Initialize MonteCarloTreeSearch object for net
            eval_mcts = MonteCarloTreeSearch(self.eval_net) # Initialize MonteCarloTreeSearch object for eval_net
            evaluator = Evaluate(current_mcts=current_mcts, eval_mcts=eval_mcts,
                                 game=self.game)
            wins, losses = evaluator.evaluate()
            print("wins:", wins)
            print("losses:", losses)
            num_games = wins + losses
            if num_games == 0:
                win_rate = 0
            else:
                win_rate = wins / num_games
            print("win rate:", win_rate)
            if win_rate > self.eval_win_rate:
                print("New model saved as best model.")
                self.net.save_model("best_model") # Save current model as the best model if it has a better win_rate
            else:
                print("New model discarded and previous model loaded.")
                self.net.load_model() 

    def play_game(self, game, training_data):
        """Plays a move based on the MCTS output.
        Stops when the game is over ."""
        mcts = MonteCarloTreeSearch(self.net)
        game_over = False
        value = 0
        self_play_data = []
        count = 0
        node = TreeNode()
        while not game_over:
            if count < self.temp_thresh:
                best_child = mcts.search(game, node, self.temp_init) # MCTS simulations to get the best child node
            else:
                best_child = mcts.search(game, node, self.temp_final) # MCTS simulations to get the best child node
            self_play_data.append([deepcopy(game.state),
                                   deepcopy(best_child.parent.child_psas),
                                   0])
            action = best_child.action
            game.play_action(action)
            count += 1
            game_over, value = game.check_game_over(game.current_player)
            best_child.parent = None
            node = best_child  # Make the child node the root node
        for game_state in self_play_data:
            value = -value
            game_state[2] = value # Update v as the value of the game result
            self.augment_data(game_state, training_data, game.row, game.column)

    def augment_data(self, game_state, training_data, row, column):
        state = deepcopy(game_state[0])
        psa_vector = deepcopy(game_state[1])
        training_data.append([state, psa_vector, game_state[2]])
        