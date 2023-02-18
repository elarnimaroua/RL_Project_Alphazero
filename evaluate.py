from mcts import TreeNode

class Evaluate(object):
    """Represents the policy and value Resnet"""

    def __init__(self, current_mcts, eval_mcts, game):
        """Initializes the class Evaluate"""
        self.current_mcts = current_mcts
        self.eval_mcts = eval_mcts
        self.game = game
        self.temp_final = 0.001
        self.num_eval_games = 12 

    def evaluate(self):
        """Self-play loop and record game states"""
        wins = 0
        losses = 0

        for i in range(self.num_eval_games):
            print("Start Evaluation Self-Play Game:", i, "\n")
            game = self.game.clone()  # Create a fresh clone for each game.
            game_over = False
            value = 0
            node = TreeNode()
            player = game.current_player
            while not game_over:
                if game.current_player == 1: # Play using the current network
                    best_child = self.current_mcts.search(game, node, self.temp_final) # MCTS simulations to get the best child node.
                else: # Play using the evaluation network
                    best_child = self.eval_mcts.search(game, node, self.temp_final)

                action = best_child.action
                game.play_action(action)  # Play the child node's action.
                game.print_board()
                game_over, value = game.check_game_over(player)
                best_child.parent = None
                node = best_child  # Make the child node the root node.

            if value == 1:
                print("win")
                wins += 1
            elif value == -1:
                print("loss")
                losses += 1
            else:
                print("draw")
            print("\n")

        return wins, losses
