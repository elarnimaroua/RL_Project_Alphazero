from mcts import MonteCarloTreeSearch, TreeNode

class HumanPlay(object):

    def __init__(self, game, net):
        """initialize board and nn"""
        self.game = game
        self.net = net
        self.temp_final = 0.001

    def play(self):
        """human vs AI"""
        print("Let's start the game")
        mcts = MonteCarloTreeSearch(self.net)
        game = self.game.clone()  # Create a clone for each game.
        game_over = False
        value = 0
        node = TreeNode()
        print("Enter your move in the form: row, column. Ex: 1,1")
        go_first = input("Do you want to go first: y/n?")
        if go_first.lower().strip() == 'y':
            print("You play as X")
            human_value = 1
            game.print_board()
        else:
            print("You play as O")
            human_value = -1

        while not game_over: #play until u reach a final state ( draw-win-lose)
            if game.current_player == human_value:
                action = input("Enter your move: ")
                if int(action[0]) != 5:
                    while game.state[int(action[0])+1][int(action[2])] == 0:
                            print('Invalid move, try again')
                            action = input("Enter your move: ")
                            if int(action[0]) == 5:
                               break

                if isinstance(action, str):
                    action = [int(n, 10) for n in action.split(",")]
                    action = (1, action[0], action[1])
                
                best_child = TreeNode()
                best_child.action = action
                #add code to see if the move was the best or not ? 
            else: # computer turn 
                best_child = mcts.search(game, node,self.temp_final)

            action = best_child.action
            game.play_action(action)
            game.print_board()
            game_over, value = game.check_game_over(game.current_player)
            best_child.parent = None
            node = best_child  # Make the child node the root node.

        if value == human_value * game.current_player:
            print("You won :o ")
        elif value == -human_value * game.current_player:
            print("You lost :/ ")
        else:
            print("Draw Match ")
        print("\n")