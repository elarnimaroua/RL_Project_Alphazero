from copy import deepcopy
import numpy as np


class ConnectFourGame():
    
    def __init__(self):
        super().__init__()
        self.row = 6 #length of the board row
        self.column = 7 #length of the board column
        self.connect = 4 #number of pieces to connect
        self.current_player = 1 
        self.state = []
        for _ in range(self.row):
            self.state.append([0 * j for j in range(self.column)])
        self.state = np.array(self.state) #the game's state
        self.directions = { 
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 1),
            5: (1, -1),
            6: (1, 0),
            7: (1, 1)
        }   #tuples to check for valid moves 

    def clone(self):
        game_clone = ConnectFourGame()
        game_clone.state = deepcopy(self.state) #deep clone of the game
        game_clone.current_player = self.current_player
        return game_clone

    def play_action(self, action):
        """play the turn """
        x = action[1]
        y = action[2]

        self.state[x][y] = self.current_player
        self.current_player = -self.current_player

    def get_valid_moves(self, current_player):
        """Returns a list of moves along with their validity."""
        valid_moves = []
        for x in range(self.row):
            for y in range(self.column):
                if self.state[x][y] == 0:
                    if x + 1 == self.row:
                        valid_moves.append((1, x, y))
                    elif x + 1 < self.row:
                        if self.state[x + 1][y] != 0:
                            valid_moves.append((1, x, y))
                        else:
                            valid_moves.append((0, None, None))
                else:
                    valid_moves.append((0, None, None))

        return np.array(valid_moves)

    def check_game_over(self, current_player):
        """Checks if the game is over and return a possible winner.
        Returns:
            An integer action value. (win: 1, loss: -1, draw: 0
        """

        player_a = current_player
        player_b = -current_player

        for x in range(self.row):
            for y in range(self.column):
                player_a_count = 0
                player_b_count = 0

                # Search for the player a.
                if self.state[x][y] == player_a:
                    player_a_count += 1

                    # Search in all 8 directions for a similar piece.
                    for i in range(len(self.directions)):
                        d = self.directions[i]

                        r = x + d[0]
                        c = y + d[1]

                        if r < self.row and c < self.column:
                            count = 1

                            # Keep searching for a connect.
                            while True:
                                r = x + d[0] * count
                                c = y + d[1] * count

                                count += 1

                                if 0 <= r < self.row and 0 <= c < self.column:
                                    if self.state[r][c] == player_a:
                                        player_a_count += 1
                                    else:
                                        break
                                else:
                                    break

                        if player_a_count >= self.connect:
                            return True, 1

                        player_a_count = 1

                # Search for the player b.
                if self.state[x][y] == player_b:
                    player_b_count += 1

                    # Search in all 8 directions for a similar piece.
                    for i in range(len(self.directions)):
                        d = self.directions[i]

                        r = x + d[0]
                        c = y + d[1]

                        if r < self.row and c < self.column:
                            count = 1

                            # Keep searching for a connect.
                            while True:
                                r = x + d[0] * count
                                c = y + d[1] * count

                                count += 1

                                if 0 <= r < self.row and 0 <= c < self.column:
                                    if self.state[r][c] == player_b:
                                        player_b_count += 1
                                    else:
                                        break
                                else:
                                    break

                        if player_b_count >= self.connect:
                            return True, -1

                        player_b_count = 1

        # There are still moves left so the game is not over
        valid_moves = self.get_valid_moves(current_player)

        for move in valid_moves:
            if move[0] is 1:
                return False, 0

        # If there are no moves left the game is over without a winner
        return True, 0

    def print_board(self):
        """Prints the board state."""
        print("   0    1    2    3    4    5    6")
        for x in range(self.row):
            print(x, end='')
            for y in range(self.column):
                if self.state[x][y] == 0:
                    print('  -  ', end='')
                elif self.state[x][y] == 1:
                    print('  X  ', end='')
                elif self.state[x][y] == -1:
                    print('  O  ', end='')
            print('\n')
        print('\n')