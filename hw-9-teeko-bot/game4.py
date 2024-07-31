import random
import numpy as np
import copy

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    # board is 5x5 list of characters

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def pieces_played(self, state):
        pieces = 0
   
        for col in range(len(state)):
            for row in range(len(state[col])):
                if state[row][col] != ' ':
                    pieces += 1
                        
        return pieces
    
    def max_value(self, state, depth):
        bstate = state
        if self.game_value(state) != 0:
            return self.game_value(state), state
        # if there is no winning state by depth 2
        # use heuristic to find best possible state
        if depth >= 3:
            return self.heuristic_game_value(state, self.my_piece), state
        else:
            a = float('-Inf')
            for s in self.succ(state):
                max_val = self.min_value(s,depth+1)
                if max_val[0] > a:
                    a = max_val[0]
                    bstate = s
        return a, bstate

    def min_value(self, state,depth):
        bstate = state
        if self.game_value(state) != 0:
            return self.game_value(state), state
        # tested depth of 3
        if depth >= 3:
            return self.heuristic_game_value(state, self.opp), state
        else:
            b = float('Inf')
            for s in self.succ(state):
                min_val = self.max_value(s,depth+1)
                if min_val[0] < b:
                    b = min_val[0]
                    bstate = s
        return b, bstate


    def make_move(self, state):
        """Selects a (row, col) space for the next move.
    
        Args:
            state (list of lists): The current state of the game.
    
        Returns:
            move (list): A list of move tuples. In the drop phase, contains
                only the (row, col) tuple indicating where to place the piece.
                In the move phase, contains both the destination and source
                coordinates for relocating the piece.
        """
        
        move = []
        value, bstate = self.max_value(state, 0)
    
        # drop phase
        if self.pieces_played(state) < 8:
            for row in range(len(bstate)):
                for col in range(len(bstate[row])):
                    if bstate[row][col] == ' ':
                        move.append((row, col))
                        return move
    
        # move phase
        for row in range(len(bstate)):
            for col in range(len(bstate[row])):
                if state[row][col] != bstate[row][col]:
                    if state[row][col] == ' ':
                        # piece moved to this position
                        move.insert(0, (row, col))
                    else:
                        # piece moved from this position
                        move.append((row, col))
        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        # if statement programs moving instead of dropping
        # note that this step does not check validity of move
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def check_validity(self, state, row, col):
        if row < 0 or row > 4 or col < 0 or col > 4 or state[row][col] != ' ':
            return False
        return True

    def succ(self, state):
        # returns list of legal successor states
        piece = self.my_piece
        states = []
        
        # drop phase
        # add 1 new current player's piece to unoccupied position
        if(self.pieces_played(state) < 8):
            for col in range(len(state)):
                for row in range(len(state[col])):
                    if state[row][col] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[row][col] = piece
                        states.append(new_state)
        # move phase
        # move a current piece to adjacent, unoccupied position
        # need to check adjacent spaces within range of board
        else:
            for col in range(len(state)):
                for row in range(len(state[col])):
                    if state[row][col] != ' ':
                        if self.check_validity(state, row+1, col):
                            new_state = copy.deepcopy(state)
                            new_state[row+1][col] = piece
                            new_state[row][col] = ' '
                            states.append(new_state)
                        if self.check_validity(state, row+1, col+1):
                            new_state = copy.deepcopy(state)
                            new_state[row+1][col+1] = piece
                            new_state[row][col] = ' '
                            states.append(new_state)
                        if self.check_validity(state, row+1, col-1):
                            new_state = copy.deepcopy(state)
                            new_state[row+1][col-1] = piece
                            new_state[row][col] = ' '
                            states.append(new_state)
                        if self.check_validity(state, row, col+1):
                            new_state = copy.deepcopy(state)
                            new_state[row][col+1] = piece
                            new_state[row][col] = ' '
                            states.append(new_state)
                        if self.check_validity(state, row, col-1):
                            new_state = copy.deepcopy(state)
                            new_state[row][col-1] = piece
                            new_state[row][col] = ' '
                            states.append(new_state)
                        if self.check_validity(state, row-1, col):
                            new_state = copy.deepcopy(state)
                            new_state[row-1][col] = piece
                            new_state[row][col] = ' '
                            states.append(new_state)
                        if self.check_validity(state, row-1, col+1):
                            new_state = copy.deepcopy(state)
                            new_state[row-1][col+1] = piece
                            new_state[row][col] = ' '
                            states.append(new_state)
                        if self.check_validity(state, row-1, col-1):
                            new_state = copy.deepcopy(state)
                            new_state[row-1][col-1] = piece
                            new_state[row][col] = ' '
                            states.append(new_state)
        return states

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for col in range(2):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col+1] == state[i+2][col+2] == state[i+3][col+3]:
                    return 1 if state[i][col]==self.my_piece else -1
        # check / diagonal wins
        for col in range(2):
            for i in range(3,5):
                if state[i][col] != ' ' and state[i][col] == state[i-1][col-1] == state[i-2][col-2] == state[i-3][col-3]:
                    return 1 if state[i][col]==self.my_piece else -1
        # check box wins
        for col in range(4):
            for i in range(4):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i][col+1] == state[i+1][col+1]:
                    return 1 if state[i][col]==self.my_piece else -1

        return 0 # no winner yet
    
    def heuristic_game_value(self, state, piece):
        # if game_value = 1 or -1, return 1 or -1
        game_value = self.game_value(state)
        if(game_value == 1 or game_value == -1):
            return game_value
        # else, evaluate non-terminal states
        best_value = -2

        # 1 piece away from winning state = 0.5
        # 2 pieces away = 0
        # 3 pieces away = -0.5
        # 4 pieces away isn't possible
        
        if piece == 'b':
            opp = 'r'
        elif piece == 'r':
            opp = 'b'
            
        # check horizontal win
        for row in state:
            my_count = sum((i.count(piece) for i in row))
            opp_count = sum((i.count(opp) for i in row))
            best_value = self.value_calc(best_value, my_count, opp_count)
        
        # check vertical win
        for col in state:
            my_count = sum((i.count(piece) for i in col))
            opp_count = sum((i.count(opp) for i in col))
            best_value = self.value_calc(best_value, my_count, opp_count)

        # check \ diagonal wins
        for row in range(2):
            for col in range(2):
                my_count = self.value(state[row][col], piece) + self.value(state[row+1][col+1], piece) + self.value(state[row+2][col+2], piece) + self.value(state[row+3][col+3], piece)
                opp_count = self.value(state[row][col], opp) + self.value(state[row+1][col+1], opp) + self.value(state[row+2][col+2], opp) + self.value(state[row+3][col+3], opp)
                best_value = self.value_calc(best_value, my_count, opp_count)
        # check / diagonal wins
        for col in range(2):
            for i in range(3,5):
                my_count = self.value(state[row][col], piece) + self.value(state[row-1][col-1], piece) + self.value(state[row-2][col-2], piece) + self.value(state[row-3][col-3], piece)
                opp_count = self.value(state[row][col], opp) + self.value(state[row-1][col-1], opp) + self.value(state[row-2][col-2], opp) + self.value(state[row-3][col-3], opp)
                best_value = self.value_calc(best_value, my_count, opp_count)
        # check box wins
        for col in range(4):
            for i in range(4):
                my_count = self.value(state[row][col], piece) + self.value(state[row+1][col], piece) + self.value(state[row][col+1], piece) + self.value(state[row+1][col+1], piece)
                opp_count = self.value(state[row][col], opp) + self.value(state[row+1][col], opp) + self.value(state[row][col+1], opp) + self.value(state[row+1][col+1], opp)
                best_value = self.value_calc(best_value, my_count, opp_count)
        
        return best_value
    
    def value(self, value, piece):
        if piece == value:
            return 1
        return 0
    
    def value_calc(self, best_value, my_count, opp_count):
        if my_count == 3 & opp_count == 0:
            best_value = max(best_value, 0.5)
        if my_count == 2 & opp_count <= 1:
            best_value = max(best_value, 0)
        if my_count == 1 & opp_count <= 2:
            best_value = max(best_value, -0.5)
        return best_value

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Jaime')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.heuristic_game_value(ai.board, ai.my_piece) > -1 and ai.heuristic_game_value(ai.board, ai.my_piece) < 1:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.heuristic_game_value(ai.board, ai.my_piece) > -1 and ai.heuristic_game_value(ai.board, ai.my_piece) < 1:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0])) # bug on this line
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
