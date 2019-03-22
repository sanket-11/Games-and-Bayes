#!/usr/bin/env python3
import numpy as np
import sys
import copy
"""
 -----------------------------------------------------------------
| Elements of AI | Assignment - 2 | tajshaik-aparappi-sspatole   
 -----------------------------------------------------------------
| Betsy with Minimax, Alpha Beta pruning and admissible heuristic
 -----------------------------------------------------------------
Code Breakdown:
Main parts of the code are
    1. Successor function which generates states, given a board state
    2. Minimax function which alternates between min/max players, making moves
    3. Heuristic function to evaluate how close to goal state is a given board
    4. Suggestion function that finds out which among the initial successors is the next
       best move i.e., tells the player what he should play next

I) Successor function - successors(board, current_player):
       Generates states based on whether a drop move is possible and rotation of each column
       Checks if #pieces of current_player in board is < pieces to be played,
       If the board contains equal num.of pieces as the given amount, no drop move is possible
       Therefore, only rotation moves will be added as successors of that state
       
       Cases where successors are not generated:
        >> When column is empty, rotation moves are not generated as it is essentially the same state
        >> When a successor generated is same as initial board passed
        >> When a successor generated is same as one generated before

II) Minimax function - minimax(board, depth, isMaximizingPlayer, alpha, beta)
        Generic minimax algorithm with a depth cutoff of 5
        Returns the best value for a given board state

III) Heuristic function - evaluate(board, current_player, goal_or_not)
        Three logics applied to calculate heuristic value for given board and player
        
        >> Number of ways current_player can win
            -- For given board and current_player, check if an opponent is present
               in current_player's horizontals, verticals and diagonals. If yes,
               that route is not a good way to reach goal (i.e, an opponent is blocking).
               Also count how many empty rows, columns and diagonals are present in board
               They are potential routes for current_player to win.
               
        >> Adjacent pieces of current_player
            -- For every piece in the board, count how many horizontal, vertical and
               diagonal neighbours of the same player ('x' or 'o') are present.
               If a piece is surrounded by it's own symbol, chances of winning is more
               Double counting is not ignored here (i.e., A is neighbor of B and vice versa)

        >> Number of empty spots in top 'n' rows of board
            -- Less weightage is given to this feature. Useful in cases of empty/nearly boards

        Count of first two features are found for 'x' and 'o' and subtracted

IV) Suggestion function - check_which_move_best(move_dict)
        To keep track of evaluated values with successors generated, a dict is maintained
        with keys as evaluated values and values as the successor states.
        This function gets the max key (which will give us the best successor) and
        compares it with initial_board passed to detect what move led to that
        best successor (e.g, dropped pebble in x col or rotated x col?)
"""

##-------- Declaring all constants here ----------##
k=0                                 # Used for converting string board into list
n = int(sys.argv[1])                # size of the board
current_player = sys.argv[2]
string_board = sys.argv[3]
time_limit = int(sys.argv[4])
avail_pieces = int((n/2)*(n+3))     # Total pieces for each player
initial_board = [['.']*(n) for i in range(n+3)]
diff = []                           # For identifying suggestion to user
##--------- End of all constants ------------------##


# Converting string board to 2d list
for i in range(0,n+3):
    for j in range(0,n):
        initial_board[i][j] = string_board[k]
        k+=1
        
# Prints board in human friendly format
# REFERENCE: https://stackoverflow.com/questions/17870612/printing-a-two-dimensional-array-in-python
def print_board(board):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in board]))
    print("----------------------------")

# Switch between players
def switch_player(current_player):
    return 'o' if current_player == 'x' else 'x'

# Checks if any win situation is present in board passed for player
# Reference: https://github.com/ImportMathRepositories/Tic_Tac_Toe/blob/master/src/main.py
def is_goal(board, player):
    potential_wins = []
    # n in a row
    for row in board[0:n]:
            potential_wins.append(set(row))

    # n in a column
    for i in range(n):
            potential_wins.append(set([board[k][i] for k in range(n)]))

    # n in a diagonal
    potential_wins.append(set([board[i][i] for i in range(n)]))
    potential_wins.append(set([board[i][(n-1) - i] for i in range(n)]))

    # Checking if any n are the same
    for group in potential_wins:
            if group == set([current_player]):
                    return True
            
    return False

# Given a board and coords, returns True if
# the passed row is the one with the last empty spot
# Used in generating drop successor states

def is_last_empty_spot(a, r, c):
    col_to_inspect = []
    last_index = 0
    for row in a:
        col_to_inspect.append(row[c])

    for i in range(0,len(col_to_inspect)):
        if col_to_inspect[i] == '.':
            last_index += 1
   
    return r==last_index-1

# Create all possible successors, given a board configuration
def successors(board, current_player):    
    states = []
    symbol = 'x' if current_player =='x' else 'o'
    
    piece_count = count_pieces(board, current_player)

    # Generate Drop pebble successors only if pebbles leftover
    if piece_count < avail_pieces:
        for row in range(n+3):
            for col in range(n):
                if board[row][col] == '.' and is_last_empty_spot(board,row,col) :
                    states.append(copy.deepcopy(board))
                    states[-1][row][col] = symbol
    
    # Generate rotate successors
    npboard=np.array(board)
    
    for i in range(n):
        #Dont rotate if column is empty
        if set(npboard[:,i]) != {'.'}:
            if(len(np.where(npboard[:,i] == '.')[0]) > 0):
                spot = max(np.where(npboard[:,i] == '.')[0].tolist())+1
            else:
                spot = 0
            npboard[spot:, i] = np.roll(npboard[spot:,i], 1)
        
            if npboard.tolist() not in states and npboard.tolist() != board :
                states.append(npboard.tolist())
            npboard = np.array(initial_board)

    return states


# Heuristic function to calcuate how close to goal is the
# board passed for current_player
# See comment block at start of file for description on chosen logic
# Went through below slide for information on Game playing and search
# Reference: https://www.ics.uci.edu/~kkask/Fall-2016%20CS271/slides/04-games.pdf

def evaluate(board, current_player, goal_or_not):
    current_player_ways = count_ways(board,current_player)
    opponent_ways = count_ways(board, switch_player(current_player))

    current_player_adj = count_adjacents(board, current_player)
    opponent_adj = count_adjacents(board, switch_player(current_player))
    
    heuristic_val = 2*(current_player_ways - opponent_ways) + \
                    0.5*(find_empty_spots_in_top_nrows(board)) +\
                    3*(current_player_adj - opponent_adj)

    # If board is a goal state, then add 30 (arbitarily chosen)
    # to make heuristic admissible i.e., heuristic will never overestimate goal
    if goal_or_not:
        return heuristic_val + 30
    else:
        return heuristic_val


# Returns how many ways can current_player win
# in given board. i.e, all routes without an opponent
# in the way is a possible win route

def count_ways(board, current_player):
    a = np.array(board[:n])
    row_set = []
    count = 0
    
    for i in a:
        row_set.append(list(set(i)))

    for i in range(n):
        t = []
        for j in range(n):
            t.append(board[j][i])
        row_set.append(list(set(t)))
    
    row_set.append(list(set(a.diagonal())))
    row_set.append(list(set(np.fliplr(a).diagonal())))

    player_wins = [[current_player,'.'],['.'], ['.',current_player]]

    for win_scene in row_set:
        if win_scene in player_wins:
            count+=1

    return count      


# Counts occurrences of neighbours of each
# element in given board and piece
def count_adjacents(a,current_player):
    h= []
    v=[]
    d=[]
    for i in range(n+3):
        for j in range(n):
            h.append(hori_neighbours(a,i,j, current_player))
            v.append(vert_neighbours(a,i,j, current_player))

    d.append(diag_neighbours(a,current_player))       
    return sum(h)+ sum(v) + sum(d)

# Returns how many neighbouring elements are there in each
# diagonal for given board and player
def diag_neighbours(a,current_player):
    count = 0
    a = np.array(a)
    main_d = a.diagonal()
    minor_d = np.fliplr(a).diagonal()

    # Checks if adjacent elements are same as that of current_player
    # Reference: https://stackoverflow.com/questions/14012562/how-to-compare-two-adjacent-items-in-the-same-list-python
    for x, y in zip(main_d, main_d[1:]):
        if x == y == current_player:
            count+=1

    for x, y in zip(minor_d, minor_d[1:]):
        if x == y == current_player:
            count+=1

    return count

# Returns count of vertical neighbours for player in board
def vert_neighbours(a,i,j, current_player):
    count = 0
    if a[i][j] != '.':
        try:
            if a[i][j] == a[i+1][j] == current_player:
                count+=1
        except Exception as e:
            pass

        try:
            if i-1 >=0:
                if a[i][j] == a[i-1][j] == current_player:
                    count+=1
        except Exception as e:
            pass
        return count
    else:
        return 0

# Returns count of horizontal neighbours for player in board
def hori_neighbours(a,i,j, current_player):
    count = 0
    if a[i][j] != '.':
        try:
            if a[i][j] == a[i][j+1] == current_player:
                count+=1
        except Exception as e:
            pass

        try:
            if j-1 >=0:
                if a[i][j] == a[i][j-1] == current_player:
                    count+=1
        except Exception as e:
            pass
        return count
    else:
        return 0
    

# Returns how many vacant spots are present
# in the top n rows alone
# Used in HEURISTIC function 
def find_empty_spots_in_top_nrows(board):
    count = 0
    for i in range(n):
        for j in range(n):
            if board[i][j] == '.':
                count+=1
    return count

# Counts how many pieces of a particular type
# are present in the board at the moment
# Pass board and piece type to count
# Used in SUCCESSORS to limit drop moves
# Returns count

def count_pieces(board, piece_to_count):
    count = 0
    for i in range(n+3):
        for j in range(n):
            if board[i][j] == piece_to_count:
                count+=1
    return count
    
# Minimax function with a depth cutoff of 5
# Does alpha beta pruning
# Referred structure for alternating between min/max from:
# https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-3-tic-tac-toe-ai-finding-optimal-move/
def minimax(board, depth, isMaximizingPlayer, alpha, beta):
    if isMaximizingPlayer:
        current_player='x'
       
    else:
        current_player='o'
    
    goal_or_not = is_goal(board, current_player)

    if(goal_or_not or depth==5):
        score = evaluate(board,current_player, goal_or_not)
        return score
        
    if isMaximizingPlayer :
        bestVal = -1000000
        children=successors(board,current_player)
        for child in children :
            value = minimax(board, depth+1, False, alpha, beta)
            bestVal = max( bestVal, value) 
            alpha = max(alpha, bestVal)
            if (beta <= alpha):
                break
        return bestVal

    else:
        bestVal = 1000000
        children2 = successors(board, current_player)
        for child in children2 :
            value = minimax(board, depth+1, True, alpha, beta)
            bestVal = min( bestVal, value)
            beta = min(beta,bestVal)
            if (beta <= alpha):
                break

        return bestVal

# Given two lists, finds index of that element/col
# which is dissimilar
def which_col_changed(changed, initial):
    for i in range(0,len(initial)):
        if changed[i] != initial[i]:
            return i+1

# Used for suggestion to user on next best move
# See comment block at start of file for details
def check_which_move_best(move_dict):
    best_state_after_move = move_dict[max(move_dict.keys())]

    # Find those rows which are different from initial state
    for i in range(0,len(best_state_after_move)):
        if best_state_after_move[i] != initial_board[i]:
            diff.append((best_state_after_move[i], initial_board[i]))

    # If drop move, then only one row would have been changed 
    if len(diff) == 1:
        return which_col_changed(diff[0][1],diff[0][0]), best_state_after_move

    # If not, then a rotate happened
    else:
        return -which_col_changed(diff[0][1],diff[0][0]), best_state_after_move
    

# Main driver
# Generates first level of successors for initialBoard and
# calls minimax for each
def solve(initialBoard,current_player):
    alpha = -1000000
    beta = 1000000
    children = successors(initialBoard,current_player)
    values=[]
    
    for child in children:
            move=minimax(child,0,False,alpha,beta)
            values.append(move)

    # Dict with best values for each children generated
    move_dict = dict(zip(values, children))
    
    best_move, best_board = check_which_move_best(move_dict)
    return best_move, best_board

# Initial call of solve 
best_move, best_board = solve(initial_board,current_player)

if best_move > 0:
    print("You should drop a pebble in column", best_move)
else:
    print("I'd recommend you rotate column", abs(best_move))

#converting board back to string
s = ""
for i in range(n+3):
    for j in range(n):
        s+=str(best_board[i][j])
print(best_move,s)

#END

