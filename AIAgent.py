
import numpy as np
from game_utils import get_valid_col_id
from game_utils import is_win
from game_utils import is_end
from game_utils import ROW_COUNT
from game_utils import COLUMN_COUNT
from game_utils import step
from simulator import Agent
import time

# Values for heuristic function
scoreFor2 = 10  # Adjusted score for 2 discs in line
scoreFor3 = 100  # Adjusted score for 3 discs in line
scoreForWin = 9999999  # High priority for winning moves



class AIAgent(Agent):
    """
    A class representing an agent that plays Connect Four.
    """

    def __init__(self, player_id=1):
        """Initializes the agent with the specified player ID.

        Parameters:
        -----------
        player_id : int
            The ID of the player assigned to this agent (1 or 2).
        """
        self.player_id = player_id
        
    def make_move(self, state):
        """
        Determines and returns the next move for the agent based on the current game state.

        Parameters:
        -----------
        state : np.ndarray
            A 2D numpy array representing the current, read-only state of the game board. 
            The board contains:
            - 0 for an empty cell,
            - 1 for Player 1's piece,
            - 2 for Player 2's piece.

        Returns:
        --------
        int
            The valid action, ie. a valid column index (col_id) where this agent chooses to drop its piece.
        """
        """ YOUR CODE HERE """

        odepth = np.count_nonzero(state)
        #for first move do center
        if odepth <= 2:
            if state[5, 1] != 0: #if agent goes second
                return 2
            elif state[5, 5] != 0:
                return 4
            return 3 #otherwise choose centre
        
        #MAX_DEPTH for AIAgent
        MAX_DEPTH = 7
        if odepth > 13:
            MAX_DEPTH = 8
        if odepth>=19:
            MAX_DEPTH = 9
        if odepth>=23:
            MAX_DEPTH = 10
        if odepth >= 25:
            MAX_DEPTH = 11
        
        #for other moves, get best action by looking 6 depths in.
        visited = {}
        pq = {}
        start = time.time()
        _, column = helper_bfs(state, 0, MAX_DEPTH, self.player_id, float('-inf'), float('inf'), visited, odepth)
        end = time.time()
        print("The time of execution of above program for depth of ", odepth, "is :", (end-start) * 10**3, "ms")
        return column

    #recursion returns winner and best action.
def helper_dfs(state, depth, MAX_DEPTH, player_id, alpha, beta, hashmap, odepth):
    if is_end(state):
        if is_win(state): # if previous player won
            winner = scoreForWin if player_id == 2 else -scoreForWin
            return winner * (MAX_DEPTH + 1 - depth), None # prioritise wins of lower depth
        return 0, None #else, if draw, return 0
    if depth >=MAX_DEPTH: #if no one won,  return heuristic
        return heuristic3(state), None
        # if odepth>=11:#     return heuristic2(state, odepth, depth), None # else:#     return heuristic1(state), None
    best_column = None

    columns = get_valid_col_id(state).tolist()
    columns.sort(key=lambda col: abs(COLUMN_COUNT//2 - col))
    columns = np.array(columns)
    # for odepth <=6, if all pieces are only in column 3, then pick column 3. 
    # for odepth <= 6, if all pieces are only at 2, 3 and 4, then pick from columns 2-4 only
    s = 0
    if odepth <= 9: 
        s = np.int64((odepth - odepth//2) + (odepth//2)*2)
        if np.sum(state[:, 3]) == s:
            return 0, 3
        elif depth == 0 and np.sum(state[:, 2:5]) == s:
            columns = columns[(columns < 5) & (columns>1)]
    elif odepth < 4 and depth == 0 :
        columns = columns[(columns < 5) & (columns>1)]
    elif odepth < 7 and depth == 0 :
        columns = columns[(columns < 6) & (columns>0)]

    if player_id == 1: #max player
        for column in columns:
            maybe_board = step(state, column, player_id, in_place=False)
            hashed_board = hash(maybe_board.tobytes())
            if (hashed_board in hashmap.keys()):
                goodness = hashmap[hashed_board]
            else:
                goodness, _ = helper(maybe_board, depth+1, MAX_DEPTH, 2, alpha, beta, hashmap, odepth)
                hashmap[hashed_board] = goodness
                hashmap[hash(np.flip(maybe_board, 1).tobytes())]  = goodness
            if alpha < goodness:
                alpha = goodness
                best_column = column
            if beta <= alpha:
                break
        return alpha, best_column
    else:
        for column in columns: 
            maybe_board = step(state, column, player_id, in_place=False)
            hashed_board = hash(maybe_board.tobytes())
            if (hashed_board in hashmap.keys()):
                goodness = hashmap[hashed_board]
            else:
                goodness, _ = helper(maybe_board, depth+1, MAX_DEPTH, 1, alpha, beta, hashmap, odepth)
                hashmap[hashed_board] = goodness
                hashmap[hash(np.flip(maybe_board, 1).tobytes())] = goodness
            if beta > goodness:
                beta = goodness
                best_column = column
            if beta <= alpha:
                break
        return beta, best_column
    
# case top case: for each column, get hscore and push into queue. if player is max player, pop the max score and, then recurse. 
# case terminating node case: terminate base on depth or is_end.

def helper_bfs(state, depth, MAX_DEPTH, player_id, alpha, beta, visited, odepth):
    if is_end(state):
        if is_win(state): # if previous player won
            winner = scoreForWin if player_id == 2 else -scoreForWin
            return winner * (MAX_DEPTH + 1 - depth), None # prioritise wins of lower depth
        return 0, None #else, if draw, return 0
    if depth >=MAX_DEPTH: #if no one won,  return heuristic
        return heuristic3(state, player_id, MAX_DEPTH, depth), None
    best_column = None

    columns = get_valid_col_id(state).tolist()
    columns.sort(key=lambda col: abs(COLUMN_COUNT//2 - col))

    columns = np.array(columns)
    # for odepth <=6, if all pieces are only in column 3, then pick column 3. 
    # for odepth <= 6, if all pieces are only at 2, 3 and 4, then pick from columns 2-4 only
    s = 0
    if odepth <= 9: 
        s = np.int64((odepth - odepth//2) + (odepth//2)*2)
        if np.sum(state[:, 3]) == s:
            return 0, 3
        elif depth == 0 and np.sum(state[:, 2:5]) == s:
            columns = columns[(columns < 5) & (columns>1)]
    elif odepth < 4 and depth == 0 :
        columns = columns[(columns < 5) & (columns>1)]
    elif odepth < 7 and depth == 0 :
        columns = columns[(columns < 6) & (columns>0)]

    column_dict = {}
    score = heuristic3(state, player_id, MAX_DEPTH, depth)

    for column in columns:
        r = np.where(state[:,column] == 0)
        row = r[0][len(r[0])-1] + 1 
        maybe_board = step(state, column, player_id, in_place=False)
        hscore = heuristic_additive(score,row, column,state,maybe_board, player_id, MAX_DEPTH, depth)
        # if hscore not in column_dict.values():
        column_dict[column] = (hscore, maybe_board)
    columns = sorted(column_dict.keys(), key=lambda column: column_dict[column][0]) # max score is in front, min score is back

    if player_id == 1: #max player
        for column in columns:
            maybe_board = column_dict[column][1]
            hashed_board = hash(maybe_board.tobytes())
            if (hashed_board in visited.keys()):
                goodness = visited[hashed_board]
            else:
                goodness, _ = helper_bfs(maybe_board, depth+1, MAX_DEPTH, 2, alpha, beta, visited, odepth)
                visited[hashed_board] = goodness
                visited[hash(np.flip(maybe_board, 1).tobytes())]  = goodness
            if alpha < goodness:
                alpha = goodness
                best_column = column
            if beta <= alpha:
                break
        return alpha, best_column
    else:
        columns.reverse()
        for column in columns:
            maybe_board = column_dict[column][1]
            hashed_board = hash(maybe_board.tobytes())
            if (hashed_board in visited.keys()):
                goodness = visited[hashed_board]
            else:
                goodness, _ = helper_bfs(maybe_board, depth+1, MAX_DEPTH, 1, alpha, beta, visited, odepth)
                visited[hashed_board] = goodness
                visited[hash(np.flip(maybe_board, 1).tobytes())] = goodness
            if beta > goodness:
                beta = goodness
                best_column = column
            if beta <= alpha:
                break
        return beta, best_column
    
#for adjacents, check if it is blocked
#row_step and col_step is -1, 0 or 1
#row and col is the starting point's index
#returns score of that line in terms of max player 1
def score_of_line(state, row_step, col_step, row, col):
    r, c = row, col
    line = []
    best_streak = [None, 0, 0]
    score1, score2 = 0, 0
    while r >= 0 and c >= 0 and r < ROW_COUNT and c < COLUMN_COUNT:
        line.append(state[r, c])
        r += row_step
        c += col_step
    line_size = len(line)
    if line_size < 4:
        return 0
    for player in range(1, 3):
        other_player = 2 if player == 1 else 1
        for i in range(line_size - 3):
            if other_player not in line[i:i+4]:
                streak = line[i:i+4].count(player)
                if streak == 3:
                    best_streak[player] = streak
                    break
                if best_streak[player] < streak:
                    best_streak[player] = streak
    if best_streak[1] == 2:
        score1 = scoreFor2
    elif best_streak[1] == 3:
        score1 = scoreFor3
    if best_streak[2] == 2:
        score2 = scoreFor2
    elif best_streak[2] == 3:
        score2 = scoreFor3
    return score1 - score2


def heuristic3(state, player_id, MAX_DEPTH, depth):
    if is_win(state): # if current player won
        winner = scoreForWin if player_id == 1 else -scoreForWin
        return winner * (MAX_DEPTH + 1 - depth) # prioritise wins of lower depth
    score = 0
    #collate horizontal scores
    for i in range(ROW_COUNT):
        score += score_of_line(state, 0, 1, i, 0) # right, start from first columns
    #collate rightward diag scores
    for i in range(4):
        score += score_of_line(state, -1, 1, ROW_COUNT - 1, i) #up and right, start from bottom rows
    #collate vertical scores
    for i in range(COLUMN_COUNT):
        score += score_of_line(state, -1, 0, ROW_COUNT - 1, i) # up, start from bottom rows
    #collate leftward diag scores
    for i in range(3,COLUMN_COUNT):
        score += score_of_line(state, -1, -1, ROW_COUNT - 1, i) # up and left, start from bottom rows
    return score

def heuristic_additive(score, row, col, state, newstate, player_id, MAX_DEPTH, depth):
    if is_win(state): # if current player won
        winner = scoreForWin if player_id == 1 else -scoreForWin
        return winner * (MAX_DEPTH + 1 - depth) # prioritise wins of lower depth
    #horizontal scores
    score += (score_of_line(newstate, 0,1,row,col) - score_of_line(state, 0,1,row,col)) 
    r,c = row, col
    while r < ROW_COUNT and COLUMN_COUNT >= 0:
        r +=1
        c -=1
    if r == ROW_COUNT-1:
        score += (score_of_line(newstate, -1,1,r,c) - score_of_line(state, -1,1,r,c))

    r,c = row, col
    while r < ROW_COUNT:
        r +=1
    if r == ROW_COUNT-1:
        score += (score_of_line(newstate, -1,0, ROW_COUNT - 1,col) - score_of_line(state, -1,0,ROW_COUNT - 1,col))

    r,c = row, col
    while r < ROW_COUNT and COLUMN_COUNT >= 0:
        r +=1
        c +=1
    if r == ROW_COUNT-1:
        score += (score_of_line(newstate, -1,-1, r,c) - score_of_line(state, -1,-1, r,c))
    return score



    
    
        


            
            
    


    

    
# #heuristic in terms of max player
# def heuristic2(state, odepth, depth):
#     #count the length of the vertical. might be 1 or more. returns the score. 
#     def countVertical(state, row, col, player, doublecounter):
#         stop, blocked, i = False, False, 1
#         while not stop:
#             if row-i < 0:
#                 stop = True
#                 continue
#             slot = state[row-i, col]
#             # if chip is player, continue adding i
#             if slot == player:
#                 i += 1
#             elif slot == 0:
#                 stop = True
#             else: #if chip is not player nor empty, it is otherPlayer
#                 stop, blocked = True, True
#         if not blocked:
#             if i == 2:
#                 return scoreFor2
#             else:
#                 return scoreFor3
#         else:
#             return 0

#     def countHorizontal(state, row, col, player, doublecounter):
#         score, stop, i = 0, False, 1
#         otherplayer = 2 if player == 1 else 2
#         if (row, col) in doublecounter:
#             return 0
#         while not stop:
#             if col+i >= COLUMN_COUNT:
#                 stop = True
#                 continue
#             slot = state[row, col+i]
#             if (row, col+i) in doublecounter:
#                 return 0
#             if slot == player:
#                 i += 1
#             else:
#                 stop = True
#         #if no of chips in a row is 2, 3 ways to form a line
#         if i==2:
#             if col+i+1<COLUMN_COUNT: #right line
#                 if state[row, col+i] != otherplayer and state[row, col +i +1] != otherplayer: #if slots are 0 or player
#                     if state[row, col +i +1] == player: #if last slot is player, add more points, but dont double count that slot's line
#                         score += scoreFor3
#                         doublecounter.append((row, col+i))
#                     else:
#                         score += scoreFor2
#             # if col-1 and col-2 exist:
#             if col-1 >= 0 and col+i<COLUMN_COUNT: #middle line
#                 if state[row,col-1] != otherplayer and state[row,col+i] != otherplayer:
#                     score += scoreFor2
#             if col-2 >= 0: #left line
#                 if state[row, col-1] != otherplayer and state[row, col-2] != otherplayer: 
#                     if state[row, col-2] == player:
#                         score += scoreFor3
#                         doublecounter.append((row, col-2))
#                     else:
#                         score += scoreFor2
#         #if no of chips in a row is 3, 2 ways to form a line
#         elif i==3:
#             if col+i<COLUMN_COUNT and state[row,col+i] != otherplayer:
#                 score += scoreFor3
#             if col-1 >=0 and state[row,col-1] != otherplayer:
#                 score += scoreFor3
#         return score
    
#     def countLeftDiagonal(state, row, col, player, doublecounter):
#         if (row, col) in doublecounter:
#             return 0
#         score = 0
#         stop = False
#         i = 1
#         otherplayer = 2 if player == 1 else 2
#         while not stop:
#             if row-i<0 or col-i<0:
#                 stop = True
#                 continue
#             if (row-i, col-i) in doublecounter:
#                 return 0
#             slot = state[row-i, col-i]
#             if slot == player:
#                 i+=1
#             else:
#                 stop = True
#         if i==2:
#             if row-i-1>=0 and col-i-1>=0:       
#                 if state[row-i, col-i]!= otherplayer and state[row-i-1, col-i-1]!= otherplayer:
#                     if state[row-i-1, col-i-1] == player:
#                         score += scoreFor3
#                     else:
#                         score += scoreFor2
#             if row-i>=0 and col-i>=0 and row+1<ROW_COUNT and col+1<COLUMN_COUNT:
#                 if state[row-i, col-i]!= otherplayer and state[row+1, col+1]!= otherplayer:
#                     score+= scoreFor2
#             if row+2<ROW_COUNT and col+2<COLUMN_COUNT:
#                 if state[row+2, col+2]!= otherplayer and state[row+1, col+1]!= otherplayer:
#                     if state[row+2, col+2] == player:
#                         score += scoreFor3
#                     else:
#                         score += scoreFor2
#         elif i==3:
#             if row-i>=0 and col-i>=0 and state[row-i, col-i]!= otherplayer:
#                 score += scoreFor3
#             if row+1<ROW_COUNT and col+1<COLUMN_COUNT and state[row+1,col+1]!= otherplayer:
#                 score+= scoreFor3
#         return score
    
#     def countRightDiagonal(state, row, col, player, doublecounter):
#         if (row, col) in doublecounter:
#             return 0
#         score = 0
#         stop = False
#         i = 1
#         otherplayer = 2 if player == 1 else 2
#         while not stop:
#             if row-i<0 or col+i>=COLUMN_COUNT: # if position is not out of board
#                 stop = True
#                 continue
#             if (row-i, col+i) in doublecounter: # if position is already counted
#                 return 0
#             slot = state[row-i, col+i]
#             if slot == player:
#                 i+=1
#             else:
#                 stop = True
#             if (row-i-1 < 0):
#                 stop =  True
#         if i == 2:
#             if row-i-1 >=0 and col+i+1 < COLUMN_COUNT:
#                 if state[row-i, col+i]!= otherplayer and state[row-i-1, col+i+1]!= otherplayer:
#                     if state[row-i-1, col+i+1] == player:
#                         score += scoreFor3
#                     else:
#                         score += scoreFor2
#             if row-i>=0 and col+i<COLUMN_COUNT and row+1<ROW_COUNT and col-1>=0:
#                 if state[row-i, col+i]!= otherplayer and state[row+1, col-1]!= otherplayer:
#                     score += scoreFor2
#             if row+2<ROW_COUNT and col-2>=0:
#                 if state[row+1, col-1]!= otherplayer and state[row+2, col-2]!= otherplayer:
#                     if state[row+2, col-2] == player:
#                         score += scoreFor3
#                     else:
#                         score += scoreFor2
#         if i==3:
#             if row+1<ROW_COUNT and col-1>=0:
#                 if state[row+1, col-1]!= otherplayer:
#                     score += scoreFor3
#             if row-i>=0 and col+i < COLUMN_COUNT:
#                 if state[row-i,col+i] != otherplayer:
#                     score += scoreFor3
#         return score
#     # check for transposition table memory only for the current depth.
#     # + number of 2's and 3's for max player
#     # - number of 2's and 3's for min player
#     score = [None, 0, 0] 
#     doublecounter = []
#     for row in reversed(range(ROW_COUNT)): #rows: 5 4 3 2 1 0
#         for col in range(COLUMN_COUNT): #col: 0 1 2 3 4 5 6
#             player = state[row, col]
#             if player != 0:  # for each valid player,
#                 # check vertical if there is no same chip below
#                 if row+1 < ROW_COUNT and state[row+1, col] != player:
#                     score[player] += countVertical(state, row, col, player, doublecounter)
#                 # check horizontal if there is no same chip on the left
#                 if col-1>=0 and state[row, col-1] != player:
#                     score[player] += countHorizontal(state, row, col, player, doublecounter)
#                 # check if there is no same chip on the bottom right
#                 if row+1<ROW_COUNT and col+1<COLUMN_COUNT and state[row+1, col+1] != player:
#                     score[player] += countLeftDiagonal(state, row, col, player, doublecounter)
#                 # check right diagonal if there is no same chip on the bottom left
#                 if row+1<ROW_COUNT and col-1>=0 and state[row+1, col-1] != player:
#                     score[player] += countRightDiagonal(state, row, col, player, doublecounter)
# #depth == 6 and odepth == 11 and state[4,1]==2 and state[3,2] == 2 and state[1,2] == 2 and (score[1]-score[2]<-14)
# # ideal and gives best score of -20
#     return score[1] - score[2]

# #heuristic in terms of max player
# def heuristic1(state):
#     #count the length of the vertical. might be 1 or more. returns the score. 
#     def countVertical(state, row, col, player):
#         stop, blocked, i = False, False, 0 # +1 to be the no of chips in the line
#         while not stop: # stop when next chip is not the player chip. 
#             slot = state[row-i, col]
#             # if chip is player, continue adding i
#             if slot == player:
#                 i += 1
#             elif slot == 0:
#                 stop = True
#             else: #if chip is not player nor empty, it is otherPlayer
#                 stop, blocked = True, True
#             if (row-i-1 < 0):
#                 stop =  True
#         if not blocked:
#             if i == 2:
#                 return scoreFor2
#             else:
#                 return scoreFor3
#         else:
#             return 0
        
#     def countHorizontal(state, row, col, player):
#         stop, blocked, i = False, False, 0
#         while not stop:
#             slot = state[row, col + i]
#             if slot == player:
#                 i += 1
#             elif slot == 0:
#                 stop = True
#             else:
#                 stop, blocked = True, True
#             if (col + i + 1 > 6):
#                 stop = True
#         if not blocked:
#             if i == 2:
#                 return scoreFor2
#             else:
#                 return scoreFor3
#         else:
#             return 0
        
#     def countLeftDiagonal(state, row, col, player):
#         stop, blocked, i = False, False, 0
#         while not stop:
#             slot = state[row-i, col-i]
#             if slot == player:
#                 i+=1
#             elif slot == 0:
#                 stop = True
#             else:
#                 stop, blocked = True, True
#             if (row-i-1 < 0):
#                 stop =  True
#             if (col-i-1 < 0):
#                 stop = True
#         if not blocked:
#             if i == 2:
#                 return scoreFor2
#             else: 
#                 return scoreFor3
#         else:
#             return 0
        
#     def countRightDiagonal(state, row, col, player):
#         stop = False
#         blocked = False
#         i = 0
#         while not stop:
#             slot = state[row-i, col+i]
#             if slot == player:
#                 i+=1
#             elif slot == 0:
#                 stop = True
#             else:
#                 stop = True
#                 blocked = True
#             if (row-i-1 < 0):
#                 stop =  True
#             if (col+i+1 > 6):
#                 stop = True
#         if not blocked:
#             if i == 2:
#                 return scoreFor2
#             else:
#                 return scoreFor3
#         else:
#             return 0
#         # + number of 2's and 3's for max player
#     # - number of 2's and 3's for min player
#     score = [None, 0, 0] 
#     for row in reversed(range(ROW_COUNT)): #rows: 5 4 3 2 1 0
#         for col in range(COLUMN_COUNT): #col: 0 1 2 3 4 5 6
#             player = state[row, col]
#             if player != 0:  # for each valid player,
#                 # check vertical if row is 5,4,3, and if there is no same chip below
#                 if row>=3 and (row == 5 or (row<5 and state[row+1, col] != player)):
#                     score[player] += countVertical(state, row, col, player)
#                 # check horizontal rightwards only if col is 0 1 2 3 and if there is no same chip on the left
#                 if col<=3 and (col == 0 or (col>0 and state[row, col-1] != player)):
#                     score[player] += countHorizontal(state, row, col, player)
#                 # check left diagonal if col is 3 4 5 6 and if there is no same chip on the bottom right
#                 if col>=3 and (col==6 or (col<6 and row<5 and state[row+1, col+1] != player)):
#                     score[player] += countLeftDiagonal(state, row, col, player)
#                 # check right diagonal if col is 0 1 2 3 and if there is no same chip on the bottom left
#                 if col <=3 and (col==0 or(col>0 and row<5 and state[row+1, col-1] != player)):
#                     score[player] += countRightDiagonal(state, row, col, player)
        
#     return score[1] - score[2]