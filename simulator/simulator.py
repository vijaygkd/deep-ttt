import itertools
import time
import numpy as np

winning_combo = [
    set([1,2,3]),
    set([4,5,6]),
    set([7,8,9]),
    set([1,4,7]),
    set([2,5,8]),
    set([3,6,9]),
    set([1,5,9]),
    set([3,5,7])  
]


def simulate_all_boards():
    """Simulates all possible
    Tic-Tac-Toe board permutations,
    and annotates winner info.
    Returns: List of boards 
    Board is represented with 9 digits as board positions played.
    Winning info is added as another digit with values,
    0 - Draw;
    1 - X (player 1) wins;
    2 - O (player 2) wins
    """
    print("Simulating Tic-Tac-Toe boards")
    time_start = time.time()

    result = []
    all_possible_boards = itertools.permutations(range(1,10), 9)
    for board in all_possible_boards:
        board = list(board)
        board.append(0)     # winner flag
        # check for wins
        for end in range(5,10):      # win check only after 5th play
            start = (end + 1) % 2   # 0 or 1
            segment = board[start:end:2]
            if did_win(segment):
                board[end:] = [0] * (len(board) - end)
                board[9] = start + 1    # player 1 = X or 2 = O wins
                board_string = ",".join(str(e) for e in board)
                if board_string not in result:
                    result.append(board_string)             
                break
        if board[9] == 0:   # Draw
            board_string = ",".join(str(e) for e in board)
            result.append(board_string)
    
    print("Total Board count:", len(result))
    print("Execution time:", int(time.time() - time_start), "secs")
    return result

def did_win(board):
    board = set(board)
    for combo in winning_combo:
        if combo.issubset(board):
            return True
    return False
