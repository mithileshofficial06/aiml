# Experiment 2: Tic-Tac-Toe using the Minimax Algorithm
#
# Algorithm:
# 1. Create an empty 3x3 board. The player is "X" and the computer is "O".
# 2. Display the board and take the player's move (1-9); reject invalid moves.
# 3. After every move, check for a win or a draw.
# 4. For the computer's turn, try every empty cell and score it using Minimax:
#    +1 if the computer wins, -1 if the player wins, 0 for a draw.
# 5. Minimax assumes the computer picks the maximum score and the player
#    picks the minimum score; the computer plays the cell with the best score.
# 6. Repeat steps 2-5 until someone wins or the board is full.

# 1. Empty board
board = [" "] * 9

def show():
    for i in range(0, 9, 3):
        print(board[i], "|", board[i+1], "|", board[i+2])
        if i < 6:
            print("--+---+--")
    print()

def winner(p):
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
        (0, 4, 8), (2, 4, 6)               # diagonals
    ]
    return any(all(board[i] == p for i in line) for line in lines)

# 4-5. Minimax
def minimax(is_computer):
    if winner("O"):
        return 1
    if winner("X"):
        return -1
    if " " not in board:
        return 0

    scores = []
    for i in range(9):
        if board[i] == " ":
            board[i] = "O" if is_computer else "X"
            scores.append(minimax(not is_computer))
            board[i] = " "

    return max(scores) if is_computer else min(scores)

def best_move():
    best_score, move = -2, None
    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(False)
            board[i] = " "
            if score > best_score:
                best_score, move = score, i
    return move

# 6. Game loop
while True:
    # 2. Player move
    show()
    move = input("Enter position (1-9): ")

    if not move.isdigit() or int(move) < 1 or int(move) > 9 or board[int(move) - 1] != " ":
        print("Invalid position! Try again.\n")
        continue

    board[int(move) - 1] = "X"

    # 3. Check result
    if winner("X"):
        show()
        print("You win!")
        break
    if " " not in board:
        show()
        print("Draw!")
        break

    # 4-5. Computer move
    pos = best_move()
    board[pos] = "O"
    print("Computer chose position", pos + 1)

    if winner("O"):
        show()
        print("Computer wins!")
        break

# ============================================================
# OUTPUT
# ============================================================
#   |   |
# --+---+--
#   |   |
# --+---+--
#   |   |
#
# Enter position (1-9): 5
# Computer chose position 1
# O |   |
# --+---+--
#   | X |
# --+---+--
#   |   |
#
# Enter position (1-9): 1
# Invalid position! Try again.
#
# O |   |
# --+---+--
#   | X |
# --+---+--
#   |   |
#
# Enter position (1-9): 9
# Computer chose position 3
# O |   | O
# --+---+--
#   | X |
# --+---+--
#   |   | X
#
# Enter position (1-9): abc
# Invalid position! Try again.
#
# O |   | O
# --+---+--
#   | X |
# --+---+--
#   |   | X
#
# Enter position (1-9): 8
# Computer chose position 2
# O | O | O
# --+---+--
#   | X |
# --+---+--
#   | X | X
#
# Computer wins!
