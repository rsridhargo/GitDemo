'''
board_size=int(input('what size of board you need '))
def print_hor_line():
    print('----'*board_size)

def print_vert_line():
    print('|   '*(board_size+1))

for i in range(board_size):
    print_hor_line()
    print_vert_line()

print_hor_line()
'''

'''
Step 1: Write a function that can print out a board. Set up your board as a list, 
where each index 1-9 corresponds with a number on a number pad,
 so you get a 3 by 3 board representation.
'''
def display_board(board):
    print('   |   |')
    print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
    print('-----------')
    print('   |   |')
    print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('-----------')
    print('   |   |')
    print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('-----------')
    print('   |   |')



'''
Step 2: Write a function that can take in a player input and assign their marker as 'X' or 'O'. 
Think about using while loops to continually ask until you get a correct answer.
'''


def player_input():
    marker = ''
    while not (marker == 'X' or marker == 'O'):
        marker = input('Player 1 choose b/w X or O').upper()

    # if player 1 choose X then player 2 takes O
    if marker == 'X':
        return ('X',"O")
    else:
        return ("O" 'X')


'''
Step 3: Write a function that takes in the board list object, a marker ('X' or 'O'), 
and a desired position (number 1-9) and assigns it to the board.
'''
def place_marker(board, marker, position):
    board[position]=marker


'''
Step 4: Write a function that takes in a board and checks to see if someone has won.
'''
def win_check(board,marker):
    '''
    checking if any player won
    to win all 3 rows or coloums are diagnols should have same marker either X or O
    '''
    return ((board[7]==board[8]==board[9]==marker) or
    (board[4]==board[5]==board[6]==marker) or
    (board[1]==board[2]==board[3]==marker) or
    (board[1]==board[4]==board[7]==marker) or
    (board[2]==board[5]==board[8]==marker) or
    (board[3]==board[5]==board[7]==marker) or
    (board[1]==board[5]==board[9]==marker))

'''
Step 5: Write a function that uses the random module to randomly decide which player goes first.
You may want to lookup random.randint() Return a string of which player went first.
'''
import random


def choose_first():
    if random.randint(0,1)==0:
        return 'player1'
    else:
        return 'player2'

'''
Step 6: Write a function that returns a boolean indicating 
whether a space on the board is freely available.
'''
def space_check(board,position):
    return board[position]==' '
'''
Step 7: Write a function that checks if the board is full and returns a boolean value. 
True if full, False otherwise.
'''
def full_board_check(board):
    for i in range(1,10):
        if space_check(board,i):
            return False   # if any position in board is empty and full_board is false
    return True

'''
Step 8: Write a function that asks for a player's next position (as a number 1-9) 
and then uses the function from step 6 to check if its a free position. 
If it is, then return the position for later use.
'''
def player_choice(board):
    position=0
    while position not in [1,2,3,4,5,6,7,8,9] or not space_check(theBoard,position):
        position = int(input('Choose your next position: (1-9)'))

    return position

'''
Step 9: Write a function that asks the player if they want to play again and 
returns a boolean True if they do want to play again.
'''
def replay():
    choice=input('Do you want to play again? Enter Yes or No:')
    if choice=='yes' or choice=='YES' or choice=='Y' or choice=='y' or choice=='Yes':
        return True



# hard part of the game starts now

# while loop to keep runnig the game

while True:
    # resetting the board to empty list
    theBoard=[' ']*10
    # what marker players choose
    player1_marker,player2_marker=player_input()
    # whose turn first
    turn=choose_first()
    print(turn + 'will go first')
    play_game=input('ready to play game Y or N').lower()
    if play_game=='y':
        game_on=True
    else:
        game_on=False

    while game_on:
        if turn=='player 1':
            # we will first display board to player 1
            display_board(theBoard)
            #we ask for players position
            position=player_choice(theBoard)
            # we mark players position on board
            place_marker(theBoard,player1_marker,position)
            display_board(theBoard)

            if win_check(theBoard,player1_marker):
                display_board(theBoard)
                print('Congratulations! player 1 has won the game!')
                game_on=False
            else:
                # if not won we have to check if there is a tie
                if full_board_check(theBoard):
                    display_board(theBoard)
                    print('Tie game')
                    game_on=False
                else:
                    turn='player 2'

        else:
            display_board(theBoard)
            # we ask for players position
            position = player_choice(theBoard)
            # we mark players position on board
            place_marker(theBoard, player2_marker, position)
            display_board(theBoard)

            if win_check(theBoard, player2_marker):
                display_board(theBoard)
                print('Congratulations! player 2 has won the game!')
                game_on = False
            else:
                # if not won we have to check if there is a tie
                if full_board_check(theBoard):
                    display_board(theBoard)
                    print('Tie game')
                    game_on = False
                else:
                    turn = 'player 1'

    if not replay():
        break
'''
print('Welcome to Tic Tac Toe!')

while True:
    # Reset the board
    theBoard = [' '] * 10
    player1_marker, player2_marker = player_input()
    turn = choose_first()
    print(turn + ' will go first.')

    play_game = input('Are you ready to play? Enter Yes or No.')

    if play_game.lower()[0] == 'y':
        game_on = True
    else:
        game_on = False

    while game_on:
        if turn == 'Player 1':
            # Player1's turn.

            display_board(theBoard)
            position = player_choice(theBoard)
            place_marker(theBoard, player1_marker, position)

            if win_check(theBoard, player1_marker):
                display_board(theBoard)
                print('Congratulations! You have won the game!')
                game_on = False
            else:
                if full_board_check(theBoard):
                    display_board(theBoard)
                    print('The game is a draw!')
                    break
                else:
                    turn = 'Player 2'

        else:
            # Player2's turn.

            display_board(theBoard)
            position = player_choice(theBoard)
            place_marker(theBoard, player2_marker, position)

            if win_check(theBoard, player2_marker):
                display_board(theBoard)
                print('Player 2 has won!')
                game_on = False
            else:
                if full_board_check(theBoard):
                    display_board(theBoard)
                    print('The game is a draw!')
                    break
                else:
                    turn = 'Player 1'

        if not replay():
            break
'''


