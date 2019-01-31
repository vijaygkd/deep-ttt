# Deep-ttt
Deep Tic Tac Toe

AI for playing the game of Tic-Tac-Toe, build using Deep Neural Networks.

## 1. Data
The data for training the AI was generated using simulations. The data consists of all possible combinations of games that are possible. Each game simulation is stopped when a winning combination is found for either of the players (X or O). The dataset also records which player who won the game or if it was a draw.

### Sample
```
5,3,9,1,8,4,2,0,0,1
```
The first nine entries in the row, indicate the nine move positions made alternatingly by player 1 (X) and player 2 (O).
Move positions range from 1 (top left corner) to 9 (bottom right corner).
Eg. the first entry 5 indicates X played in the center of the board.
```
First move
- - -
- X -
- - -
```
Zeros in position 8 and 9 indicate no moves were made since the game was over on move 7. The last digit in the data indicates the player who won the game. 0 : Draw, 1 : X won, 2 : O won
```
Final board
O X O
O X -
- X X
Player 1 won!
```
