<div align="center">

<img
  width="200"
  alt="Nectar Logo"
  src="https://github.com/user-attachments/assets/7e8514a1-19ee-46f2-a23a-7054d3e2f432">

<h3>Nectar</h3>
<b>My first NN chess engine</b>
<br>
<br>

[![License](https://img.shields.io/github/license/Dragjon/Nectar?style=for-the-badge)](https://opensource.org/license/mit)
![Static Badge](https://img.shields.io/badge/Version-0.0.4-yellow?style=for-the-badge)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/dragjon/Nectar?style=for-the-badge)

</div>

## Overview
My first successful attempt at using neural networks to create a relatively strong chess engine that can beat me!
### Encoding Positions
The chess positions are encoded into a 384-element array like this, if its black to move (nstm) we flip the positions and sides
```python
def encode_fen_to_384(fen, turn):
    # Define the index mapping for each piece type in the 384-element array
    piece_to_index = {
        'P': 0,  'p': 0,  # Pawns
        'N': 64, 'n': 64, # Knights
        'B': 128, 'b': 128, # Bishops
        'R': 192, 'r': 192, # Rooks
        'Q': 256, 'q': 256, # Queens
        'K': 320, 'k': 320  # Kings
    }
    
    # Initialize the 384-element array
    board_array = np.zeros(384, dtype=int)
    
    # Split the FEN string to get the board layout
    board, _ = fen.split(' ', 1)
    
    # Split the board part into rows
    rows = board.split('/')
    
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                # Empty squares, advance the column index
                col_idx += int(char)
            else:
                # Piece, determine its position in the 384-element array
                piece_index = piece_to_index[char]
                board_position = row_idx * 8 + col_idx

                if turn == 0:
                    # stm
                    if char.isupper():
                        # White piece
                        board_array[piece_index + board_position] = 1
                    else:
                        # Black piece
                        board_array[piece_index + board_position] = -1
                    col_idx += 1
                else:
                    # nstm
                    if char.isupper():
                        # White piece
                        board_array[piece_index + board_position ^ 56] = -1
                    else:
                        # Black piece
                        board_array[piece_index + board_position ^ 56] = 1
                    col_idx += 1
    
    return board_array
```
### Architechture
I trained 1 neural network for predicting WDL instead of 2 to train more data, as I could've just flipped the colors and positions of nstm. The architechture is 384x32x1 where the inputs are encoded board positions
```python
model = Sequential([
    Dense(32, input_shape=(input_shape,)),
    Lambda(SCReLU),
    Dense(1, activation='sigmoid')
])
```
I used the SCReLU activation function for hidden layers to allow for non-linearity (11.5 +/- 9.2)
```python
def SCReLU(x):
    return tf.square(tf.clip_by_value(x, 0, 1))
```
### Training
The data I used for my training is stash [data](https://drive.google.com/file/d/1LaaW7bNHBnyEdt51MP6SAZCbSdPzlk8d/view) which I parsed into 2 csv files for white and black.
### Rating Changes
| Version | SPRT Elo Gains | Main Changes|
|:-:|:-:|:-:|
| 0.0.4 | 10.6 +/- 8.5 | Tweaked Futility pruning, Changed 2 NNs to 1 NN (for both colours) |
| 0.0.3 | 57.0 +/- 26.3 | Tuned with weather factory, Changed some implementations |
| 0.0.2 | 11.5 +/- 9.2 | SCReLU nets |
| 0.0.1 | - | Initial Release |
### Rating
```
Rank Name                          Elo     +/-   Games   Score    Draw
   1 snowy-v0.2                    203      65     150   76.3%    4.7%
   2 CDrill_1800_Build_4           170      60     150   72.7%    8.0%
   3 Napolean-v1.4.0               139      58     150   69.0%    7.3%
   4 shallowblue-v2.0.0            -56      55     150   42.0%    6.7%
   5 NectarV0.0.2                 -241      66     150   20.0%    9.3%
   6 NectarV0.0.3                 -241      67     150   20.0%    6.7%

Approx: 1432 - 1781
```
### Future plans
- Change input layer to 728 instead of 384
- Do perspective networks
- Do the UE of NNUE
