<div align="center">

<img
  width="200"
  alt="Nectar Logo"
  src="https://github.com/user-attachments/assets/7e8514a1-19ee-46f2-a23a-7054d3e2f432">

<h3>Nectar</h3>
<b>Just a random bee who plays chess and collects nectar</b>
<br>
<br>

[![License](https://img.shields.io/github/license/Dragjon/Nectar?style=for-the-badge)](https://opensource.org/license/mit)
![Static Badge](https://img.shields.io/badge/Version-0.1.5-yellow?style=for-the-badge)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/dragjon/Nectar?style=for-the-badge)<br>
[![Lichess rapid rating](https://lichess-shield.vercel.app/api?username=NectarBOT&format=rapid)](https://lichess.org/@/Tokenstealer/perf/rapid)
[![Lichess blitz rating](https://lichess-shield.vercel.app/api?username=NectarBOT&format=blitz)](https://lichess.org/@/Tokenstealer/perf/blitz)
[![Lichess bullet rating](https://lichess-shield.vercel.app/api?username=NectarBOT&format=bullet)](https://lichess.org/@/Tokenstealer/perf/bullet)
</div>

## Overview
My first successful attempt at using neural networks to create a relatively strong chess engine that can beat me! You can track my progress [here](https://sprt.free.nf)
## Playing
### Locally
You can download the precompiled executables for windows [here](https://github.com/Dragjon/Nectar/releases). Note that the compiled version does not come with a GUI. You have to connect it with a chess GUI such as Arena, Banksia or Cutechess. Make sure to set working directory to the directory of the engine executable as the engine reads off its weights from there.
### Online
You can play the latest release of Nectar online at lichess [here](https://lichess.org/@/NectarBOT). Note that it will not always be online.
## Search Features
### Techniques
- Fail-soft negamax search
- Aspiration window search
- Principal variation search (triple PVS)
- Quiescence search
- Efficient updates (UE)
### Pruning
- Beta pruning
- Reverse futility pruning
- Null move pruning
- Futility pruning
- QSearch standing pat pruning
- QSearch delta pruning
- Transposition table cutoffs
### Reductions/Extensions
- Late moves reduction (Log formula)
- Check extensions
- Internal iterative reductions
### Move ordering
- Transposition table ordering
- MVV-LVA ordering
- Killer moves (1 move per ply)
- History heuristic (with gravity and malus)

## Neural Network
Trained with [Nano](https://github.com/Dragjon/Nano)
### Encoding Positions
The chess positions are encoded into a 768-element array like this, in white's POV
```python
def encode_fen(fen):
    # Define the index mapping for each piece type in the 384-element array
    piece_to_index = {
        'P': 0,  'p': 384,  # Pawns
        'N': 64, 'n': 448, # Knights
        'B': 128, 'b': 512, # Bishops
        'R': 192, 'r': 576, # Rooks
        'Q': 256, 'q': 640, # Queens
        'K': 320, 'k': 704  # Kings
    }
    
    # Initialize the 384-element array
    board_array = np.zeros(768, dtype=np.int8)  
    
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
                board_array[piece_index + board_position] = 1
                col_idx += 1
    
    return board_array
```
### Architecture 
A 768->N->1 network in white's POV
```python
model = Sequential([
    Dense(16, input_shape=(input_shape,)),
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
* V0.1.4 onwards: Self-generated data with 5k soft nodes limit and filter when pos is check or bestmove is a capture
* Before V0.1.3: The data I used for my training is stash [data](https://drive.google.com/file/d/1LaaW7bNHBnyEdt51MP6SAZCbSdPzlk8d/view) parsed by cj5716
### Rating Changes
| Version | SPRT Elo Gains | Main Changes|
|:-:|:-:|:-:|
| 0.1.5 | 34.8 +/- 19.7 | Changed to 16 hl
| 0.1.4 | 51.0 +/- 24.6 | Changed input shape and num hidden layers, Efficient updates, SIMD, Hand-retuning values, Refactors
| 0.1.3 | 16.7 +/- 12.3 | Changed 32hl -> 8hl
| 0.1.2 | 10.6 +/- 8.5 | Added history malus + overwrite killers
| 0.1.1 | 57.7 +/- 26.4 | Quantisation
| 0.1.0 | 29.7 +/- 18.0 | Changed scale factor + minor refactor
| 0.0.9 | 85.5 +/- 33.0 | Changed scale factor
| 0.0.8 | 29.7 +/- 18.0 | No reset killers + History Gravity 
| 0.0.7 | 27.6 +/- 17.2 | LMR Log Formula
| 0.0.6 | 36.6 +/- 20.3 | Hand tuned search parameters
| 0.0.5 | 35.1 +/- 19.8 | 2x More data + Trained with AdamW optimiser
| 0.0.4 | 10.6 +/- 8.5 | Tweaked Futility pruning, Changed 2 NNs to 1 NN (for both colours) |
| 0.0.3 | 57.0 +/- 26.3 | Tuned with weather factory, Changed some implementations |
| 0.0.2 | 11.5 +/- 9.2 | SCReLU nets |
| 0.0.1 | - | Initial Release |
## Rating
```
Rank Name                          Elo     +/-   Games   Score    Draw
   1 snowy-v0.2                    211      70     120   77.1%   10.8%
   2 Napolean-v1.4.0                35      59     120   55.0%   13.3%
   3 CDrill_1800_Build_4             0      59     120   50.0%   10.0%
   4 NectarV0.1.5                  -53      60     120   42.5%   11.7%
   5 shallowblue-v2.0.0           -187      70     120   25.4%    5.8%

Elo: ~2000
```
## Credits
- Sebastian Lague (For the whole chess framework)
- Ciekce [Stormphrax] (For teaching me how NNUE works and the NORMAL way to do stuff and AdamW)
- mid_88 [Spaghet] (Explaining how to load binary weights)
- cj5716 [Alexandria] (Parsing the stash data)
- Gediminas (For the UCI interface for Sebastian Lagues's framework)
- gabe [Obsidian] (NNUE explanations)
- jw [Obsidian] (NNUE explanations)
- Matt [Heimdall] (Nicely documented code, based some params on it)
- And a lot more people in the [Engine Programming Discord](https://discord.gg/ZaDHayGV) and [Stockfish Discord](https://discord.gg/ZH62b2rS)
## Future plans
- ~Change input layer to 728 instead of 384~
- Do perspective networks
- ~Do the UE of NNUE~
