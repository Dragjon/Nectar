<div align="center">

<img
  width="200"
  alt="Nectar Logo"
  src="https://github.com/user-attachments/assets/7e8514a1-19ee-46f2-a23a-7054d3e2f432">

<h3>Nectar</h3>
<b>My first NN chess engine</b>
<br>
<br>

[![License](https://img.shields.io/github/license/Dragjon/Imperious?style=for-the-badge)](https://opensource.org/license/mit)
![Static Badge](https://img.shields.io/badge/Version-0.0.1-yellow?style=for-the-badge)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/dragjon/nectar?style=for-the-badge)

</div>

## Overview
My first successful attempt at using neural networks to train a relatively strong chess engine that can beat me!
### Architechture
I trained 2 networks for evaluating each color, white and black. The architechture is 384x32x1 where the inputs are encoded board positions
```python
 model = Sequential([
     Dense(32, activation='relu', input_shape=(input_shape,)),
     Dense(1, activation='sigmoid')
 ])
```
### Training
The data I used for my training is stash [data](https://drive.google.com/file/d/1LaaW7bNHBnyEdt51MP6SAZCbSdPzlk8d/view) which I parsed into 2 csv files for white and black.
### Future plans
- Change input layer to 728 instead of 384
- Do perspective networks
- Do the UE of NNUE
