
AlphaZero implementation based on the paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm".

The algorithm learns to play the Connect4 game. It uses Monte Carlo Tree Search and a Deep Residual Network to evaluate
the board state and play the most optimal move.


## Useful Commands
**To train the model from scratch.**:
```
python main.py --load_model 0
``` 

**To train the model using the previous best model as a starting point**:
```
python main.py --load_model 1
``` 

**To play a game vs the previous best model**:
```
python main.py --load_model 1 --human_play 1
``` 

This implementation is based on the github repository : https://github.com/blanyal/alpha-zero