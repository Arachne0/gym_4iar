# SH
python version 3.10


## References
### AlphaZero_Gomoku
https://github.com/junxiaosong/AlphaZero_Gomoku

### Elo rating system
https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details

### Requirements
https://escholarship.org/uc/item/8wm748d8

### GIF
https://ransakaravihara.medium.com/how-to-create-gifs-using-matplotlib-891989d0d5ea





pip install -r requirement.txt

```bash
pip install torch torchvision torchaudio
```

### Common
```bash
pip install torch torchvision torchaudio
pip install gymnasium
pip install pyglet
pip install scipy
pip install numpy
pip install matplotlib
pip install gym==0.26.2
pip install wandb
```

Task : four in a row (9 x 4)

### Thinking
- How many quantiles should be used?
3, 9, 27,81  ...?  (for c51)

- Distribution RL Network output (64, quantile) -> ex(64, 32)
so, loss and entropy are lager than AC model (complex calculation)


## RUN main.py
python3 src/main.py --rl_model AC --n_playout 20 
python3 src/main.py --rl_model QAC --n_playout 20 
python3 src/main.py --rl_model DQN --n_playout 20 --epsilon 0.4 
python3 src/main.py --rl_model QRAC --n_playout 20 --quantiles 9 
python3 src/main.py --rl_model QRQAC --n_playout 20 --quantiles 9 
python3 src/main.py --rl_model QRDQN --n_playout 20 --epsilon 0.4 --quantile 9 

python3 src/main.py --rl_model EQRQAC --n_playout 20 
python3 src/main.py --rl_model EQRDQN --epsilon 0.4 --n_playout 20 

## RUN evaluate_pure.py
python3 src/evaluate.py \
--mcts1 mcts \
--rl_model1 AC \
--n_playout1 20 \
--init_model1 models/Training/AC_nmcts20/train_004.pth \
--mcts2 mcts_pure \
--n_playout2 20


--mcts1 mcts --rl_model1 AC --n_playout1 20 --init_model1 /Users/arachne0/gym_4iar/models/Training/AC_nmcts20/train_004.pth --mcts2 mcts_pure --n_playout2 20

# adjust hyperparameter
### tuning parameter 
n_playout = 20  # = MCTS simulations(n_mcts) & training 2, 20, 50, 100, 400
quantiles = 9  # training 3, 9, 27, 81


### MCTS parameter
buffer_size = queue(36 * 20) # board size * last 20 games
c_puct = 5
epochs = 10  # During each training iteration, the DNN is trained for 10 epochs.
self_play_sizes = 100
temperature = 0.1



## Summary

the overall process can be broadly divided into the self-play and start-play phases. 

During self-play, a single Monte Carlo Tree Search (MCTS) alternately plays as both black and white, 
learning through a policy-value network composed of a Multi-Layer Perceptron (MLP), 
akin to an actor-critic model.

The MCTS trained during self-play becomes MCTS 1, and in subsequent games, MCTS 2 is set to choose actions 
purely through MCTS without passing through the policy-value network. 

In the start-play phase, MCTS 1 and MCTS 2, now representing black and white policies, engage in games. 
By default, MCTS 1 is configured to play as black.

The win rate of MCTS 1 during these games is used as a benchmark, and the corresponding policy version 
is saved if the win rate is higher. 
This saved policy version is then compared with the ongoing policies, allowing for the continuous refinement 
of policies based on their performance. 
