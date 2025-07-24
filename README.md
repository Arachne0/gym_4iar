
## RUN main.py

### requirements.txt
```bash
pip install -r requirement.txt
```

### Usage Examples
```bash
python src/main.py --rl_model AC --n_playout 20 
python src/main.py --rl_model QRAC --n_playout 20 --quantiles 9 
python src/main.py --rl_model QAC --n_playout 20 
python src/main.py --rl_model QRQAC --n_playout 20 --quantiles 9 
python src/main.py --rl_model DQN --n_playout 20 --epsilon 0.4 
python src/main.py --rl_model QRDQN --n_playout 20 --epsilon 0.4 --quantile 9 
python src/main.py --rl_model EQRQAC --n_playout 20 
python src/main.py --rl_model EQRDQN --epsilon 0.4 --n_playout 20 
```

## RUN evaluate_pure.py

### Usage Examples

```bash
python3 src/evaluate.py \
--mcts1 mcts \
--rl_model1 AC \
--n_playout1 20 \
--init_model1 models/Training/AC_nmcts20/train_004.pth \
--mcts2 mcts_pure \
--n_playout2 20
```

## Summary
Task : four in a row (9 x 4)

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


## References
### AlphaZero_Gomoku
https://github.com/junxiaosong/AlphaZero_Gomoku

### Elo rating system
https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details

### Requirements
https://escholarship.org/uc/item/8wm748d8

### GIF
https://ransakaravihara.medium.com/how-to-create-gifs-using-matplotlib-891989d0d5ea

