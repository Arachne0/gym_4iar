import argparse
import random
import numpy as np
import logging
import datetime
import wandb

from collections import defaultdict, deque
from fiar_env import Fiar, turn, action2d_ize
from network import PolicyValueNet
from mcts.mcts import MCTSPlayer
from mcts.mcts_efficient import EMCTSPlayer
from utils.file_utils import *


def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter configuration for RL+MCTS")

    # Tuning parameters
    parser.add_argument("--n_playout", type=int, required=False, choices=[2, 20, 50, 100, 400])
    parser.add_argument("--quantiles", type=int, required=False, choices=[3, 9, 27, 81])

    # Efficient search hyperparameters
    parser.add_argument("--search_resource", type=int, required=False, default=8100)

    # RL model type
    parser.add_argument("--rl_model", type=str, required=False, choices=[
        "DQN", "QRDQN", "EQRDQN", "AC", "QRAC", "EQRAC"
    ], help="RL model to use")
    parser.add_argument("--epsilon", type=float, required=False, default=0.4)

    # MCTS parameters
    parser.add_argument("--c_puct", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.1)

    # Policy update parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learn_rate", type=float, default=1e-3)
    parser.add_argument("--lr_multiplier", type=float, default=1.0)
    parser.add_argument("--lr_mul", type=float, default=1.0)
    parser.add_argument("--kl_targ", type=float, default=0.02)
    parser.add_argument("--training_iter", type=int, default=100)

    # Evaluation
    parser.add_argument("--init_model", type=str, default=None)
    args = parser.parse_args()

    return args


def setup_logger(args):
    import os
    if not os.path.exists(f'logs/{args.rl_model}'):
        os.makedirs(f'logs/{args.rl_model}')
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.rl_model in ["QRAC", "EQRAC", "QRDQN", "EQRDQN"]:
        log_filename = f"logs/{args.rl_model}/nplayout_{args.n_playout}_quantiles_{args.quantiles}_{current_time}.log"
    else:
        log_filename = f"logs/{args.rl_model}/nplayout_{args.n_playout}_{current_time}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
        ]
    )
    print(f"Log file created: {log_filename}")


def get_equi_data(env, play_data):
    """augment the data set by rotating 180 degrees and flipping both horizontally and vertically
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, mcts_prob, winner in play_data:
        board_height, board_width = env.state_.shape[1], env.state_.shape[2]

        # Original state and MCTS probabilities
        extend_data.append((state, mcts_prob.flatten(), winner))

        # Rotate 180 degrees
        equi_state_180 = np.array([np.rot90(s, 2) for s in state])
        equi_mcts_prob_180 = np.rot90(mcts_prob.reshape(board_height, board_width), 2)
        extend_data.append((equi_state_180, equi_mcts_prob_180.flatten(), winner))

        # Flip horizontally
        equi_state_hor = np.array([np.fliplr(s) for s in state])
        equi_mcts_prob_hor = np.fliplr(mcts_prob.reshape(board_height, board_width))
        extend_data.append((equi_state_hor, equi_mcts_prob_hor.flatten(), winner))

        # Flip vertically
        equi_state_ver = np.array([np.flipud(s) for s in state])
        equi_mcts_prob_ver = np.flipud(mcts_prob.reshape(board_height, board_width))
        extend_data.append((equi_state_ver, equi_mcts_prob_ver.flatten(), winner))

    return extend_data


def collect_selfplay_data(env, mcts_player, game_iter, n_games=100):
    # self-play 100 games and save in data_buffer(queue)
    # in data_buffer store all steps of self-play so, it should be large enough
    data_buffer = deque(maxlen=36 * n_games * 4)  # board size * n_games * augmentation times
    win_cnt = defaultdict(int)

    for self_play_i in range(n_games):
        rewards, play_data = self_play(env, mcts_player, game_iter, self_play_i)
        play_data = list(play_data)[:]
        # augment the data
        play_data = get_equi_data(env, play_data)
        data_buffer.extend(play_data)
        win_cnt[rewards] += 1

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("\n ---------- Self-Play win: {}, tie:{}, lose: {} ----------".format(win_cnt[1], win_cnt[0], win_cnt[-1]))
    print("Win rate : ", round(win_ratio * 100, 3), "%")

    return data_buffer, win_ratio


def self_play(env, mcts_player, game_iter, self_play_i=0):
    obs, _ = env.reset()
    states, mcts_probs, current_player = [], [], []

    player_0 = 0
    player_1 = 1 - player_0
    obs_post[0] = obs[player_0]
    obs_post[1] = obs[player_1]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[player_0] + obs[player_1]

    while True:
        temp = 1 if env.state_[3].sum() <= 15 else 0
        move, move_probs = mcts_player.get_action(env, game_iter, temp, return_prob=1)

        # store the data
        states.append(obs_post.copy())
        mcts_probs.append(move_probs)
        current_player.append(player_0)

        obs, reward, terminated, info = env.step(move)

        player_0 = 1 - player_0
        player_1 = 1 - player_0
        obs_post[0] = obs[player_0]
        obs_post[1] = obs[player_1]
        obs_post[2] = np.zeros_like(obs[0])
        obs_post[3] = obs[player_0] + obs[player_1]

        end, winners = env.winner()

        if end:
            obs, _ = env.reset()
            if len(current_player) == 36 and winners == 0:  # draw
                print('self_play_draw')

            mcts_player.reset_player()  # reset MCTS root node

            print("game: {}, self_play:{}, episode_len:{}".format(
                game_iter + 1, self_play_i + 1, len(current_player)))
            winners_z = np.zeros(len(current_player))

            if winners != 0:  # non-draw
                """ if winners is 1, winner_index is 0 (black win),
                if winners is -1, winner_index is 1 (white win)
                """
                winner_index = 0 if winners == 1 else 1 
                losers_index = 1 - winner_index
                current_player = np.array(current_player)

                winners_z[current_player == winner_index] = 1.0
                winners_z[current_player == losers_index] = -1.0
            return winners, zip(states, mcts_probs, winners_z)


def policy_update(lr_mul, policy_value_net, data_buffers=None, rl_model=None):
    k, kl, loss, entropy = 0, 0, 0, 0
    lr_multiplier = lr_mul
    update_data_buffer = [data for buffer in data_buffers for data in buffer]

    """update the policy-value net"""
    mini_batch = random.sample(update_data_buffer, args.batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]

    if rl_model in ["DQN", "QRDQN", "EQRDQN"]:
        loss, entropy = policy_value_net.train_step(state_batch,
                                                    mcts_probs_batch,
                                                    winner_batch,
                                                    args.learn_rate)
    else:
        old_probs, old_v = policy_value_net.policy_value(state_batch)

        for k in range(args.epochs):
            loss, entropy = policy_value_net.train_step(state_batch,
                                                        mcts_probs_batch,
                                                        winner_batch,
                                                        args.learn_rate * lr_multiplier)

            new_probs, new_v = policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > args.kl_targ * 4:  # early stopping if D_KL diverges badly
                break

        # adaptively adjust the learning rate
        if kl > args.kl_targ * 2 and lr_multiplier > 0.1:
            lr_multiplier /= 1.5
        elif kl < args.kl_targ / 2 and lr_multiplier < 10:
            lr_multiplier *= 1.5

    return loss, entropy, lr_multiplier, policy_value_net

def policy_evaluate(env, curr_mcts_player, old_mcts_player, game_iter, n_games=30):  # total 30 games
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of trainingSS
    """
    trained_model_wins = 0
    trained_model_losses = 0
    ties = 0
    
    games = n_games // 2

    for i in range(games):
        winner = start_play(env, curr_mcts_player, old_mcts_player, turn=0)
        
        if winner == 1:      # win current MCTS(Black) 
            trained_model_wins += 1
        elif winner == -1:   # win old MCTS(White)
            trained_model_losses += 1
        else:
            ties += 1
        
        print(f"Game {game_iter+1}-{i+1} (Black): Result {winner}")
    
    for i in range(games):
        winner = start_play(env, old_mcts_player, curr_mcts_player, turn=1)
        
        if winner == -1:     # win current MCTS(Black)
            trained_model_wins += 1
        elif winner == 1:    # win old MCTS(White)
            trained_model_losses += 1
        else:
            ties += 1
            
        print(f"Game {game_iter+1}-{games+i+1} (White): Result {winner}")

    win_ratio = 1.0 * trained_model_wins / n_games
    
    print(f"\n---------- Evaluation Result (Iter {game_iter+1}) ----------")
    print(f"Total Games: {n_games}")
    print(f"Current Agent Wins: {trained_model_wins} (Ratio: {win_ratio:.2f})")
    print(f"Current Agent Losses: {trained_model_losses}")
    print(f"Ties: {ties}")
    print("------------------------------------------------------")

    return win_ratio, curr_mcts_player


def start_play(env, player1, player2, turn):
    """start a game between two players"""
    obs, _ = env.reset()
        
    current_player = turn
    opponent_player = 1 - current_player
    player1.set_player_ind(current_player)
    player2.set_player_ind(opponent_player)
    players = {current_player: player1, opponent_player: player2}
    player_in_turn = players[current_player]

    while True:
        # synchronize the MCTS tree with the current state of the game
        move = player_in_turn.get_action(env, temp=0.1, return_prob=0)
        obs, reward, terminated, info = env.step(move)
        assert env.state_[3][action2d_ize(move)] == 1, ("Invalid move", action2d_ize(move))
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]

        else:
            obs, _ = env.reset()
            return winner   


if __name__ == '__main__':
    import time
    start = time.time()

    args = get_args()
    
    setup_logger(args)
    initialize_wandb(args)

    # initialize environment
    env = Fiar()
    obs, _ = env.reset()

    player0 = turn(obs)
    player1 = 1 - player0

    obs_post = obs.copy()
    obs_post[0] = obs[player0]
    obs_post[1] = obs[player1]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[player0] + obs[player1]

    policy_value_net = PolicyValueNet(env.state().shape[1], env.state().shape[2], args)
    if args.rl_model in ["EQRAC", "EQRDQN"]:
        curr_mcts_player = EMCTSPlayer(policy_value_net.policy_value_fn, args, is_selfplay=1)
    else:
        curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, args, is_selfplay=1)

    train_buffer = deque(maxlen=20)
    best_old_model = None
    start_iter = 0
    
    try:
        for i in range(start_iter, args.training_iter):
            """collect self-play data each iteration 100 games"""
            selfplay_batch, win_ratio = collect_selfplay_data(env, curr_mcts_player, i)
            train_buffer.append(selfplay_batch)
            end = time.time()
            print("Elapsed:", end - start)

            """Policy update with data buffer"""
            loss, entropy, lr_multiplier, policy_value_net = policy_update(lr_mul=args.lr_multiplier,
                                                                           policy_value_net=policy_value_net,
                                                                           data_buffers=train_buffer,
                                                                           rl_model=args.rl_model)
            wandb.log({"win_rate/self_play": round(win_ratio * 100, 3),
                       "loss": loss,
                       "entropy": entropy})

            if i == 0:
                """make mcts agent training, eval version"""
                model_file, eval_model_file = create_models(args, i)
                policy_value_net.save_model(model_file)
                policy_value_net.save_model(eval_model_file)

            else:
                existing_files = get_existing_files(args)
                old_i = max(existing_files)
                best_old_model, _ = create_models(args, old_i-1)
                policy_value_net_old = PolicyValueNet(env.state_.shape[1], env.state_.shape[2], args, best_old_model)

                """The most recent model with the highest win rate among the trained models"""
                if args.rl_model in ["EQRAC", "EQRDQN"]:
                    old_mcts_player = EMCTSPlayer(policy_value_net_old.policy_value_fn, args, is_selfplay=0)
                else:
                    old_mcts_player = MCTSPlayer(policy_value_net_old.policy_value_fn, args, is_selfplay=0)

                """Evaluation model"""
                if args.rl_model in ["EQRAC", "EQRDQN"]:
                    curr_mcts_player = EMCTSPlayer(policy_value_net.policy_value_fn, args, is_selfplay=0)
                else:
                    curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, args, is_selfplay=0)

                win_ratio, curr_mcts_player = policy_evaluate(env, curr_mcts_player, old_mcts_player, i)

                if (i + 1) % 10 == 0:  # save model 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 (1+10: total 11)
                    _, eval_model_file = create_models(args, i)
                    policy_value_net.save_model(eval_model_file)

                print("Win rate : ", round(win_ratio * 100, 3), "%")
                wandb.log({"win_rate/evaluate": round(win_ratio * 100, 3)})

                if win_ratio > 0.5:
                    model_file, _ = create_models(args, i)
                    policy_value_net.save_model(model_file)
                    print(" ---------- New best policy!!! ---------- ")

                else:
                    # if worse it just reject and does not go back to the old policy
                    print(" ---------- Low win-rate ---------- ")

        wandb.finish()

    except KeyboardInterrupt:
        print('\n\rquit')