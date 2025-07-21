import argparse
import random
import numpy as np

from collections import defaultdict, deque
from fiar_env import Fiar, turn, action2d_ize
from network import PolicyValueNet
from mcts.mcts import MCTSPlayer
from mcts.mcts_pure import PMCTSPlayer
from mcts.mcts_efficient import EMCTSPlayer
from utils.file_utils import *


# def load_yaml_config(path):
#     with open(path, 'r') as f:
#         return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation between two MCTS-based players")

    # --- Player 1 Arguments ---
    parser.add_argument("--mcts1", type=str, required=True, choices=["mcts", "mcts_pure"],
                        help="Type of MCTS for player 1")
    parser.add_argument("--rl_model1", type=str, required=False, choices=[
        "DQN", "QRDQN", "AC", "QAC", "QRAC", "QRQAC", "EQRDQN", "EQRQAC"
    ], help="RL model to use for player 1 (if mcts1 == 'mcts')")
    parser.add_argument("--n_playout1", type=int, required=True, choices=[2, 20, 50, 100, 400],
                        help="Number of playouts for player 1")
    parser.add_argument("--quantiles1", type=int, required=False, choices=[3, 9, 27, 81],
                        help="Quantile value for player 1 (if using quantile-based model)")
    parser.add_argument("--epsilon1", type=float, required=False, choices=[0.1, 0.4, 0.7],
                        help="Epsilon for player 1 (used in exploration)")
    parser.add_argument("--init_model1", type=str, required=False, default=None,
                        help="Path to pretrained model file for player 1")

    # --- Player 2 Arguments ---
    parser.add_argument("--mcts2", type=str, required=True, choices=["mcts", "mcts_pure"],
                        help="Type of MCTS for player 2")
    parser.add_argument("--rl_model2", type=str, required=False, choices=[
        "DQN", "QRDQN", "AC", "QAC", "QRAC", "QRQAC", "EQRDQN", "EQRQAC"
    ], help="RL model to use for player 2 (if mcts2 == 'mcts')")
    parser.add_argument("--n_playout2", type=int, required=True, choices=[2, 20, 50, 100, 400],
                        help="Number of playouts for player 2")
    parser.add_argument("--quantiles2", type=int, required=False, choices=[3, 9, 27, 81],
                        help="Quantile value for player 2 (if using quantile-based model)")
    parser.add_argument("--epsilon2", type=float, required=False, choices=[0.1, 0.4, 0.7],
                        help="Epsilon for player 2 (used in exploration)")
    parser.add_argument("--init_model2", type=str, required=False, default=None,
                        help="Path to pretrained model file for player 2")

    # Efficient search hyperparameters
    parser.add_argument("--effi_n_playout", type=int, required=False, choices=[2, 20, 50, 100, 400])
    parser.add_argument("--search_resource", type=int, required=False, choices=[2, 20, 50, 100, 400])

    # MCTS parameters
    parser.add_argument("--c_puct", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--training_iter", type=int, default=100)

    # Policy update parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learn_rate", type=float, default=1e-3)
    parser.add_argument("--lr_multiplier", type=float, default=1.0)
    parser.add_argument("--lr_mul", type=float, default=1.0)
    parser.add_argument("--kl_targ", type=float, default=0.02)

    # Evaluation
    parser.add_argument("--init_model", type=str, default=None)
    args = parser.parse_args()

    return args


def build_mcts_player(mcts_type, rl_model, n_playout, quantiles, epsilon, init_model, board_shape, args):
    if mcts_type == "mcts_pure":
        return PMCTSPlayer(args)

    args.quantiles = quantiles
    args.rl_model = rl_model
    args.n_playout = n_playout
    args.epsilon = epsilon
    args.init_model = init_model
    policy_value_net = PolicyValueNet(board_shape[1], board_shape[2], args)

    if rl_model in ["EQRDQN", "EQRQAC"]:
        return EMCTSPlayer(policy_value_net.policy_value_fn, args)
    else:
        return MCTSPlayer(policy_value_net.policy_value_fn, args)


def policy_evaluate(env, mcts_player1, mcts_player2, n_games=30):  # total 30 games
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of trainingSS
    """
    win_cnt = defaultdict(int)
    pd, nq = 0, 0
    for j in range(n_games):
        winner, available = start_play(env, mcts_player1, mcts_player2)
        win_cnt[winner] += 1
        print(f"game: {j + 1}, game_len: {len(available)}")

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("---------- win: {}, tie:{}, lose: {} ----------".format(win_cnt[1], win_cnt[0], win_cnt[-1]))
    return win_ratio, pd, nq


def start_play(env, player1, player2):
    """start a game between two players"""
    obs, _ = env.reset()
    players = [0, 1]
    p1, p2 = players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    current_player = 0
    player_in_turn = players[current_player]

    while True:
        # synchronize the MCTS tree with the current state of the game
        move, pd, nq = player_in_turn.get_action(env, temp=0.1)
        obs, reward, terminated, info = env.step(move)
        assert env.state_[3][action2d_ize(move)] == 1, ("Invalid move", action2d_ize(move))
        end, winner = env.winner()

        # if game_iter + 1 in [1, 10, 20, 31, 50, 100]:
        #     graph_name = f"training/game_iter_{game_iter + 1}"
        #     wandb.log({
        #         f"{graph_name}_pd": pd,
        #         f"{graph_name}_nq": nq})

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]

        else:
            if winner == 0:
                print("draw")
            available = np.nonzero(env.state_[3].flatten() == 1)[0]
            obs, _ = env.reset()
            return winner, available


if __name__ == '__main__':
    args = get_args()

    # initialize wandb
    # initialize_wandb(args)

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

    board_shape = env.state().shape
    mcts_player1 = build_mcts_player(args.mcts1, args.rl_model1, args.n_playout1,
                                     args.quantiles1, args.epsilon1,
                                     args.init_model1, board_shape, args)
    mcts_player2 = build_mcts_player(args.mcts2, args.rl_model2, args.n_playout2,
                                     args.quantiles2, args.epsilon2,
                                     args.init_model1, board_shape, args)


    # if args.mcts1 == "mcts_pure":
    #     mcts_player1 = PMCTSPlayer(args)
    # else:
    #     policy_value_net1 = PolicyValueNet(env.state().shape[1], env.state().shape[2], args)
    #     if args.rl_model1 in ["EQRDQN", "EQRQAC"]:
    #         curr_mcts_player = EMCTSPlayer(policy_value_net1.policy_value_fn, args)
    #     else:
    #         curr_mcts_player = MCTSPlayer(policy_value_net1.policy_value_fn, args)
    #
    # if args.mcts2 == "mcts_pure":
    #     mcts_player2 = PMCTSPlayer(args)
    # else:
    #     policy_value_net2 = PolicyValueNet(env.state().shape[1], env.state().shape[2], args)
    #     if args.rl_model1 in ["EQRDQN", "EQRQAC"]:
    #         curr_mcts_player = EMCTSPlayer(policy_value_net2.policy_value_fn, args)
    #     else:
    #         curr_mcts_player = MCTSPlayer(policy_value_net2.policy_value_fn, args)

    try:
        win_ratio = policy_evaluate(env, mcts_player1, mcts_player2)

    except KeyboardInterrupt:
        print('\n\rquit')