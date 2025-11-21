import numpy as np
import copy

import wandb
import torch


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def random_argmax(array):
    return np.random.choice(np.flatnonzero(array == np.max(array)))


def get_leaf_value(leaf_value_, rl_model, idx_srted=None):
    """Return scalar leaf value based on model type."""
    if rl_model == "EQRDQN":
        if idx_srted is not None:
            return leaf_value_[idx_srted[-1]]  # max Q-value from sorted index
        return leaf_value_.max()  # fallback to max
    elif rl_model == "EQRAC":
        return leaf_value_.mean()  # mean regardless of idx_srted
    else:
        raise ValueError(f"Unsupported rl_model: {rl_model}")


def get_fixed_indices(p):
    if p == 1:
        return [13, 40, 67]
    elif p == 2:
        return [4, 13, 22, 31, 40, 49, 58, 67, 76]
    elif p == 3:
        return [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79]
    elif p == 4:
        return list(range(81))
    else:
        raise ValueError("p should be between 1 and 4")


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None

    def get_Q_value(self, action):
        """
        Returns the MCTS-estimated Q-value (self._Q) for a child node
        reached by the given action.
        """
        if action in self._children:
            # The Q-value for action 'a' from state 's' is stored in
            # the child node corresponding to (s, a).
            return self._children[action]._Q
        return 0.0  # Return 0 for unvisited/non-existent actions

    def get_sorted_actions(self):
        """
        Sorts all expanded actions based on their current MCTS Q-value (self._Q)
        in descending order.
        Returns: A list of actions (integers/keys).
        """
        # Get the Q-value for each expanded child node
        action_q_pairs = {
            action: child_node._Q
            for action, child_node in self._children.items()
        }

        # Sort the actions based on the Q-values in descending order
        sorted_actions = sorted(action_q_pairs, key=action_q_pairs.get, reverse=True)

        return sorted_actions

    @property
    def children(self):
        return self._children


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, args):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = args.c_puct
        self.rl_model = args.rl_model
        self.epsilon = args.epsilon
        self.planning_depth = 0

        self.resource = 0
        self.search_resource = args.search_resource
        self.n_playout = 0
        self.max_width = 81
        self.act_gap = 0
        self.threshold = 0.1

    def _playout(self, env):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root

        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            obs, reward, terminated, info = env.step(action)
            self.planning_depth += 1

        available, action_probs, leaf_value = self._policy(env)
        self.p = 1

        while self.search_resource >= self.max_width:
            if len(available) > 0:
                n_indices = get_fixed_indices(self.p)
                action_probs_ = np.zeros_like(action_probs)

                if self.rl_model == "EQRDQN":
                    leaf_value_ = leaf_value[n_indices, :].cpu().mean(
                        axis=0).squeeze()  # leaf_value shape : 2-dim to 1-dim
                    idx_max = available[np.argmax(leaf_value_[available])]
                    action_probs_[idx_max] = 1
                    action_probs_[available] += self.epsilon / len(available)
                    action_probs_[idx_max] -= self.epsilon
                    action_probs = action_probs_

                    _, idx_srted = leaf_value_.sort()
                    act_gap = torch.abs(leaf_value_[idx_srted[-1]] - leaf_value_[idx_srted[-2]])
                    leaf_value_max = leaf_value_[idx_srted[-1]]

                    if act_gap > self.threshold:
                        action_probs_zip = zip(available, action_probs[available])
                        self.leaf_update(action_probs_zip, leaf_value_max, env, node)

                    else:
                        self.p += 1
                        self.update_search_resource()

                    if self.p == 5:
                        action_probs_zip = zip(available, action_probs[available])
                        self.p = 4
                        self.leaf_update(action_probs_zip, leaf_value_max, env, node)

                elif self.rl_model == "EQRAC":
                    available_probs = action_probs[available]

                    if len(available_probs) >= 2:
                        sorted_available = np.argsort(available_probs)[::-1]

                        a_best = available[sorted_available[0]]
                        a_second_best = available[sorted_available[1]]

                        # Get V-Value for s_{t+1}^{1}
                        env_copy_1 = copy.deepcopy(env)
                        obs_1, reward_1, terminated_1, info_1 = env_copy_1.step(a_best)
                        available_1, _, value_1 = self._policy(env_copy_1)

                        # Get V-Value for s_{t+1}^{2}
                        env_copy_2 = copy.deepcopy(env)
                        obs_2, reward_2, terminated_2, info_2 = env_copy_2.step(a_second_best)
                        available_2, _, value_2 = self._policy(env_copy_2)

                        n_indices = get_fixed_indices(self.p)
                        leaf_value_max = value_1[n_indices].cpu().mean()
                        leaf_value_second = value_2[n_indices].cpu().mean()

                        # Action Gap: | V(s_{t+1} | a_{best}) - V(s_t+1 | a_{second_best}) |
                        act_gap = torch.abs(leaf_value_max - leaf_value_second)

                        if act_gap > self.threshold:  # sufficient action gap
                            action_probs_zip = zip(available, action_probs[available])
                            self.leaf_update(action_probs_zip, leaf_value_max, env, node)

                        else:  # need more quantiles
                            self.p += 1
                            self.update_search_resource()

                        if self.p == 5:  # quantiles = 243 -> 81
                            action_probs_zip = zip(available, action_probs[available])
                            self.p = 4
                            self.leaf_update(action_probs_zip, leaf_value.mean().cpu(), env, node)

                    else:  # available_probs < 2
                        action_probs_zip = zip(available, action_probs[available])
                        self.search_resource = 0
                        self.leaf_update(action_probs_zip, leaf_value.mean().cpu(), env, node)

                else:
                    raise ValueError("rl_model should be EQRDQN or EQRAC")

            else:  # ended game
                action_probs_zip = zip(available, action_probs[available])
                self.search_resource = 0
                self.leaf_update(action_probs_zip, leaf_value.mean().cpu(), env, node)

    def leaf_update(self, action_probs, leaf_value, env, node):
        self.update_search_resource()

        # Check for end of game
        end, winners = env.winner()

        if not end:
            node.expand(action_probs)
        else:
            if winners == 0:  # tie
                leaf_value = 0.0
            elif winners == env.turn():
                leaf_value = 1.0
            else:
                leaf_value = -1.0
        node.update_recursive(-leaf_value)

    def get_move_probs(self, env, game_iter, temp):  # state.shape = (5,9,4)
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        self.n_playout = 0
        while self.search_resource >= self.max_width:
            env_copy = copy.deepcopy(env)
            self.planning_depth = 1
            self.n_playout += 1
            self._playout(env_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        if temp == 1:
            act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        else:
            act_probs = np.zeros(len(acts))
            act_probs[random_argmax(np.array(visits))] = 1.0

        if game_iter + 1 in [1, 10, 20, 31, 50, 100]:
            graph_name = f"training/game_iter_{game_iter + 1}"
            wandb.log({
                f"{graph_name}_pd": self.planning_depth,
                f"{graph_name}_nq": 3 ** self.p,
                f"{graph_name}_n_playout": self.n_playout,
                f"{graph_name}_full_search_rate": 1 if self.p == 4 else 0,
            })

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def update_search_resource(self):
        if self.p == 1:
            self.search_resource -= 3
        elif self.p == 2:
            self.search_resource -= 6
        elif self.p == 3 or 4:
            self.search_resource -= 3 * (3 ** (self.p - 2)) * (2 ** min(1, self.p - 1))
        else:
            raise ValueError("p should be between 1 and 4")

    def __str__(self):
        return "MCTS"


class EMCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function, args, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, args)
        self._is_selfplay = is_selfplay
        self.rl_model = args.rl_model
        self.epsilon = args.epsilon
        self.elo = 1500
        self.resource = args.search_resource

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env, temp=0.1, game_iter=0, return_prob=0):  # env.state_.shape = (5,9,4)
        sensible_moves = np.nonzero(env.state_[3].flatten() == 0)[0]
        move_probs = np.zeros(env.state_.shape[1] * env.state_.shape[2])
        self.mcts.search_resource = self.resource

        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, game_iter, temp)  # env.state_.shape = (5,9,4)
            move_probs[list(acts)] = probs

            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "training MCTS {}".format(self.player)