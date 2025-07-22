import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def random_argmax(array):
    return np.random.choice(np.flatnonzero(array == np.max(array)))


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
        self._n_playout = args.n_playout
        self.rl_model = args.rl_model
        self.epsilon = args.epsilon
        self.quantiles = args.quantiles
        self.planning_depth = 0
        self.number_of_quantiles = 0

    def _playout(self, env):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        self.planning_depth, self.number_of_quantiles = 0, 0

        while True:
            self.planning_depth += 1
            if node.is_leaf():
                break

            # Greedily select next move.
            action, node = node.select(self._c_puct)
            obs, reward, terminated, info = env.step(action)

        if self.rl_model in ["DQN", "QRDQN"]:
            available, action_probs, leaf_value = self._policy(env)

            if len(available) > 0:
                action_probs = np.zeros_like(action_probs)

                # leaf_value_: shape = (n_actions,)
                if self.rl_model == "DQN":
                    leaf_value_ = leaf_value.cpu().numpy()
                else:  # QRDQN
                    leaf_value_ = leaf_value.cpu().mean(axis=0).squeeze()

                # mask out illegal actions
                masked_leaf_value = np.zeros_like(leaf_value_)
                masked_leaf_value[available] = leaf_value_[available]

                # greedy action selection among available actions
                idx_max = available[np.argmax(masked_leaf_value[available])]
                action_probs[idx_max] = 1

                # bellman optimality: max Q-value
                leaf_value = masked_leaf_value[available].max()
                action_probs = zip(available, action_probs[available])

            else:
                if self.rl_model == "DQN":
                    leaf_value = leaf_value.cpu().numpy().max()
                    """ use oracle """
                    """ calculate next node.select's node._Q """
                    # leaf_temp = node._Q + (leaf_value - node._Q)/ (node._n_visits+1)
                    # # do we need to calculate next node._u?
                    # leaf_value = leaf_value_[leaf_temp.argmax()]

                    """ use bellman expection to cal state value """
                    # leaf_value = (leaf_value_*action_probs).mean()
                else:
                    leaf_value = leaf_value.cpu().mean(axis=0).max()
                action_probs = zip(available, action_probs[available])


        # if self.rl_model in "DQN":
        #     available, action_probs, leaf_value = self._policy(env)
        #     if len(available) > 0:
        #         action_probs = np.zeros_like(action_probs)
        #
        #         leaf_value_ = leaf_value.cpu().numpy()
        #         masked_leaf_value = np.zeros_like(leaf_value_)
        #         masked_leaf_value[available] = leaf_value_[available]
        #
        #         idx_max = available[np.argmax(masked_leaf_value[available])]
        #         action_probs[idx_max] = 1
        #
        #         """ use bellman optimality to cal state value """
        #         leaf_value = masked_leaf_value[available].max()
        #         action_probs = zip(available, action_probs[available])
        #
        #         """ use oracle """
        #         """ calculate next node.select's node._Q """
        #         # leaf_temp = node._Q + (leaf_value - node._Q)/ (node._n_visits+1)
        #         # # do we need to calculate next node._u?
        #         # leaf_value = leaf_value_[leaf_temp.argmax()]
        #
        #         """ use bellman expection to cal state value """
        #         # leaf_value = (leaf_value_*action_probs).mean()
        #     else:
        #         leaf_value = leaf_value.cpu().numpy().max()
        #         action_probs = zip(available, action_probs[available])
        #
        # elif self.rl_model == "QRDQN":
        #     available, action_probs, leaf_value = self._policy(env)
        #
        #     if len(available) > 0:
        #         action_probs = np.zeros_like(action_probs)
        #         leaf_value_ = leaf_value.cpu().mean(axis=0).squeeze()
        #
        #         masked_leaf_value = np.zeros_like(leaf_value_)
        #         masked_leaf_value[available] = leaf_value_[available]
        #
        #         idx_max = available[np.argmax(masked_leaf_value[available])]
        #         action_probs[idx_max] = 1
        #
        #         leaf_value = masked_leaf_value[available].max()
        #         action_probs = zip(available, action_probs[available])
        #     else:
        #         leaf_value = leaf_value.cpu().mean(axis=0).max()
        #         action_probs = zip(available, action_probs[available])

        else:  # state version AC, QRAC, QAC, QRQAC
            available, action_probs, leaf_value = self._policy(env)
            action_probs = zip(available, action_probs[available])

            if self.rl_model == "QRQAC":
                leaf_value = leaf_value.mean(dim=0).squeeze()

            if self.rl_model in ["QAC", "QRQAC"]:
                leaf_value = leaf_value[available].mean()

        if self.rl_model in ["QRDQN", "QRQAC"]:
            self.number_of_quantiles += self.quantiles

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

    def get_move_probs(self, env, temp):  # state.shape = (5,9,4)
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):  # for 400 times
            env_copy = copy.deepcopy(env)
            self._playout(env_copy)

        pd, nq = self.planning_depth, self.number_of_quantiles

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        if temp == 1:
            act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        else:
            act_probs = np.zeros(len(acts))
            act_probs[random_argmax(np.array(visits))] = 1.0

        return acts, act_probs, pd, nq

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function, args, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, args)
        self._is_selfplay = is_selfplay
        self.rl_model = args.rl_model
        self.epsilon = args.epsilon
        self.elo = 1500

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env, temp=0.1, return_prob=0):  # env.state_.shape = (5,9,4)
        sensible_moves = np.nonzero(env.state_[3].flatten() == 0)[0]
        move_probs = np.zeros(env.state_.shape[1] * env.state_.shape[2])

        if len(sensible_moves) > 0:
            acts, probs, pd, nq = self.mcts.get_move_probs(env, temp)  # env.state_.shape = (5,9,4)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                if self.rl_model in ["DQN", "QRDQN"]:
                    # epsilon-greedy exploration for value-based methods
                    if np.random.rand() < self.epsilon:
                        move = np.random.choice(acts)
                    else:
                        move = acts[np.argmax(probs)]
                else:
                    # Dirichlet noise for policy-based methods
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
                return move, move_probs, pd, nq
            else:
                return move, pd, nq
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "training MCTS {}".format(self.player)