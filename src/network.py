import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
import numpy as np
import os


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def to_tensor(x, device):
    return torch.tensor(np.array(x), dtype=torch.float32, device=device)


def quantile_regression_loss(pred, target, tau, kappa):
    u = target - pred
    abs_u = torch.abs(u)
    huber_loss = torch.where(
        abs_u <= kappa,
        0.5 * u.pow(2),
        kappa * (abs_u - 0.5 * kappa)
    )
    quantile_weight = torch.abs(tau - (u.detach() < 0).float())
    return (quantile_weight * huber_loss).mean()


def apply_masking(probs, available):
    masked_probs = np.zeros_like(probs)
    if len(available) == 0:
        return masked_probs

    masked_probs[available] = probs[available]
    total_sum = masked_probs.sum()

    if total_sum > 0:
        masked_probs /= total_sum
    else:
        masked_probs /= (total_sum + 1)
        # fallback: uniform over available actions
        # masked_probs[available] = 1.0 / len(available)
    return masked_probs


# def compute_masked_act_probs(log_act_probs, state_batch):
#     act_probs = torch.exp(log_act_probs).cpu().numpy()
#     available_mask = (state_batch[:, 3] == 0).cpu().float().numpy()
#     available_mask = available_mask.reshape(64, 36)
#     masked_act_probs = act_probs * available_mask
#
#     if available_mask.sum(axis=1).all() > 0:
#         act_probs = masked_act_probs / masked_act_probs.sum(axis=1, keepdims=True)
#     else:
#         act_probs = masked_act_probs / (masked_act_probs.sum() + 1)
#
#     return act_probs


class DQN(nn.Module):
    """value network module"""

    def __init__(self, board_width, board_height):
        super(DQN, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action value layers
        self.act_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.act_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.act_fc2 = nn.Linear(64, self.num_actions)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action value layers
        x_val = F.relu(self.act_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.act_fc1(x_val))
        x_val = self.act_fc2(x_val)

        # action value to policy
        x_act = F.log_softmax(x_val, dim=1)

        return x_act, x_val


class QRDQN(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height, quantiles):
        super(QRDQN, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height
        self.N = quantiles

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action value layers
        self.act_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.act_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.act_fc2 = nn.Linear(64, self.num_actions * self.N)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action value layers
        x_val = F.relu(self.act_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.act_fc1(x_val))
        x_val = self.act_fc2(x_val)
        x_val = x_val.view(-1, self.N, self.num_actions)  # batch / quantile / action_space

        # action value to policy layers
        x_act = x_val.mean(dim=1)
        x_act = F.log_softmax(x_act, dim=1)

        return x_act, x_val


class EQRDQN(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height, quantiles):
        super(EQRDQN, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height
        self.N = quantiles

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action value layers (previous state value)
        self.act_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.act_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.act_fc2 = nn.Linear(64, self.num_actions * self.N)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action value layers
        x_val = F.relu(self.act_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.act_fc1(x_val))
        x_val = self.act_fc2(x_val)
        x_val = x_val.view(-1, self.N, self.num_actions)  # batch / quantile / action_space

        # action value to policy layers
        x_act = x_val.mean(dim=1)
        x_act = F.log_softmax(x_act, dim=1)

        return x_act, x_val


class AC(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super(AC, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # policy gradient layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy gradient layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class QRAC(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height, quantiles):
        super(QRAC, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.N = quantiles

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # policy gradient layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, self.N)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy gradient layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        x_val = x_val.view(-1, self.N)

        return x_act, x_val


class EQRAC(nn.Module):  # Efficient Quantile Regression action value actor critic
    """policy-value network module"""

    def __init__(self, board_width, board_height, quantiles):
        super(EQRAC, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height
        self.N = quantiles

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # policy gradient layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # action value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, self.N)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy gradient layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = self.act_fc1(x_act)
        x_act = F.log_softmax(x_act, dim=1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)
        x_val = x_val.view(-1, self.N)

        return x_act, x_val

    # def forward_partially(self, state_input):
    #     # common layers
    #     x = F.relu(self.conv1(state_input))
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #
    #     # policy gradient layers
    #     x_act = F.relu(self.act_conv1(x))
    #     x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
    #     x_act = self.act_fc1(x_act)
    #     x_act = F.log_softmax(x_act, dim=1)
    #
    #     # action value layers
    #     x_val = F.relu(self.val_conv1(x))
    #     x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
    #     x_val = F.relu(self.val_fc1(x_val))
    #     # x_val = self.val_fc2(x_val)
    #     # x_val = x_val.view(-1, self.N, self.num_actions)
    #
    #     return x_val

        # X_val * W ?? W? (?? shape, 81*36)
        # ?? W? ?? ?? ??? (?? shape, 9*36)?? ??? ?? ?
        # iter [ 0 ]
        # idx_iter = [0,1,2, 36,37,38, ...]
        # Z_k3 = x_val @ self._policy.val_fc2.weight.data[:, idx_iter] -->  (batchsize, 3*36)

        # iter [ 1 ]
        # idx_c = [3,4,5,6,7,8, 39,40,41,42,43,44, ...]
        # Z_k9 = torch or np.zeros((batchsize, 9*36))
        # Z_k9[:, idx_iter] = Z_k3
        # Z_k9[: idx_c] = x_val @ self._policy.val_fc2.weight.data[:, idx_c]
        # idx_iter = np.union(idx_iter, [3,4,5,6,7,8, 39,40,41,42,43,44, ...]).sort()

        # iter [ 2 ]
        # ...


class PolicyValueNet:
    """policy-value network """

    def __init__(self, board_width, board_height, args, old_model=None):
        self.l2_const = 1e-4  # coef of l2 penalty
        self.gamma = 0.99
        self.kappa = 1.0
        self.N = args.quantiles
        self.rl_model = args.rl_model
        self.old_model = old_model
        self.trained_model = args.init_model

        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        if not self.N is None:
            self.quantile_mid_tau = torch.FloatTensor([(i - 0.5) / self.N for i in range(1, self.N + 1)]).to(self.device)

        # Define model constructors
        model_map = {
            "AC": lambda: AC(board_width, board_height),
            "DQN": lambda: DQN(board_width, board_height),
            "QRDQN": lambda: QRDQN(board_width, board_height, args.quantiles),
            "QRAC": lambda: QRAC(board_width, board_height, args.quantiles),
            "EQRAC": lambda: EQRAC(board_width, board_height, args.quantiles),
            "EQRDQN": lambda: EQRDQN(board_width, board_height, args.quantiles),
        }
        if args.rl_model in model_map:
            self.policy_value_net = model_map[args.rl_model]().to(self.device)
        else:
            print("error")

        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)
        if args.init_model:
            state_dict = torch.load(args.init_model, map_location=self.device, weights_only=True)
            self.policy_value_net.load_state_dict(state_dict)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = np.array(state_batch)
        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # Forward through policy-value network
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = torch.exp(log_act_probs).cpu().numpy()
            value = value.cpu().numpy()

        return act_probs, value

    def policy_value_fn(self, env):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        available = np.nonzero(env.state_[3].flatten() == 0)[0]
        current_state = np.ascontiguousarray(env.state_.reshape(-1, 5, env.state_.shape[1], env.state_.shape[2]))
        current_state = torch.from_numpy(current_state).float().to(self.device)

        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(current_state)
            act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()
            masked_act_probs = apply_masking(act_probs, available)

            if self.rl_model in ["DQN", "QRDQN", "EQRDQN"]:  # if action-value version
                value = value.cpu().numpy().squeeze()
                masked_value = np.zeros_like(value)

                if self.rl_model in ["DQN"]:
                    masked_value[available] = value[available]
                else:  # "QRDQN",  "EQRDQN"
                    masked_value[:, available] = value[:, available]
                value = torch.tensor(masked_value)

        return available, masked_act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        state_batch = to_tensor(state_batch, self.device)
        mcts_probs = to_tensor(mcts_probs, self.device)
        winner_batch = to_tensor(winner_batch, self.device)

        log_act_probs, value = self.policy_value_net(state_batch)

        if self.rl_model in ["DQN"]:
            target = winner_batch.detach()
            value, _ = torch.max(value, dim=1, keepdim=True)
            loss = F.mse_loss(value.view(-1), target)

        elif self.rl_model in ["QRDQN", "EQRDQN"]:
            batch_size, n_quantiles, action_dim = value.shape

            q_expectation = value.mean(dim=1)
            selected_action = torch.argmax(q_expectation, dim=1)
            selected_action = selected_action.view(-1, 1, 1).expand(-1, n_quantiles, 1)
            quantile_pred = torch.gather(value, dim=2, index=selected_action).squeeze(2)

            target = winner_batch.detach().unsqueeze(1).expand(-1, n_quantiles)
            tau = self.quantile_mid_tau.view(1, -1).to(self.device)
            loss = quantile_regression_loss(quantile_pred, target, tau, self.kappa)

        elif self.rl_model in ["AC", "QRAC", "EQRAC"]:
            target = winner_batch.detach()
            value_loss = F.mse_loss(value.view(-1), target)
            policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
            loss = value_loss + policy_loss

        else:
            raise ValueError(f"Unknown rl_model: {self.rl_model}")

        # when call backward, the grad will accumulate. so zero grad before backward
        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        # backward and optimize
        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        # Ensure that the directory exists before saving the file
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        torch.save(net_params, model_file)
