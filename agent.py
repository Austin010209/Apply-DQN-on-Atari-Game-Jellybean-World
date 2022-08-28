import os
from pathlib import Path
from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import math
import random
import os.path as p

GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 50000

MIN_REPLAY_SIZE = 1000

EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000

TARGET_UPDATE_FREQ = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GAMMA = 0.99
# BATCH_SIZE = 64
# BUFFER_SIZE = 100000
# MIN_REPLAY_SIZE = 10000
# EPSILON_START = 1.0
# EPSILON_END = 0.02
# EPSILON_DECAY = 100000
# TARGET_UPDATE_FREQ = 1000
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


class Network(nn.Module):
    def __init__(self, env, init_weights=True, cfg=None):  # it actually has little to do with the env itself;
        # it in fact just asks for size of input and output
        super().__init__()
        if cfg is None:
            # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 512, 512]
            cfg = [16, 16, "M", 32, 32, "M", 64, 128, 128]
        self.cfg = cfg
        self.feature = self.make_layers(cfg, True)
        self.num_classes = env['action_space'].n
        # self.classifier = nn.Linear(cfg[-1] + 3, 128)
        self.classifier = nn.Linear(cfg[-1], 128)
        self.fc1 = self.get_dense()
        if init_weights:
            self._initialize_weights()

    def get_dense(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, self.num_classes)

        )

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        # in_channels = 4
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        # layers += []
        return nn.Sequential(*layers)

    def forward(self, x):

        # image = x[:, :900].view(-1, 4, 15, 15)
        image = x[:, :900].view(-1, 3, 15, 15)
        # scent = x[:, 900:]

        x = self.feature(image)
        # x = self.get_dense()
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        # x = torch.cat((x, scent), 1)

        # x = self.fc1(x)

        x = self.classifier(x)
        y = self.fc1(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def act(self, obs):
        obs = np.array(obs)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action


class Agent():
    '''The agent class that is to be filled.
         You are allowed to add any method you
         want to this class.
    '''

    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.online_net, self.target_net = self.get_NNs(self.env_specs)

#         self.load_model()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4, betas=(0.99, 0.999))
        self.step = 0
        self.epsilon = self.get_eps(self.step)

    def load_model(self):
        save_path = "./GROUP_001/weights.pth"
        saved = torch.load(save_path)
        self.online_net.load_state_dict(saved["state_dict"])
        self.target_net.load_state_dict(saved["state_dict"])

    def save_model(self, model_name, model: Network, path: Path):
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(model.state_dict(), str(path / f"{model_name}"))

    def get_eps(self, step):
        # return
        return np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    def get_NNs(self, env):
        online_net = Network(env)
        target_net = Network(env)
        target_net.load_state_dict(online_net.state_dict())
        online_net = online_net.to(device)
        target_net = target_net.to(device)
        return online_net, target_net

    def get_action(self, mode, epsilon, act_space, online_net, obs):
        rnd_sample = random.random()

        if self.step > MIN_REPLAY_SIZE and mode == "train" and rnd_sample > epsilon:
            return online_net.act(obs)
        elif mode == "eval":
            if (rnd_sample <= 2):
                return online_net.act(obs)
            else:
                return act_space.sample()
        else:
            # return act_space.sample()
            if self.step <= MIN_REPLAY_SIZE / 4:
                return 0
            elif self.step > MIN_REPLAY_SIZE / 4 and self.step <= MIN_REPLAY_SIZE / 2:
                return 1
            elif self.step > MIN_REPLAY_SIZE / 2 and self.step <= 3 * MIN_REPLAY_SIZE / 4:
                return 2
            else:
                return act_space.sample()

    def get_exp_from_buffer(self, transitions):
        ## transitions : observation space

        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=device)
        return obses_t, actions_t, rews_t, dones_t, new_obses_t

    def load_weights(self, root_path = None):
        save_path = p.join(root_path, "weights.pth")
        
        # if Path(save_path).exists():
        saved = torch.load(save_path)
        self.online_net.load_state_dict(saved["state_dict"])
        self.target_net.load_state_dict(saved["state_dict"])
#         return 0
        # else:
        #     raise FileNotFoundError("the file does not exist!")
        # Add root_path in front of the path of the saved network parameters
        # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters

    def transform_obs_to_feature(self, obs):
        _, vision, _, _ = obs
        #
        # return list(features) + list(scent) + list(vision)
        return list(vision)
        #

    def act(self, curr_obs, mode='eval'):
        if mode == "train":
            self.step += 1
        return self.get_action(mode, self.get_eps(self.step), self.env_specs["action_space"], self.online_net,
                               self.transform_obs_to_feature(curr_obs))

    def update(self, curr_obs, action, reward, next_obs, done, timestep, verbose=False):
        cur__obs = self.transform_obs_to_feature(curr_obs)
        next__obs = self.transform_obs_to_feature(next_obs)
        transition = (cur__obs, action, reward, done, next__obs)
        self.replay_buffer.append(transition)
        if self.step < MIN_REPLAY_SIZE:  # just fill the buffer
            return

        transitions = random.sample(self.replay_buffer, BATCH_SIZE)

        obses_t, actions_t, rews_t, dones_t, new_obses_t = self.get_exp_from_buffer(transitions)

        target_q_values = self.target_net(new_obses_t)

        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        # update function #
        # targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values
        targets = rews_t + GAMMA * max_target_q_values
        # Compute Loss
        q_values = self.online_net(obses_t)  # action value
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)  # real values

        # loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        # Gradient Descent

        criterion = nn.MSELoss()
        loss = criterion(action_q_values, targets)
        # if verbose:
        # print("training loss = {}".format(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.step % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
