import torch
import torch.nn as nn
import torch.nn.functional as F


# define the critic network
class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.conv1 = nn.Conv2d(env_params['obs'][-1], 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(3136 + env_params['goal'], 512)
        self.fc2 = nn.Linear(512, env_params['action'])

    def forward(self, obs, goal):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 3136)
        x = torch.cat([x, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BackwardMap(nn.Module):
    def __init__(self, env_params, embed_dim):
        super(BackwardMap, self).__init__()
        self.fc1 = nn.Linear(env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.backward_out = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        backward_value = self.backward_out(x)
        return backward_value


class ForwardMap(nn.Module):
    def __init__(self, env_params, embed_dim):
        super(ForwardMap, self).__init__()
        self.embed_dim = embed_dim
        self.num_actions = env_params['action']
        self.conv1 = nn.Conv2d(env_params['obs'][-1], 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(3136 + embed_dim, 512)
        self.forward_out = nn.Linear(512, embed_dim * env_params['action'])

    def forward(self, obs, w):
        w = w / torch.sqrt(1 + torch.norm(w, dim=-1, keepdim=True) ** 2 / self.embed_dim)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 3136)
        x = torch.cat([x, w], dim=1)
        x = F.relu(self.fc1(x))
        forward_value = self.forward_out(x)
        return forward_value.reshape(-1, self.embed_dim, self.num_actions)
