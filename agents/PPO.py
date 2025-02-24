import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.distributions import Categorical

# 创建OpenAI Gym环境
env = gym.make('CartPole-v1')

# 设置超参数
lr = 0.0003
gamma = 0.99
eps_clip = 0.2
K_epochs = 4
batch_size = 64
update_timestep = 4000

# 神经网络：策略网络和价值网络
class PPO_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO_Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)  # 策略输出：动作的概率分布
        self.fc4 = nn.Linear(128, 1)  # 价值函数输出：状态值

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)  # 动作的logits
        value = self.fc4(x)  # 状态值
        return logits, value

# 创建PPO网络
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_net = PPO_Network(input_dim, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

# 经验回放存储
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.old_logprobs = []

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.old_logprobs = []

    def store(self, state, action, reward, next_state, done, old_logprob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.old_logprobs.append(old_logprob)

    def sample(self):
        return torch.tensor(self.states, dtype=torch.float32), \
               torch.tensor(self.actions, dtype=torch.long), \
               torch.tensor(self.rewards, dtype=torch.float32), \
               torch.tensor(self.next_states, dtype=torch.float32), \
               torch.tensor(self.dones, dtype=torch.float32), \
               torch.tensor(self.old_logprobs, dtype=torch.float32)

# 计算GAE
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages

# 训练过程
def train():
    buffer = ReplayBuffer()
    running_reward = 0
    timestep = 0

    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = policy_net(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())
            buffer.store(state, action.item(), reward, next_state, done, log_prob.item())

            state = next_state
            episode_reward += reward
            timestep += 1

            if timestep % update_timestep == 0:
                states, actions, rewards, next_states, dones, old_logprobs = buffer.sample()

                _, values = policy_net(states)
                _, next_values = policy_net(next_states)

                advantages = compute_gae(rewards, values.detach(), next_values.detach(), dones)
                advantages = torch.tensor(advantages, dtype=torch.float32)

                targets = advantages + values.detach()

                for _ in range(K_epochs):
                    logits, value = policy_net(states)
                    dist = Categorical(torch.softmax(logits, dim=-1))
                    new_logprobs = dist.log_prob(actions)

                    ratio = torch.exp(new_logprobs - old_logprobs)

                    surrogate_loss = ratio * advantages
                    clipped_surrogate_loss = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
                    loss = -torch.min(surrogate_loss, clipped_surrogate_loss).mean() + 0.5 * (targets - value).pow(2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                buffer.clear()

                running_reward = 0.9
