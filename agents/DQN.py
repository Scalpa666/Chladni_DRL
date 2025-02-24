import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque

from envs.simulatedPlate import SimulatedPlate


# Define QNetwork which inherits from the nn.Module class
class QNetwork(nn.Module):
    # Define the structure of the neural network
    def __init__(self, input_size, output_size):
        # Execute the constructor of the parent class nn.Module
        super(QNetwork, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    # Define the forward propagation process of the neural network
    def forward(self, x):
        # Apply the ReLU activation function
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  # 衰减率
        self.epsilon_min = epsilon_min  # 最小探索率

        self.q_network = QNetwork(state_size, action_size).float()  # 创建 Q 网络
        self.target_network = QNetwork(state_size, action_size).float()  # 创建目标网络
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)  # 优化器
        self.loss_fn = nn.MSELoss()  # 损失函数
        self.learning_rate = learning_rate

        self.replay_buffer = ReplayBuffer(10000)  # 经验回放池
        self.batch_size = 528  # 每次训练的批次大小

        # 同步目标网络
        self.update_target_network()

    def load_q_network(self, state_dict):
        self.q_network.load_state_dict(state_dict)

    def update_target_network(self):
        """将 Q 网络的权重复制到目标网络中"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state):
        # The choice between exploration and exploitation
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                # Reshape state to match the input size expected by the network
                k = np.reshape(state, (1, -1))
                # Convert to torch tensor for the model
                state_tensor = torch.tensor(k, dtype=torch.float32)

                # Gets the Q value of the current state
                q_values = self.q_network(state_tensor).detach().numpy()
                # Select Q for the largest action index
                action = np.argmax(q_values)

            return action

    def learn(self):
        """从经验回放池中采样，进行训练"""
        # If the experience pool does not have enough samples, do not train
        if self.replay_buffer.size() < self.batch_size:
            return  0

        # Sample from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists to numpy arrays before creating tensors
        # By converting the lists to NumPy arrays first, the operation of creating tensors becomes more efficient.
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # The predicted Q value of the QNetwork
        q_values = self.q_network(states).gather(1, actions)

        # Target Q value
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculated loss
        loss = self.loss_fn(q_values, target_q_values)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


def save_results_to_csv(train_list):
    df = pd.DataFrame(train_list, columns=['Episode', 'Action Number', 'Avg Reward', 'Avg_loss', 'Arrival'])
    # 创建一个DataFrame
    df = pd.DataFrame(train_list, columns=['Episode', 'Action Number', 'Avg Reward', 'Avg_loss', 'Arrival'])

    # 判断文件是否存在，如果存在则以附加模式写入（不写入列名）
    file_path = 'results/train/dqn_training_states.csv'
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        # 如果文件不存在，则创建文件并写入列名
        df.to_csv(file_path, mode='w', header=True, index=False)

    print("Training log has been saved to 'dqn_training_states.csv'")


# Save the model and optimizer after 1000 episodes
def save_model(model, optimizer, episode):
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, f'results/train/dqn_model_checkpoint_{episode}.pth')
    print(f"Model saved at episode {episode}.")


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    print(f"Model loaded from episode {episode}.")
    return state_dict


def train_dqn(agent, plate, num_episodes):
    train_states = []

    for episode in range(num_episodes):
        print(f"Current episode: {episode + 1}")

        # Generate initial state
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        action_num = 0
        done = False
        finish = False

        while not finish:
            action = agent.act(state)
            # Done equal to arrival
            # End when out of bounds or upon arrival
            next_state, reward, done, finish = plate.play(action)
            action_num += 1

            # Store experience to replay buffer
            state_1d = state.reshape(-1)
            next_state_1d = next_state.reshape(-1)
            agent.replay_buffer.push(state_1d, action, reward, next_state_1d, done)

            # Train model
            episode_loss += agent.learn()
            state = next_state

            episode_reward += reward

            # Update target network
            if episode % 10 == 0:
                agent.update_target_network()

        # Save the states during training
        avg_reward = episode_reward / action_num
        avg_loss = episode_loss / action_num
        train_states.append([episode, action_num, avg_reward, avg_loss, done])
        print(f"Episode {episode}/{num_episodes}, Action_num: {action_num}, Avg_reward: {avg_reward}, Avg_loss: {avg_loss}, Arrival: {done}")

        if episode % 1000 == 0:
            save_model(agent.q_network, agent.optimizer, num_episodes)
            save_results_to_csv(train_states)
            train_states = []

    save_results_to_csv(train_states)
    save_model(agent.q_network, agent.optimizer, num_episodes)

    return


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use GPU
        print("Using GPU")
    else:
        device = torch.device('cpu')  # Fall back to CPU
        print("Using CPU")

    # Initialization env
    data = scipy.io.loadmat('envs/vectorField_RL_2019_P2.mat')
    env = SimulatedPlate(data)

    # Initialize the DQN Agent
    state_size = env.state_dimension
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)

    # state_dict = load_model('../results/train/dqn_model_checkpoint_10.pth')
    # agent.load_q_network(state_dict)

    # Train DQN
    train_dqn(agent, env, num_episodes=10000)

# 绘制奖励曲线
# import matplotlib.pyplot as plt
#     plt.plot(rewards)
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.show()

