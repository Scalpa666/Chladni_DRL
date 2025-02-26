import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from envs.simulatedPlateD import SimulatedPlateD


class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPONetwork, self).__init__()
        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Policy output: Probability distribution of actions
        self.logits = nn.Linear(128, output_dim)
        # Value function output: status value
        self.value = nn.Linear(128, 1)

    def forward(self, state):
        x = self.base(state)
        return self.logits(x), self.value(x)


class PPOAgent:
    """PPO Agent implementation"""
    def __init__(self, state_dim, action_dim):
        # Hyperparameters
        self.clip_epsilon = 0.2  # Clipping range
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.entropy_coeff = 0.01  # Entropy bonus coefficient
        self.epochs = 4  # Training epochs per update
        self.batch_size = 64  # Mini-batch size

        self.policy = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.mse_loss = nn.MSELoss()

    # Select action and return log probability + value
    def act(self, state):
        with torch.no_grad():
            logits, value = self.policy(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


    # Calculate generalized advantage estimation (GAE)
    def compute_gae(self, rewards, values, dones):
        advantages = []
        last_advantage = 0
        next_value = 0
        next_non_terminal = 0

        # Reverse temporal processing
        for t in reversed(range(len(rewards))):
            # Delta is the temporal difference error:
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            next_value = values[t]
            # Compute the GAE
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantage
            advantages.insert(0, advantage)

            next_non_terminal = 1 - dones[t]

        return torch.tensor(advantages, dtype=torch.float32)


    # Update policy using collected experiences
    def update(self, storage):
        # Convert storage to tensors
        states = torch.tensor(storage.states, dtype=torch.float32)
        actions = torch.tensor(storage.actions, dtype=torch.long)
        old_log_probs = torch.tensor(storage.log_probs, dtype=torch.float32)
        returns = torch.tensor(storage.returns, dtype=torch.float32)

        # Calculate advantage estimation using Monte Carlo
        # values = torch.tensor(storage.values, dtype=torch.float32)
        # advantages = returns - values

        # Use GAE to calculate advantage
        advantages = self.compute_gae(storage.rewards, storage.values, storage.dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            # Random mini-batch sampling
            indices = torch.randperm(len(states))

            # Traverse the experience set according to batch size
            for start in range(0, len(states), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                # Get batch data
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Calculate new policy
                logits, values = self.policy(batch_states)
                # Create a discrete probability distribution
                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Calculate probability ratio between the new policy and the old policy
                ratio = (new_log_probs - batch_old_log_probs).exp()


                # Compute the first surrogate objective which is ratio multiplied by the advantage
                surr1 = ratio * batch_advantages
                # Compute the second surrogate objective clipping ratio
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = self.mse_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


# Experience storage for PPO
class Storage:
    def __init__(self):
        self.advantages = None
        self.returns = None
        self.values = None
        self.log_probs = None
        self.dones = None
        self.rewards = None
        self.actions = None
        self.states = None
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.returns = []
        self.advantages = []

    # Store experience
    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    # Calculate returns
    def calculate_returns(self, gamma=0.99):
        returns = []
        advantages = []
        R = 0

        # Calculate discounted returns (reverse)
        for t in reversed(range(len(self.rewards))):
            R = self.rewards[t] + gamma * R * (1 - self.dones[t])
            returns.insert(0, R)

        # Store returns
        self.returns = returns


# Example training loop pseudocode
def train(agent, env, total_episodes):
    storage = Storage()

    for episode in range(total_episodes):
        print(f"Current episode: {episode + 1}")

        state = env.reset()
        done = False

        while not done:
            action, log_prob, value = agent.act(torch.FloatTensor(state))
            next_state, reward, done, _ = env.step(action)
            storage.store(state, action, reward, done, log_prob, value)
            state = next_state

        # Final value estimation
        # _, _, last_value = agent.act(torch.FloatTensor(state))

        # Calculate the returns using Monte Carlo methods after the current episode ends
        storage.calculate_returns()

        # Update the policy
        agent.update(storage)
        storage.reset()


if __name__ == "__main__":
    total_episodes = 1000

    # The data needed to create env
    data_path = "displacement_field50.csv"
    env = SimulatedPlateD(data_path)

    agent = PPOAgent(state_dim=env.state_dimension,
                     action_dim=env.action_size)

    train(agent, env, total_episodes)
