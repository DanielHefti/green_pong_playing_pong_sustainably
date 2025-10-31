# Implementation of a Deep Q-Network (DQN) agent for reinforcement learning tasks. Inspired by the implementation of Mnih et al. (2015)
import gymnasium as gym # OpenAI Gym for environment simulation
import ale_py # ALE (Arcade Learning Environment) for Atari games
import torch # PyTorch for building and training neural networks
import torch.nn as nn # PyTorch's neural network module
from collections import deque # Deque for efficient appending and popping of elements
import random # Random module for sampling from the replay buffer
import numpy as np # NumPy for numerical operations
from gymnasium.wrappers import AtariPreprocessing # AtariPreprocessing wrapper for preprocessing Atari games
from codecarbon import track_emissions # CodeCarbon for tracking carbon emissions during training
#import cv2 # OpenCV for image processing

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN architecture
class DQN(nn.Module):
    # Initializes the DQN model with convolutional layers followed by fully connected layers.
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # Input: 4 stacked frames, Output: 32 feature maps
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), # Fully connected layer
            nn.ReLU(),
            nn.Linear(512, action_space), # Output layer: number of actions
        )

    # Forward pass through the network, computes the output of the network (Q-values) given an input state
    def forward(self, x):
        return self.net(x)
    
# Replay Buffer Class
class ReplayBuffer:
    # Initializes the replay buffer with a specified capacity
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # Buffer to store transitions

    # Push a new transition into the replay buffer
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done)) # Add a new transition to the buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # Randomly sample a batch of transitions from the buffer
        state, action, reward, next_state, done = zip(*batch) # Unzip the batch into individual components
        return (
            torch.tensor(np.array(state), dtype=torch.float32).to(device),
            torch.tensor(np.array(action), dtype=torch.int64).unsqueeze(1).to(device),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(next_state), dtype=torch.float32).to(device),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# Epsilon-greedy action selection
def select_action(state, epsilon, policy_net, action_space):
    if random.random() < epsilon: # With probability epsilon, select a random action
        return random.randrange(action_space) # Return a random action from the action space
    else: # Otherwise, select the action with the highest Q-value
        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device) # Convert state to tensor and move to device
            q_values = policy_net(state_tensor) # Get Q-values from the policy network
            return q_values.argmax().item() # Return the index of the action with the highest Q-value

# Function to train the DQN agent on the Atari Pong environment
@track_emissions()
def train_dqn():
    # Register the Atari environment
    gym.register_envs(ale_py) # Register the Atari environments in Gymnasium
    env = gym.make("PongNoFrameskip-v4") # Create the Pong environment with no frame skipping
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False) # Preprocess the environment with frame_skip=1 to avoid double frame-skipping
    env = gym.wrappers.FrameStackObservation(env, 4) # Stack 4 frames to create a state representation

    # Hyperparameters
    batch_size = 32 # Size of the batch for training
    gamma = 0.99 # Discount factor for future rewards
    buffer_capacity = 250000 # Capacity of the replay buffer (1 Miollion frames / 4 frame skip)
    target_update_freq = 10000 # Frequency of updating the target network
    epsilon_start = 1.0 # Initial value of epsilon for exploration
    epsilon_end = 0.1 # Final value of epsilon for exploration
    epsilon_decay = 0.0000036 #Exploration phase until 250000 steps (1 Million Framse / 4 frame skip = 250000 steps) # Decay rate for epsilon 
    max_steps = 12_500_000  # 50 million frames / 4 frame skip
    replay_start_size = 50000 # Minimum size of replay buffer before training starts
    update_frequency = 4 # Frequency of updating the target network

    # Parameters for early stopping
    activate_early_stopping = False # Flag to activate early stopping
    acceptable_threshold = 10.0 # Acceptable mean reward threshold for early stopping
    reward_window_size = 30 # Number of episodes to consider for mean reward

    # Initialize networks and optimizer
    action_space = env.action_space.n # Number of actions in the environment
    policy_net = DQN(action_space).to(device) # Policy network
    target_net = DQN(action_space).to(device) # Target network
    target_net.load_state_dict(policy_net.state_dict()) # Initialize target network with policy network weights
    target_net.eval() # Set target network to evaluation mode

    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95, eps=0.01) # Optimizer for training the policy network
    replay_buffer = ReplayBuffer(buffer_capacity) # Initialize the replay buffer

    # Training loop
    steps_done = 0 # Counter for total steps taken
    episode = 0 # Counter for episodes completed
    episode_rewards = [] # List to store rewards for each episode

    while steps_done < max_steps:
        state, _ = env.reset() # Reset the environment to start a new episode
        total_reward = 0 # Initialize total reward for the episode
        done = False # Flag to check if the episode is done

        while not done and steps_done < max_steps:
            # Epsilon decay for exploration
            epsilon = max(epsilon_end, epsilon_start - (steps_done * epsilon_decay))

            # Select action 
            action = select_action(state, epsilon, policy_net, action_space)

            # Take action in the environment
            next_state, reward, terminated, truncated, info = env.step(action) # Step the environment with the selected action
            done = terminated or truncated # Combine terminated and truncated to determine if episode is done
            reward = np.clip(reward, -1, 1)  # Clip reward to [-1, 1]
            replay_buffer.push(state, action, reward, next_state, done) # Store the transition in the replay buffer
            state = next_state # Update the current state
            total_reward += reward # Accumulate the total reward for the episode
            steps_done += 1 # Increment the steps

            # Train the policy network if enough samples are available in the replay buffer
            if len(replay_buffer) >= replay_start_size and steps_done % update_frequency == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Current Q-values
                q_values = policy_net(states).gather(1, actions) # Get Q-values for the selected actions

                # Target Q-values
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards + gamma * next_q_values * (1 - dones) # Compute target Q-values using the Bellman equation

                # Loss and optimization
                loss = nn.SmoothL1Loss()(q_values, target_q_values)  # Huber loss for stability
                optimizer.zero_grad() # Zero the gradients
                loss.backward() # Backpropagate the loss
                optimizer.step() # Update the policy network parameters
            
            # Update the target network periodically
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
        episode += 1 # Increment the episode counter
        episode_rewards.append(total_reward) # Store total reward for the episode
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}, Steps Done: {steps_done}") # Print episode statistics

        # Early stopping based on mean reward over last reward_window_size episodes
        if activate_early_stopping and len(episode_rewards) >= reward_window_size:
            mean_recent_reward = np.mean(episode_rewards[-reward_window_size:])
            if mean_recent_reward > acceptable_threshold:
                print(f"Early stopping: Mean reward over last {reward_window_size} episodes is {mean_recent_reward:.2f}, which is above the threshold {acceptable_threshold}.")
                break

    # Close the environment after training
    print(f"Training completed after {steps_done} steps and {episode} episodes.")
    print(f"Full list of rewards: {episode_rewards}") # Print the list of rewards for each episode
    env.close() # Close the environment to free resources

    # Save the trained model
    torch.save(policy_net.state_dict(), "dqn_pong.pth") # Save the policy network weights to a file
    # Note: The model can be loaded later using:
    # policy_net.load_state_dict(torch.load("dqn_pong.pth"))
    # and policy_net.eval() to set it to evaluation mode.
    # This allows the trained model to be used for inference or further training.

# Function to evaluate the trained DQN agent on the Atari Pong environment
# 5 minutes of gameplay at 60Hz (60 × 60 × 5 = 18,000 frames)
@track_emissions()
def evaluate_dqn(num_episodes=30, max_steps_per_episode=18000, epsilon=0.05):

    env = gym.make("PongNoFrameskip-v4")
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4)
    action_space = env.action_space.n

    policy_net = DQN(action_space).to(device)
    policy_net.load_state_dict(torch.load("dqn_pong.pth"))
    policy_net.eval()

    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        # Apply random number of no-op actions at the start (up to 30)
        no_op_steps = random.randint(0, 30)
        for _ in range(no_op_steps):
            state, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                state, _ = env.reset()

        while not done and steps < max_steps_per_episode:
            action = select_action(state, epsilon, policy_net, action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            steps += 1

        rewards.append(total_reward)
        print(f"Evaluation Episode {episode+1}: Total Reward = {total_reward}")

    avg_reward = np.mean(rewards)
    print(f"Average reward over {num_episodes} evaluation episodes: {avg_reward}")
    print(f"Full list of rewards during evaluation: {rewards}") # Print the list of rewards for each evaluation episode
    env.close()

if __name__ == "__main__":
    print("Starting DQN training on Atari Pong environment...")
    train_dqn() # Start training the DQN agent
    print("Training completed. Model saved as 'dqn_pong.pth'.")
    print("Starting evaluation of the trained DQN agent...")
    evaluate_dqn()
    print("Evaluation completed.")
    print("Average reward during evaluation: Check console output for details.")
    