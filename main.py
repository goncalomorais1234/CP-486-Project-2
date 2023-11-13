# Importing the required library
import numpy as np

# Solving a Problem with Q-learning Algorithm
# Creating a Simple Input for Testing the Implementation

# Defining Variables
states = ['A', 'B', 'C', 'D']  
actions = ['Stay', 'Move']  
rewards = {
    ('A', 'Stay', 'A'): -1,
    ('A', 'Move', 'B'): -2,
    ('B', 'Stay', 'B'): -1,
    ('B', 'Move', 'A'): -2,
    ('B', 'Move', 'C'): -2,
    ('C', 'Stay', 'C'): -1,
    ('C', 'Move', 'B'): -2,
    ('C', 'Move', 'D'): 10,  
    ('D', 'Stay', 'D'): 0  
}

# Initializing the Q-table
Q_table = np.zeros((len(states), len(actions)))
learning_rate = 0.8
discount_factor = 0.95

# Implementing the Q-learning Algorithm
def q_learning(state, action, next_state, reward):
    current_state_index = states.index(state)
    action_index = actions.index(action)
    next_state_index = states.index(next_state)

    # Updating Q-value using the Q-learning formula
    Q_table[current_state_index, action_index] = (1 - learning_rate) * Q_table[current_state_index, action_index] + \
        learning_rate * (reward + discount_factor * np.max(Q_table[next_state_index, :]))

# Training the Q-learning Algorithm
for _ in range(1000):  # Running for a certain number of episodes
    # Randomly selecting a state and action
    state = np.random.choice(states)
    action = np.random.choice(actions)

    # Simulating the environment and getting the next state and reward
    next_state = np.random.choice(states)
    # Handling errors for undefined rewards
    reward = rewards.get((state, action, next_state), 0)  
    q_learning(state, action, next_state, reward)

# Testing the Learned Q-table
def get_optimal_action(state):
    state_index = states.index(state)
    optimal_action_index = np.argmax(Q_table[state_index, :])
    return actions[optimal_action_index]

# Testing with a starting state
current_state = 'A'
while current_state != 'D':
    print(f"Current State: {current_state}")
    optimal_action = get_optimal_action(current_state)
    print(f"Optimal Action: {optimal_action}")
    next_state = 'B' if optimal_action == 'Move' else current_state
    print(f"Next State: {next_state}")
    print("------")
    current_state = next_state

print("Reached the destination!")
