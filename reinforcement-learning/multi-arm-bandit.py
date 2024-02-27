import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def run_experiment(NUM_ARMS, NUM_STEPS, method, epsi=0.1):


    def pull(action, Q):
        reward = np.random.normal(loc=Q[action], scale=1)  
        return reward

    Q = np.random.normal(loc=np.random.random(), scale=1, size=NUM_ARMS)
    Q = np.ones(NUM_ARMS) * 0.5
    N = np.ones(NUM_ARMS)
    R = np.zeros(NUM_STEPS)
    action_values = defaultdict(list)
    # print("epsi, ", epsi)
    for step in range(NUM_STEPS):
    
        if np.random.rand() <= epsi:
            action = np.random.randint(0, NUM_ARMS)
        else:
            action = np.argmax(Q)

        reward = pull(action, Q)
        action_values[action].append(reward)
        if method == "sample_average":
            Q[action] = sum(action_values[action]) / len(action_values[action])+1

        elif method == "step_size":
            Q[action] = Q[action] + 1/(step+1) * (reward - Q[action])

        elif method == "ucb":
            Q[action] = Q[action] + (np.sqrt(2 * np.log(step+1)) / N[action]) 

        elif method == "non_stationary":
            Q[action] = Q[action] + 0.1 * (reward - Q[action])

        N[action] += 1
        R[step] = 0 if step == 0 else reward

    return Q, N-1, R


REPETITIONS = 100
NUM_STEPS = 10000
NUM_ARMS = 10

cumulative_rewards_sa_01 = np.zeros(NUM_STEPS)
num_actions_sa_01 = np.zeros(NUM_ARMS)

cumulative_rewards_sa_001 = np.zeros(NUM_STEPS)
num_actions_sa_001 = np.zeros(NUM_ARMS)

cumulative_rewards_sa_00 = np.zeros(NUM_STEPS)
num_actions_sa_00 = np.zeros(NUM_ARMS)

cumulative_rewards_step_size = np.zeros(NUM_STEPS)
num_actions_step_size = np.zeros(NUM_ARMS)

cumulative_rewards_ucb = np.zeros(NUM_STEPS)
num_actions_ucb = np.zeros(NUM_ARMS)

cumulative_rewards_non_stationary = np.zeros(NUM_STEPS)
num_actions_non_stationary = np.zeros(NUM_ARMS)



for rep in range(REPETITIONS):

    Q, N, R = run_experiment(NUM_ARMS, NUM_STEPS, method="sample_average", epsi=0.1)
    num_actions_sa_01 += N
    cumulative_rewards_sa_01 += R

    Q, N, R = run_experiment(NUM_ARMS, NUM_STEPS, method="sample_average", epsi=0.01)
    num_actions_sa_001 += N
    cumulative_rewards_sa_001 += R

    Q, N, R = run_experiment(NUM_ARMS, NUM_STEPS, method="sample_average", epsi=0.0)
    num_actions_sa_00 += N
    cumulative_rewards_sa_00 += R

    Q, N, R = run_experiment(NUM_ARMS, NUM_STEPS, method="step_size")
    num_actions_step_size += N
    cumulative_rewards_step_size += R

    Q, N, R = run_experiment(NUM_ARMS, NUM_STEPS, method="ucb")
    num_actions_ucb += N
    cumulative_rewards_ucb += R

    Q, N, R = run_experiment(NUM_ARMS, NUM_STEPS, method="non_stationary")
    num_actions_non_stationary += N 
    cumulative_rewards_non_stationary += R

average_rewards_sa_01 = cumulative_rewards_sa_01 / REPETITIONS
average_rewards_sa_001 = cumulative_rewards_sa_001 / REPETITIONS
average_rewards_sa_00 = cumulative_rewards_sa_00 / REPETITIONS
average_rewards_step_size = cumulative_rewards_step_size / REPETITIONS
average_rewards_ucb = cumulative_rewards_ucb / REPETITIONS
average_rewards_non_stationary = cumulative_rewards_non_stationary / REPETITIONS


plt.plot(average_rewards_sa_01, label="e=0.1")
plt.plot(average_rewards_sa_001, label="e=0.01")
plt.plot(average_rewards_sa_00, label="e=0.0")

plt.plot(average_rewards_step_size, label="step-size")
plt.plot(average_rewards_ucb, label="ucb")
plt.plot(average_rewards_non_stationary, label="non-stationary")

plt.xlabel("Steps")
plt.ylabel("Average Cumulative Reward")
plt.legend()

plt.savefig("multi-armed-bandit.png")
plt.show()