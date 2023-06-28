# monte carlo on frozen lake
import gymnasium as gym
import numpy as np

N_S = 16
N_A = 4
GAMMA = 0.999


def first_visit_mc(policy, env, T):
    V = np.zeros((N_S))
    Returns = [[]] * N_S

    for k in range(100):
        observation, info = env.reset()

        rewards = []
        observations = []
        for i in range(T):
            a = policy[observation]
            observation, reward, terminated, truncated, info = env.step(int(a))
            rewards.append(reward)
            observations.append(observation)
            if terminated or truncated:
                break

        G = 0
        for j in range(i+1):
            G = GAMMA * G + rewards[i]

            s_t = observations[j]

            if j == 0 or s_t not in observations[: j - 1]:
                Returns[s_t].append(G)
                V[s_t] = np.mean(Returns[s_t])
    return V

def main():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    policy = [2] * 16

    print(first_visit_mc(policy, env, 16))

main()