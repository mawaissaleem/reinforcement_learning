# policy iteration
import gymnasium as gym
import numpy as np

THETA = 0.1
N_S = 16
N_A = 4
MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]
GAMMA = 0.999

MAPCONCAT = "".join(MAP)


def reward(c):
    if c == "G":
        return 1
    else:
        return 0


REWARD = np.array([reward(c) for c in MAPCONCAT])

HOLE = np.array([c == "H" for c in MAPCONCAT])


def model_prob(s, a):
    # if MAPCONCAT[s] == "G":
    #     sprime = 5
    # elif HOLE[s]:
    #     sprime = s
    # else:
    x, y = s % 4, s // 4
    if a == 0:  # left
        if x > 0:
            x -= 1
    elif a == 1:  # down
        if y < 3:
            y += 1
    elif a == 2:  # right
        if x < 3:
            x += 1
    elif a == 3:  # up
        if y > 0:
            y -= 1
    sprime = x + y * 4
    sprimevector = np.zeros((N_S,))
    sprimevector[sprime] = 1
    return sprimevector


def init():
    V = np.zeros((N_S))
    policy = np.zeros((N_S))
    return V, policy


def policy_eval(V, policy):
    while True:
        delta = 0.0
        for s in range(N_S):
            if MAPCONCAT[s] != "H" and MAPCONCAT[s] != "G":
                v = V[s]
                # (16 x 1) * ((16 x 1) + (16 x 1))
                V[s] = np.sum(model_prob(s, policy[s]) * (REWARD + GAMMA * V))
                if s == 14 and policy[s] == 2:
                    assert V[s] == 1
                delta = max(delta, np.abs(v - V[s]))
            # print(V)
        if delta < THETA:
            break

    return V


def policy_improvement(V, policy):
    policy_stable = True
    for s in range(N_S):
        old_action = policy[s]
        policy[s] = np.argmax(
            [np.sum(model_prob(s, a) * (REWARD + GAMMA * V)) for a in range(N_A)]
        )
        # if s == 14:
            # print(
            #     f"argmax of {[np.sum(model_prob(s, a) * (REWARD + GAMMA * V)) for a in range(N_A)]}"
            # )
        if old_action != policy[s]:
            policy_stable = False
    return policy_stable, policy


def main():
    V, policy = init()
    while True:
        V = policy_eval(V, policy)
        print(f"V={V}")
        policy_stable, policy = policy_improvement(V, policy)
        print(f"policy: {policy}")
        if policy_stable:
            break

    # print(f"optimal V is {V}")
    # print(f"optimal policy is {policy}")

    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
    observation, info = env.reset()

    while True:
        a = policy[observation]
        # print(a)
        observation, reward, terminated, truncated, info = env.step(int(a))

        if terminated or truncated:
            break


main()