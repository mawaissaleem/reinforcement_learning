# policy iteration
import gymnasium as gym
import numpy as np
import pandas as pd

THETA = 0.1
N_S = 48
N_A = 4
MAP1= ("SSSSSSSSSSSS" * 3) + "S"
MAP2 = ("H" * 10) + "G"

MAP = MAP1 + MAP2
GAMMA = 0.999


def reward(c):
    if c == "G":
        return 0
    elif c == "H":
        return -100
    else:
        return -1


REWARD = np.array([reward(c) for c in MAP])
print(REWARD)
HILL = np.array([c == "H" for c in MAP])
print(HILL)

def model_prob(s, a):
    # if MAP[s] == "G":
    #     sprime = 5
    # elif HOLE[s]:
    #     sprime = s
    # else:
    x, y = s % 12, s // 12
    # print(x,y)
    if a == 3:  # left
        if x > 0:
            x -= 1
    elif a == 2:  # down
        if y < 3:
            y += 1
    elif a == 1:  # right
        if x < 11:
            x += 1
    elif a == 0:  # up
        if y > 0:
            y -= 1
    sprime = x + y * 12
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
        n_iterations = 0
        for s in range(N_S):
            # if MAP[s] != "G" and MAP[s] != "H":
            if MAP[s] == "S":
                v = V[s]
                # (16 x 1) * ((16 x 1) + (16 x 1))
                V[s] = np.sum(model_prob(s, policy[s]) * (REWARD + GAMMA * V))
                # if s == 14 and policy[s] == 2:
                #     assert V[s] == 1
                delta = max(delta, np.abs(v - V[s]))
            elif MAP[s] == "H":
                v = V[s]
                # (16 x 1) * ((16 x 1) + (16 x 1))
                V[s] = np.sum(model_prob(s, policy[s]) * (REWARD + GAMMA * V))
                # if s == 14 and policy[s] == 2:
                #     assert V[s] == 1
                delta = max(delta, np.abs(v - V[s]))
                # n_iterations += 1
        # print(V)
        if delta < THETA:
            break

    return V


def policy_improvement(V, policy):
    policy_stable = True
    for s in range(N_S):
        old_action = policy[s]
        # print(s)
        # print([model_prob(s, a) for a in range(N_A)])
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

    print(f"optimal V is {V}")
    print(f"optimal policy is {policy}")

    env = gym.make("CliffWalking-v0", render_mode="human")
    observation, info = env.reset()
    states = []
    action_taken = []
    while True:
        a = policy[observation]
        # print(a)
        states.append(observation)
        action_taken.append(a)
        print(f"{observation}: {a}")
        observation, reward, terminated, truncated, info = env.step(int(a))

        if terminated or truncated:
            break

    df = pd.DataFrame({"Current State": states, "Action Taken":action_taken}, index=None).set_index("Current State")    
    print(df)


# main()
# V, policy = init()
# V = model_prob(V, policy)
# V, n_iterations = policy_eval(V, policy)
# print(V)
# print(n_iterations)
main()

# V, policy = init()
# V = model_prob(36,2)
# print(V)