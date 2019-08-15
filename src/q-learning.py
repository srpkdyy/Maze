import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML


def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])

    pi = np.nan_to_num(pi)
    return pi


def get_action_and_next_s(s, Q, epsilon, pi_0):
    direction = ['up', 'right', 'down', 'left']
    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]

    if next_direction == 'up':
        action = 0
        s_next = s - 3
    elif next_direction == 'right':
        action = 1
        s_next = s + 1
    elif next_direction == 'down':
        action = 2
        s_next = s + 3
    else:
        action = 3
        s_next = s - 1

    return [action, s_next]

def Sarsa(s, a, r, next_s, next_a, Q, eta, gamma):
    goal = 8
    if next_s == goal:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * Q[next_s, next_a] - Q[s, a])

    return Q





def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 0
    s_a_history = [[0, np.nan]]

    while(1):
        a, next_s = get_action_and_next_s(s, Q, epsilon, pi)

        s_a_history[-1][1] = a
        s_a_history.append([next_s, np.nan])

        if next_s == 8:
            r = 1
            next_a = np.nan
        else:
            r = 0
            next_a, _ = get_action_and_next_s(next_s, Q, epsilon, pi)
        
        Q = Sarsa(s, a, r, next_s, next_a, Q, eta, gamma)

        if next_s == 8:
            break
        else:
            s = next_s

    return [s_a_history, Q]


def init():
    line.set_data([], [])
    return (line,)


def animate(i):
    state = s_a_history[i][0]
    x = (state%3) + 0.5
    y = 2.5 - state//3
    line.set_data(x, y)
    return (line,)


fig = plt.figure(figsize=(5,5))
ax = plt.gca()

plt.plot([1, 1], [0, 1], color='red', linewidth=2)
plt.plot([1, 2], [2, 2], color='red', linewidth=2)
plt.plot([2, 2], [2, 1], color='red', linewidth=2)
plt.plot([2, 3], [1, 1], color='red', linewidth=2)

plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')
plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 1.5, 'S5', size=14, ha='center')
plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

plt.tick_params(axis='both', which='both', bottom='off', top='off',
labelbottom='off', right='off', left='off', labelleft='off')

line, = ax.plot([0.5], [2.5], marker='o', color='g', markersize=60)

# plt.show()

#                    up, right, down, left
theta_0 = np.array([[np.nan, 1, 1, np.nan],
                    [np.nan, 1, np.nan, 1],
                    [np.nan, np.nan, 1, 1],
                    [1, 1, 1, np.nan],
                    [np.nan, np.nan, 1, 1],
                    [1, np.nan, np.nan, np.nan],
                    [1, np.nan, np.nan, np.nan],
                    [1, 1, np.nan, np.nan],
                    ])

[a, b] = theta_0.shape
Q = np.random.rand(a, b) * theta_0
pi_0 = simple_convert_into_pi_from_theta(theta_0)

eta = 0.1
gamma = 0.9
epsilon = 0.5
v = np.nanmax(Q, axis=1)
episode = 1
is_continue = True

while episode <= 100:
    print('episode :' + str(episode))

    epsilon /= 2

    s_a_history, Q = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)

    new_v = np.nanmax(Q, axis=1)
    print(np.sum(np.abs(new_v - v)))
    v = new_v

    print('迷路を解くのにかかったステップ：' + str(len(s_a_history) - 1))

    episode += 1


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(s_a_history), interval=200, repeat=False)

anim.save('sarsa.mp4')
print(Q)