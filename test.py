import gym, badminton

env = gym.make('badminton-v0')
env.reset()
for _ in range(100000):
    env.render()
    action = env.action_space.sample()
    #print(action)
    env.step(action)