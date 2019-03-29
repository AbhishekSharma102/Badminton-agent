import numpy as np
#import tensorflow as tf
import gym
import time
#import spinup
import badminton
#from spinup.utils.run_utils import setup_logger_kwargs

env = gym.make('badminton-v0')
env.reset()
for i in range(100000):
    env.render()
    action = env.action_space.sample()
    #print(action)
    env.step(action)

#logger_kwargs = setup_logger_kwargs('test0.txt')
#spinup.td3(lambda: gym.make('badminton-v0'), logger_kwargs=logger_kwargs)
