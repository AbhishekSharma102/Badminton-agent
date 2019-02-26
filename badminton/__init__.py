from gym.envs.registration import register

register(
    id='badminton-v0',
    entry_point='badminton.envs:BadmintonEnv',
)