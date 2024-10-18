from stable_baselines3.common.env_util import make_vec_env


def example_vec_env():
    env_id = "LunarLander-v2"
    n_envs = 16
    env = make_vec_env(env_id, n_envs=n_envs)
    return env
