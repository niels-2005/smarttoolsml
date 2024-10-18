from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_rl_model(default_model, eval_env, n_eval_episodes: int):
    """_summary_

    Args:
        default_model (_type_): _description_
        eval_env (_type_): _description_
        n_eval_episodes (int): _description_

    Example usage:
        env_id = "Pendulum-v1"

        eval_env = gym.make(env_id)
        # or
        eval_envs = make_vec_env(env_id, n_envs=5)

        default_model = SAC(
            "MlpPolicy",
            "Pendulum-v1",
            verbose=1,
            seed=0,
            batch_size=64,
            policy_kwargs=dict(net_arch=[64, 64]),
        ).learn(8000)

        n_eval_episodes = 100

        evaluate_rl_model(default_model, eval_env, n_eval_episodes)
    """
    mean_reward, std_reward = evaluate_policy(
        default_model, eval_env, n_eval_episodes=n_eval_episodes
    )
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
