from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env


def build_model_with_eval_callback():
    env_id = "LunarLander-v2"
    n_envs = 16
    env = make_vec_env(env_id, n_envs=n_envs)

    # Create the evaluation envs
    eval_envs = make_vec_env(env_id, n_envs=5)

    # Adjust evaluation interval depending on the number of envs
    eval_freq = int(1e5)
    eval_freq = max(eval_freq // n_envs, 1)

    # Create evaluation callback to save best model
    # and monitor agent performance
    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path="./logs/",
        eval_freq=eval_freq,
        n_eval_episodes=10,
    )

    tensorboard_log = "./tb_logs/"

    # Instantiate the agent
    # Hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=64,
        gae_lambda=0.98,
        gamma=0.999,
        n_epochs=4,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    model.learn(
        total_timesteps=int(5e6),
        callback=eval_callback,
        progress_bar=True,
        log_interval=250,
    )
