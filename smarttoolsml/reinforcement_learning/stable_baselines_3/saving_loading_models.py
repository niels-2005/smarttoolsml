import os


def save_model(save_path: str, model):
    """_summary_

    Args:
        save_path (str): _description_
        model (_type_): _description_

    Example usage:
        save_path = "ppo/best_model
        model = PPO("MlpPolicy", "Pendulum-v1", verbose=0).learn(8_000)

        save_model(save_path, model)
    """
    model.save(save_path)


def load_model(algorithm, save_path):
    """_summary_

    Args:
        algorithm (_type_): _description_
        save_path (_type_): _description_

    Example usage:
        from stable_baselines3 import PPO

        save_path = "ppo/best_model

        load_model(PPO, save_path)
    """
    loaded_model = algorithm.load(save_path)
    return loaded_model
