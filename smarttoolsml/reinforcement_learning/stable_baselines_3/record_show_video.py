import base64
from pathlib import Path

import gymnasium as gym
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder


def record_and_show_video(
    env_id,
    model,
    video_length=500,
    prefix="",
    video_folder="videos/",
    show_video: bool = True,
):
    """_summary_

    Args:
        env_id (_type_): _description_
        model (_type_): _description_
        video_length (int, optional): _description_. Defaults to 500.
        prefix (str, optional): _description_. Defaults to "".
        video_folder (str, optional): _description_. Defaults to "videos/".

    Example usage:
        env_id = "CartPole-v1"
        env = gym.make(env_id)
        model = PPO(MlpPolicy, env, verbose=0)
        video_length = 500
        prefix = "ppo_cartpole"
        video_folder = "ppo_cartpole"

        record_and_show_video(env_id, model, video_length, prefix, video_folder, show_video = True)
    """
    record_video(
        env_id=env_id,
        model=model,
        video_length=video_length,
        prefix=prefix,
        video_folder=video_folder,
    )

    if show_video:
        show_videos(video_folder=video_folder, prefix=prefix)


def show_videos(video_folder="", prefix=""):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_folder).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def record_video(env_id, model, video_length=500, prefix="", video_folder="videos"):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()
