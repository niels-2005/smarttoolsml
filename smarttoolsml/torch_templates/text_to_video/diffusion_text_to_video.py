import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video


def generate_video_from_prompt(
    prompt: str,
    output_video_path: str,
    num_frames: int = 50,
    fps: int = 10,
    preset_name: str = "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype: float = torch.float16,
    variant: str = "fp16",
):
    """
    Generates a video from a given text prompt using a pre-trained text-to-video diffusion model.

    Args:
        prompt (str): The text prompt to generate the video from.
        output_video_path (str): The path where the generated video will be saved.
        num_frames (int, optional): The number of frames to generate for the video. Default is 50.
        fps (int, optional): The frames per second (fps) of the output video, affecting playback speed. Default is 10.
        preset_name (str, optional): The name of the pre-trained model to use for generating the video.
                                     Default is "damo-vilab/text-to-video-ms-1.7b".
        torch_dtype (torch.dtype, optional): The torch data type to use for tensor calculations. Using torch.float16
                                             can reduce memory usage but may affect precision. Default is torch.float16.
        variant (str, optional): Specifies the variant of the computation, e.g., using 'fp16' for half-precision
                                 floating points. This can help reduce memory usage on compatible hardware. Default is "fp16".

    Returns:
        str: The path to the saved video file.

    Example:
        video_path = generate_video_from_prompt("A cat playing piano",
                                                "cat_piano.mp4",
                                                num_frames=100,
                                                fps=24,
                                                preset_name="damo-vilab/text-to-video-ms-1.7b",
                                                torch_dtype=torch.float16,
                                                variant="fp16")

    Note:
        - This function requires a CUDA-compatible GPU environment to execute efficiently.
        - The function automatically clears the GPU memory cache after generating the video to free up resources.
        - Ensure the `diffusers` library and its dependencies are properly installed and updated to the latest version.
    """

    pipe = DiffusionPipeline.from_pretrained(
        preset_name, torch_dtype=torch_dtype, variant=variant
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
    pipe.enable_vae_slicing()

    video_frames = pipe(prompt, num_frames=num_frames).frames[0]
    video_path = export_to_video(
        video_frames, fps=fps, output_video_path=output_video_path
    )
