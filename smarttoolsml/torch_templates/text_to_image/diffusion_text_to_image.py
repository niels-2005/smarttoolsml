import matplotlib.pyplot as plt
import torch
from diffusers import DiffusionPipeline


def generate_image_from_prompt(
    prompt: str,
    preset_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype: float = torch.float16,
    use_safetensors: bool = True,
    variant: str = "fp16",
    figsize: tuple[int, int] = (10, 10),
):
    """
    Generates an image based on a given text prompt using a pre-trained diffusion model.

    Args:
        prompt (str): The text prompt based on which the image will be generated.
        preset_name (str): The name of the pre-trained model to be loaded from Hugging Face's model repository.
                           Default is "stabilityai/stable-diffusion-xl-base-1.0".
        torch_dtype (torch.dtype): The data type for torch tensors. Default is torch.float16 for reduced memory usage.
        use_safetensors (bool): Indicates whether to use SafeTensors to potentially enhance memory efficiency. Default is True.
        variant (str): Specifies the variant of the model to be used, if applicable. Default is 'fp16' for reduced precision computation.
        figsize (tuple[int, int]): The size of the figure in which the generated image will be plotted. Default is (10, 10).

    Returns:
        PIL.Image.Image: The generated image as a PIL Image object.

    Example:
        image = generate_image_from_prompt("A futuristic city skyline at sunset",
                                           preset_name="stabilityai/stable-diffusion-xl-base-1.0",
                                           torch_dtype=torch.float16,
                                           use_safetensors=True,
                                           variant='fp16',
                                           figsize=(10, 10))
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(
        preset_name,
        torch_dtype=torch_dtype,
        use_safetensors=use_safetensors,
        variant=variant,
    )
    pipe.to(device)
    image = pipe(prompt=prompt).images[0]

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    return image
