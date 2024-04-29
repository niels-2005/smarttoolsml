import torch
import transformers


def llama(
    prompt: str,
    model_id: str = "meta-llama/Meta-Llama-3-8B",
    job: str = "text_generation",
):
    """
    Generates a response to a given text prompt using the specified LLaMA model.

    This function initializes a text generation pipeline with a specified LLaMA model for the given job type, configures
    the model to use bfloat16 data type for efficient computation, and automatically maps the model to the appropriate
    device (e.g., GPU or CPU) based on availability and compatibility. The function then uses this pipeline to generate
    a response to the provided text prompt.

    Args:
        prompt (str): The text prompt to which the model will generate a response.
        model_id (str, optional): The identifier for the LLaMA model to be used. Defaults to "meta-llama/Meta-Llama-3-8B".
        job (str, optional): The type of job for the pipeline to perform, such as 'text_generation'. Defaults to 'text_generation'.

    Returns:
        str: The generated text response from the model.

    Example usage:
        # Generate a response using the default LLaMA text generation pipeline
        response = llama("What is the capital of France?")
        print(response)
    """
    pipeline = transformers.pipeline(
        job,
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    answer = pipeline(prompt)
    return answer
