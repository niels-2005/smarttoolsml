import torch
import transformers


def gpt2(
    prompt: str,
    model_id: str = "gpt2-large",
    job: str = "text_generation",
    max_length: int = 50,
    num_return_sequences: int = 1,
    truncation: bool = True,
):
    """
    Generates a response to a given text prompt using the specified GPT2 model.

    This function initializes a text generation pipeline with a specified GPT2 model, tailored for the task specified by the 'job' parameter.
    It also sets specific model behaviors such as maximum length of the generated text, whether to truncate input if necessary,
    and the number of returned sequences. The model utilizes bfloat16 data type for efficient computation and is automatically
    mapped to an appropriate device (e.g., GPU or CPU) based on availability and compatibility.

    Args:
        prompt (str): The text prompt to which the model will generate a response.
        model_id (str, optional): The identifier for the GPT2 model to be used. Defaults to "gpt2-large".
        job (str, optional): The type of job for the pipeline to perform, such as 'text_generation'. Defaults to 'text_generation'.
        max_length (int, optional): The maximum length of the response text. Defaults to 50.
        num_return_sequences (int, optional): The number of different sequences to generate from the given prompt. Defaults to 1.
        truncation (bool, optional): Whether to truncate input text if it exceeds the maximum length supported by the model. Defaults to True.

    Returns:
        list or str: The generated text response(s) from the model. Returns a list if `num_return_sequences` is greater than 1, otherwise a string.

    Example usage:
        # Generate a response using the default GPT2 text generation pipeline
        response = gpt2("What is the capital of France?")
        print(response)
    """
    generator = transformers.pipeline(
        job,
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    texts = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences, truncation=truncation)
    return texts
