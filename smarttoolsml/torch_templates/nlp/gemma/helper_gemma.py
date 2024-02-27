import keras
import keras_nlp

def generate_response(instruction: str, gemma_model: keras.Model, max_length: int = 512):
    """
    Generates a response using a Gemma model based on the provided instruction.

    Args:
      instruction (str): The instruction or query for which a response is desired. This could be a question, a command, or any text requiring an AI-based response.
      gemma_model (keras.Model): The pre-trained Gemma model used to generate the response.
      max_length (int, optional): The maximum length of the generated response. Default is 512 tokens.

    Example:
      instruction = "What will the weather be like tomorrow?"
      gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

      generate_response(instruction, gemma_model, max_length = 512)
    """
    response = ''
    prompt = f'Instruction:\n{instruction}\n\nResponse:\n{response}'
    print(gemma_model.generate(prompt, max_length=max_length))
