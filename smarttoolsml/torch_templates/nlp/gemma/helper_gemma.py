import keras
import keras_nlp
import pandas as pd

tokenizer = keras_nlp.models.GemmaTokenizer.from_preset("gemma_instruct_2b_en")


def get_text_token_len(df_column: pd.Series) -> tuple[list, list]:
    """
    Calculates the length of texts and their tokenized forms from a pandas Series.

    Parameters:
    - df_column (pd.Series): A pandas Series containing text data to be analyzed.

    Returns:
    - tuple: A tuple containing two lists:
        - The first list contains the lengths of the texts in words.
        - The second list contains the lengths of the tokenized texts in tokens.
    """
    text_len = []
    token_len = []

    for text in df_column.values:
        length = len(text.split())
        text_len.append(length)

    for text in df_column.values:
        tokens = tokenizer(text)
        token_len.append(len(tokens))

    return text_len, token_len


def get_prep_gemma(instruction: list, output: list) -> pd.DataFrame:
    """
    Creates a DataFrame from instruction and output lists, preprocesses it for Gemma,
    and appends text and token lengths before returning the processed DataFrame.

    Parameters:
    - instruction (list): A list of instruction texts.
    - output (list): A list of outputs corresponding to each instruction.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the original instructions and outputs,
      preprocessed text for Gemma, lengths of preprocessed texts in words and tokens,
      with duplicates removed and sorted by token length in descending order.
    """

    df = pd.DataFrame({"instruction": instruction, "output": output})

    print(f"Len DataFrame: {len(df)}")
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"Len DataFrame without Dups/NaN's: {len(df)}")

    data = df.apply(
        lambda row: f"Instruction:\n{row['instruction']}\n\nResponse:\n{row['output']}",
        axis=1,
    ).values.tolist()

    df["prep_gemma"] = data

    text_len, token_len = get_text_token_len(df_column=df["prep_gemma"])

    df["text_len_gemma"] = text_len
    df["token_len_gemma"] = token_len
    df = df.sort_values(by="token_len_gemma", ascending=False)

    return df


def generate_response(
    instruction: str, gemma_model: keras.Model, max_length: int = 512
):
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
    response = ""
    prompt = f"Instruction:\n{instruction}\n\nResponse:\n{response}"
    print(gemma_model.generate(prompt, max_length=max_length))
