import torch
from transformers import pipeline


def load_larger_model_bfloat16():
    pipe = pipeline(
        model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device_map="auto"
    )
    output = pipe("This is a cool example!", do_sample=True, top_p=0.95)


def load_larger_model_8bit():
    pipe = pipeline(
        model="facebook/opt-1.3b",
        device_map="auto",
        model_kwargs={"load_in_8bit": True},
    )
    output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
