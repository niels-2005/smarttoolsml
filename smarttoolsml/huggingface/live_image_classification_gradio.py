import gradio as gr
from transformers import pipeline


def live_image_classification():
    pipe = pipeline("image-classification", model="google/vit-base-patch16-224")
    gr.Interface.from_pipeline(pipe).launch()
