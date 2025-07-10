import gradio as gr
from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

def predict_next_word(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(fn=predict_next_word, inputs="text", outputs="text", title="Next Word Predictor").launch()
