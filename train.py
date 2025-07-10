
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], truncation=True)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

block_size = 128

def group_texts(examples):
    concat = sum(examples["input_ids"], [])
    total_len = (len(concat) // block_size) * block_size
    input_ids = [concat[i:i+block_size] for i in range(0, total_len, block_size)]
    return {"input_ids": input_ids, "labels": input_ids, "attention_mask": [[1]*block_size]*len(input_ids)}

data = tokenized.map(group_texts, batched=True)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=data["train"],
    eval_dataset=data["validation"]
)

trainer.train()
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
