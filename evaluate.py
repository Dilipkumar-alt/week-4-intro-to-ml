import math
from transformers import GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

model = GPT2LMHeadModel.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenized = dataset.map(lambda e: tokenizer(e["text"], truncation=True), batched=True, remove_columns=["text"])

def group_texts(examples):
    block_size = 128
    concat = sum(examples["input_ids"], [])
    total_len = (len(concat) // block_size) * block_size
    input_ids = [concat[i:i+block_size] for i in range(0, total_len, block_size)]
    return {"input_ids": input_ids, "labels": input_ids, "attention_mask": [[1]*block_size]*len(input_ids)}

data = tokenized.map(group_texts, batched=True)

args = TrainingArguments(output_dir="./eval")
trainer = Trainer(model=model, args=args, eval_dataset=data["validation"])
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
