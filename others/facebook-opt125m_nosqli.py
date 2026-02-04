import torch
from datasets import Dataset
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

torch.cuda.empty_cache()
# payloads = pd.read_csv('data/nosqli/nosqli.txt', names=["payloads"], nrows=15000, on_bad_lines='skip')
payloads = pd.read_csv('data/nosqli/nosqli.txt', names=["payloads"], nrows=15000, on_bad_lines='skip')
# payloads = payloads.astype(str)

dataset = Dataset.from_pandas(payloads)
dataset = dataset.train_test_split(test_size=0.2) # 80% train, 20% test (loss)

model_name = "facebook/opt-125m" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
     

with open('data/nosqli/vocab.json') as file:
    vocab = json.load(file)

# Extract the values into a list
special_tokens = list(vocab.keys())
print(special_tokens)
     

tokenizer.add_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
     

def tokenize_function(examples):
    # Tokenize the text and prepare labels
    encoding = tokenizer(examples["payloads"], truncation=True, padding="max_length", max_length=128)
    encoding["labels"] = encoding["input_ids"]
    return encoding
     

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["payloads"])
     

training_args = TrainingArguments(
    output_dir="models/pretrain-models/facebook_opt125m-checkpoints_nosqli",
    per_device_train_batch_size=2,   
    per_device_eval_batch_size=2,
    eval_strategy="epoch",
    learning_rate=2e-5,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="models/pretrain-models/facebook_opt125m-checkpoints_nosqli",
    logging_steps=10,
    fp16=True,
    remove_unused_columns=False,
    report_to="none"
)
     

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)
     

# Fine-tune the model
torch.cuda.empty_cache()
trainer.train()
     

model.save_pretrained('models/pretrain-models/facebook_opt125m_nosqli')
tokenizer.save_pretrained('models/pretrain-models/facebook_opt125m_nosqli')
     