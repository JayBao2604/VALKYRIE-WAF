"""
Pretrain LLM for WAF bypass payload generation.
Simple version optimized for Kaggle with OPT-125M.
"""

import torch
from datasets import Dataset
import json
import argparse
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def load_payloads(attack_type: str, data_dir: str = "data/generated", max_samples: int = None):
    """Load payloads for specified attack type."""
    payload_file = Path(data_dir) / attack_type / f"{attack_type}.txt"
    
    if not payload_file.exists():
        raise FileNotFoundError(f"Payload file not found: {payload_file}")
    
    print(f"Loading payloads from {payload_file}...")
    
    payloads = []
    with open(payload_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                payloads.append(line)
    
    print(f"Loaded {len(payloads)} payloads")
    return Dataset.from_dict({"text": payloads})

def load_vocab(attack_type: str, data_dir: str = "data/generated"):
    """Load vocabulary for specified attack type."""
    vocab_file = Path(data_dir) / attack_type / f"{attack_type}_vocab.json"
    
    if not vocab_file.exists():
        print(f"Warning: Vocab file not found: {vocab_file}")
        return []
    
    with open(vocab_file) as f:
        vocab = json.load(f)
    
    special_tokens = list(vocab.keys())
    print(f"Loaded {len(special_tokens)} special tokens")
    return special_tokens

def tokenize_function(examples, tokenizer):
    """Tokenize payloads with truncation."""
    encoding = tokenizer(
        examples["text"], 
        truncation=True,
        max_length=128,
        padding=False,
        return_attention_mask=True
    )
    return encoding

def main(args):
    """Main training function."""
    print(f"\n{'='*60}")
    print(f"Pretraining {args.model_name} on {args.attack_type.upper()} payloads")
    print(f"{'='*60}\n")
    
    # Load payloads
    dataset = load_payloads(args.attack_type, args.data_dir, args.max_samples)
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = load_vocab(args.attack_type, args.data_dir)
    if special_tokens:
        num_added = tokenizer.add_tokens(special_tokens)
        print(f"Added {num_added} special tokens")
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16 if args.fp16 else torch.float32,
    )
    
    # Resize embeddings if special tokens were added
    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Output directory
    output_dir = Path(args.output_dir) / args.attack_type / args.model_name.split('/')[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments - simple configuration for Kaggle
    # Auto-detect strategy based on dataset size
    use_steps_strategy = len(train_dataset) > 1000
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="steps" if use_steps_strategy else "epoch",
        eval_steps=200 if use_steps_strategy else None,
        save_strategy="steps" if use_steps_strategy else "epoch",
        save_steps=200 if use_steps_strategy else None,
        save_total_limit=2,
        fp16=args.fp16,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # Train
    print("\nStarting training...")
    print(f"Total training steps: {len(train_dataset) // args.batch_size * args.num_epochs}")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save metadata
    metadata = {
        "attack_type": args.attack_type,
        "model_name": args.model_name,
        "num_samples": len(dataset),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "special_tokens_added": len(special_tokens) if special_tokens else 0,
        "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", None)
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain LLM on attack payloads")
    
    # Required arguments
    parser.add_argument("--attack-type", type=str, required=True,
                       choices=["sqli", "xss", "cmdi", "nosqli", "ssrf"],
                       help="Type of attack to train on")
    
    # Optional arguments
    parser.add_argument("--model-name", type=str, 
                       default="facebook/opt-125m",
                       help="Pretrained model to use")
    parser.add_argument("--data-dir", type=str, 
                       default="data/generated",
                       help="Directory containing generated payloads")
    parser.add_argument("--output-dir", type=str, 
                       default="models/pretrained",
                       help="Directory to save pretrained models")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to use (for testing)")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                       help="Disable mixed precision training")
    parser.set_defaults(fp16=False)
    
    args = parser.parse_args()
    main(args)

