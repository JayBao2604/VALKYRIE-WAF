#!/usr/bin/env python3
import os
import pickle
import argparse
import numpy as np
import tensorflow as tf

from SeqGAN.generator_rce import Generator  # hoặc thay bằng import generator nếu bạn để tên khác

# Load tokenizer
def load_tokenizer(tok_path):
    with open(tok_path, 'rb') as f:
        tok = pickle.load(f)
    return tok

# Build and load generator model
def build_and_load_generator(tokenizer, model_dir, seq_length, emb_dim=64, hidden_dim=64, start_token=0):
    vocab_size = len(tokenizer.word_index) + 1
    gen = Generator(vocab_size, emb_dim, hidden_dim, seq_length, start_token=start_token)
    gen.build(input_shape=(None, seq_length))
    weights_path = os.path.join(model_dir, 'generator_weights')
    gen.load_weights(weights_path)
    print(f"[Loaded] Generator weights from {weights_path}")
    return gen

# Convert token sequence to string
def decode_sequence_char(seq_ids, tokenizer):
    inv = tokenizer.index_word
    chars = []
    for idx in seq_ids:
        i = int(idx)
        if i == 0:
            continue
        ch = inv.get(i, '')
        chars.append(ch)
    return ''.join(chars)

# Load IP prefixes from training data
def load_ip_prefixes(payload_path):
    ips = []
    with open(payload_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ';' in line:
                ip = line.split(';', 1)[0]
                if ip.count('.') >= 1:
                    ips.append(ip)
    return list(set(ips))

# Generate and save payloads
def generate_and_save_variable_length(gen, tokenizer, out_file, ip_prefixes=None,
                                      n_samples=100, batch_size=32,
                                      seq_length=80, temperature=1.0,
                                      min_len=7, max_len=100, top_k=0,
                                      seed=None, eos_token=None):
    if seed is not None:
        np.random.seed(seed)
    max_len_effective = min(max_len, seq_length)
    samples_ids = []
    n_batches = int(np.ceil(n_samples / float(batch_size)))

    for _ in range(n_batches):
        seq_ids_batch = gen.generate(batch_size, temperature=temperature, top_k=top_k,
                                     eos_token=eos_token, max_len=seq_length).numpy()
        for row in seq_ids_batch:
            decoded = decode_sequence_char(row, tokenizer)

            # Lấy phần nằm trong <START> và <END>
            if '<START>' in decoded and '<END>' in decoded:
                si = decoded.find('<START>') + len('<START>')
                ei = decoded.find('<END>')
                body = decoded[si:ei]
            elif '<START>' in decoded:
                si = decoded.find('<START>') + len('<START>')
                body = decoded[si:]
            else:
                body = decoded

            # Ép tiền tố IP nếu không có
            if ';' not in body and ip_prefixes:
                prefix = np.random.choice(ip_prefixes)
                body = f"{prefix};{body}"

            if len(body) < min_len:
                continue
            if len(body) > max_len_effective:
                body = body[:max_len_effective]

            samples_ids.append(body)
            if len(samples_ids) >= n_samples:
                break
        if len(samples_ids) >= n_samples:
            break

    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as fo:
        for p in samples_ids[:n_samples]:
            fo.write(p + '\n')
    print(f"[Saved] {len(samples_ids[:n_samples])} samples to {out_file} (len {min_len}-{max_len_effective})")

# Entry point
def main(args):
    tok_path = os.path.join(args.save_dir, 'tokenizers', f"{args.attack}_tokenizer.pkl")
    model_dir = os.path.join(args.save_dir, 'models', args.attack)
    if not os.path.exists(tok_path):
        raise SystemExit(f"Tokenizer not found: {tok_path}")
    if not os.path.exists(model_dir):
        raise SystemExit(f"Model dir not found: {model_dir}")

    tokenizer = load_tokenizer(tok_path)
    print("[OK] Loaded tokenizer. Vocab size:", len(tokenizer.word_index)+1)

    gen = build_and_load_generator(tokenizer, model_dir, seq_length=args.seq_length,
                                   emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                                   start_token=args.start_token)

    eos_id = tokenizer.word_index.get('<END>')

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"{args.attack}_samples.txt")

    # Load IP prefix từ file training
    ip_path = os.path.join('data', args.attack, f'{args.attack}.txt')
    ip_prefixes = load_ip_prefixes(ip_path) if os.path.exists(ip_path) else None

    generate_and_save_variable_length(
        gen, tokenizer, out_file, ip_prefixes=ip_prefixes,
        n_samples=args.n_samples, batch_size=args.batch_size,
        seq_length=args.seq_length, temperature=args.temperature,
        min_len=args.min_len, max_len=args.max_len,
        top_k=args.top_k, seed=args.seed, eos_token=eos_id
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True)
    parser.add_argument("--save_dir", default="save")
    parser.add_argument("--out_dir", default="generated")
    parser.add_argument("--seq_length", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--start_token", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--min_len", type=int, default=7)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
