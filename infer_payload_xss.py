#!/usr/bin/env python3
import os
import pickle
import argparse
import numpy as np
import tensorflow as tf

from SeqGAN.generator_xss import Generator

TEMPLATES = [
    '<img src=x onerror="{}">',
    '<form action="/" onsubmit="{}"><input type=submit></form>',
    '<body onload="{}">',
    '<a href="javascript:{}">click</a>',
    '<embed src="{}">',
]

def load_tokenizer(tok_path):
    with open(tok_path, 'rb') as f:
        tok = pickle.load(f)
    return tok

def build_and_load_generator(tokenizer, model_dir, seq_length, emb_dim=64, hidden_dim=64, start_token=0):
    vocab_size = len(tokenizer.word_index) + 1
    gen = Generator(vocab_size, emb_dim, hidden_dim, seq_length, start_token=start_token)
    # build
    gen.build(input_shape=(None, seq_length))
    weights_path = os.path.join(model_dir, 'generator_weights')
    gen.load_weights(weights_path)
    print(f"[Loaded] Generator weights from {weights_path}")
    return gen

def decode_sequence_between_markers(seq_ids, tokenizer, char_level=True):
    """
    Decode and return substring between literal markers '<START>' and '<END>' if present.
    If not found, return entire decoded string.
    """
    inv = tokenizer.index_word
    # build string by concatenating tokens (char_level True -> tokens are characters)
    toks = [inv.get(int(i), '') for i in seq_ids if int(i) != 0]
    text = ''.join(toks) if char_level else ''.join([t for t in toks if t != ''])
    # find markers
    s_idx = text.find('<START>')
    e_idx = text.find('<END>')
    if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
        return text[s_idx + len('<START>'): e_idx]
    # fallback: if only START present
    if s_idx != -1:
        return text[s_idx + len('<START>'):]
    # fallback: return whole
    return text

def generate_and_save_with_templates(gen, tokenizer, out_file, n_samples=100, batch_size=32,
                                     seq_length=60, char_level=True, temperature=1.0,
                                     min_len=7, max_len=100, top_k=0, use_template=True, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # try to get end token id if the tokenizer contains it
    end_id = tokenizer.word_index.get('<END>')
    start_id = tokenizer.word_index.get('<START>')

    samples = []
    n_batches = int(np.ceil(n_samples / float(batch_size)))
    for _ in range(n_batches):
        seqs = gen.generate(batch_size, temperature=temperature, top_k=top_k, eos_token=end_id, max_len=seq_length).numpy()
        for row in seqs:
            # decode between markers
            decoded = decode_sequence_between_markers(row, tokenizer, char_level=char_level)
            # trim length constraints
            if len(decoded) < min_len:
                # we can skip or pad; skip for quality
                continue
            if len(decoded) > max_len:
                decoded = decoded[:max_len]
            if use_template:
                template = np.random.choice(TEMPLATES)
                try:
                    payload = template.format(decoded)
                except Exception:
                    payload = decoded
            else:
                payload = decoded
            samples.append(payload)
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as fo:
        for p in samples[:n_samples]:
            fo.write(p + '\n')
    print(f"[Saved] {len(samples[:n_samples])} samples to {out_file}")

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

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"{args.attack}_samples.txt")

    generate_and_save_with_templates(
        gen, tokenizer, out_file,
        n_samples=args.n_samples, batch_size=args.batch_size,
        seq_length=args.seq_length, char_level=args.char_level,
        temperature=args.temperature, min_len=args.min_len, max_len=args.max_len,
        top_k=args.top_k, use_template=args.use_template, seed=args.seed
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True)
    parser.add_argument("--save_dir", default="save")
    parser.add_argument("--out_dir", default="generated")
    parser.add_argument("--seq_length", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--start_token", type=int, default=0)
    parser.add_argument("--char_level", action="store_true", help="tokenizer was char-level")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0, help="top-k sampling (0 = disabled)")
    parser.add_argument("--min_len", type=int, default=7)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--use_template", action="store_true", help="wrap generated body into HTML XSS templates")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
