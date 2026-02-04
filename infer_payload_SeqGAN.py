# infer_payload.py
import os
import pickle
import argparse
import numpy as np
import tensorflow as tf

# import Generator class (đường dẫn tuỳ repo)
from SeqGAN.generator import Generator

def load_tokenizer(tok_path):
    with open(tok_path, 'rb') as f:
        tok = pickle.load(f)
    return tok

def build_and_load_generator(tokenizer, model_dir, seq_length, emb_dim=64, hidden_dim=64, start_token=0):
    # vocab_size phải giống lúc train: len(word_index)+1
    vocab_size = len(tokenizer.word_index) + 1
    gen = Generator(vocab_size, emb_dim, hidden_dim, seq_length, start_token=start_token)
    # bắt buộc build mô hình trước khi load weights
    gen.build(input_shape=(None, seq_length))
    weights_path = os.path.join(model_dir, 'generator_weights')
    gen.load_weights(weights_path)
    print(f"[Loaded] Generator weights from {weights_path}")
    return gen

def decode_sequence(seq_ids, tokenizer, char_level=False):
    # seq_ids: list/1D np array of ints
    # remove padding (0) và map bằng tokenizer.index_word
    words = []
    for idx in seq_ids:
        i = int(idx)
        if i == 0:
            continue
        tok = tokenizer.index_word.get(i, '')
        # Keep OOV token as-is if present
        words.append(tok)
    if char_level:
        return ''.join(words)
    else:
        return ''.join([w for w in words if w != ''])

def generate_and_save_variable_length(gen, tokenizer, out_file, n_samples=100, batch_size=32,
                                      seq_length=60, char_level=False, temperature=1.0,
                                      min_len=7, max_len=100, seed=None):
    """
    Sinh n_samples; mỗi mẫu có length ngẫu nhiên trong [min_len, max_len_effective]
    Nếu max_len > seq_length thì max_len_effective = seq_length.
    """
    if seed is not None:
        np.random.seed(seed)

    max_len_effective = min(max_len, seq_length)
    if min_len < 1:
        raise ValueError("min_len must be >= 1")
    if min_len > max_len_effective:
        raise ValueError(f"min_len ({min_len}) > effective max_len ({max_len_effective}). Increase seq_length or lower min_len.")

    samples_ids = []
    n_batches = int(np.ceil(n_samples / float(batch_size)))
    for _ in range(n_batches):
        # gen.generate returns [batch, seq_length]
        seq_ids_batch = gen.generate(batch_size, temperature=temperature).numpy()
        for row in seq_ids_batch:
            # choose random length for this sample
            length = int(np.random.randint(min_len, max_len_effective + 1))
            truncated = row[:length].tolist()
            samples_ids.append(truncated)
            if len(samples_ids) >= n_samples:
                break

    # decode and write
    with open(out_file, 'w', encoding='utf-8') as fo:
        for ids in samples_ids[:n_samples]:
            payload = decode_sequence(ids, tokenizer, char_level=char_level)
            fo.write(payload + '\n')

    print(f"[Saved] {len(samples_ids[:n_samples])} samples to {out_file} (length range {min_len}-{max_len_effective})")

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

    generate_and_save_variable_length(
        gen, tokenizer, out_file,
        n_samples=args.n_samples, batch_size=args.batch_size,
        seq_length=args.seq_length, char_level=args.char_level,
        temperature=args.temperature,
        min_len=args.min_len, max_len=args.max_len,
        seed=args.seed
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True)
    parser.add_argument("--save_dir", default="save")
    parser.add_argument("--out_dir", default="generated")
    parser.add_argument("--seq_length", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--start_token", type=int, default=0)
    parser.add_argument("--char_level", action='store_true', help="tokenizer was char-level")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min_len", type=int, default=7, help="minimum payload length (>=1)")
    parser.add_argument("--max_len", type=int, default=100, help="maximum payload length (<= seq_length)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for sampling")
    args = parser.parse_args()
    main(args)
