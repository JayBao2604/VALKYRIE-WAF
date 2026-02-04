#!/usr/bin/env python3
import os
import re
import pickle
import argparse
import numpy as np
import tensorflow as tf

# **Chú ý**: import đúng lớp Generator tương ứng với file generator_ssrf.py của bạn
from SeqGAN.generator_xss import Generator

# --- helpers ---------------------------------------------------------------

def load_tokenizer(tok_path):
    with open(tok_path, 'rb') as f:
        tok = pickle.load(f)
    return tok

def build_and_load_generator(tokenizer, model_dir, seq_length, emb_dim=64, hidden_dim=64, start_token=0):
    vocab_size = len(tokenizer.word_index) + 1
    gen = Generator(vocab_size, emb_dim, hidden_dim, seq_length, start_token=int(start_token))
    # build to create weights
    gen.build(input_shape=(None, seq_length))
    weights_path = os.path.join(model_dir, 'generator_weights')
    gen.load_weights(weights_path)
    print(f"[Loaded] Generator weights from {weights_path}")
    return gen

def decode_sequence_char(seq_ids, tokenizer):
    """Char-level decode: join chars (no spaces). Ignore pad(0)."""
    inv = tokenizer.index_word
    chars = []
    for idx in seq_ids:
        i = int(idx)
        if i == 0:
            continue
        ch = inv.get(i, '')
        chars.append(ch)
    return ''.join(chars)

# --- extract prefixes (schemes / hosts) from train file --------------------

SCHEME_HOST_RE = re.compile(
    r'(?P<scheme>[a-zA-Z][a-zA-Z0-9+\-.]*):\/\/(?P<host>\[[^\]]+\]|[^\/\s:]+(?::\d+)?)(?P<rest>\/.*)?'
)

def extract_scheme_hosts_from_file(path):
    """
    Return list of scheme+host strings found in training data, e.g. 'http://127.0.0.1:80'
    """
    prefixes = set()
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            m = SCHEME_HOST_RE.search(ln)
            if m:
                scheme = m.group('scheme')
                host = m.group('host')
                prefixes.add(f"{scheme}://{host}")
            else:
                # fallback: lines that begin with scheme without //
                if ln.startswith('file://') or ln.startswith('jar://') or ln.startswith('gopher://') or ln.startswith('ldap://') or ln.startswith('sftp://') or ln.startswith('ftp://') or ln.startswith('http://') or ln.startswith('https://'):
                    # try simplest split
                    parts = ln.split('/', 3)
                    if len(parts) >= 3:
                        prefixes.add(parts[0] + '//' + parts[2].split('/')[0])
    return sorted(prefixes)

# --- main generation -------------------------------------------------------

def ensure_prefix(body, scheme_hosts):
    """
    Ensure 'body' begins with scheme://host... If not, prepend a random scheme_host.
    We try simple heuristics: if body contains '://' then assume has scheme; else prepend.
    """
    if not body:
        return body
    if '://' in body:
        return body
    if scheme_hosts:
        pref = np.random.choice(scheme_hosts)
        # if body starts with '/' remove duplicate slashes
        if body.startswith('/'):
            return pref + body
        else:
            # if body starts with '[' or digit/alpha then insert separator if needed
            return pref + '/' + body
    else:
        # fallback: if body starts with a leading slash or path-like, add http://127.0.0.1
        return 'http://127.0.0.1' + ('/' + body if not body.startswith('/') else body)

def generate_and_save_variable_length(gen, tokenizer, out_file, scheme_hosts=None,
                                      n_samples=100, batch_size=32,
                                      seq_length=120, temperature=1.0,
                                      min_len=8, max_len=200, top_k=0,
                                      seed=None, eos_token=None):
    """
    Generate variable-length SSRF payloads.
    - scheme_hosts: list like ['http://127.0.0.1:80', 'ldap://[::]:2375', ...]
    """
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    max_len_effective = min(max_len, seq_length)
    samples = []
    n_batches = int(np.ceil(n_samples / float(batch_size)))

    for _ in range(n_batches):
        # gen.generate returns tensor (maybe int64); cast to numpy
        seqs = gen.generate(batch_size, temperature=temperature, top_k=top_k, eos_token=eos_token, max_len=seq_length).numpy()
        for row in seqs:
            decoded = decode_sequence_char(row, tokenizer)

            # strip markers if present
            if '<START>' in decoded and '<END>' in decoded:
                si = decoded.find('<START>') + len('<START>')
                ei = decoded.find('<END>')
                body = decoded[si:ei]
            elif '<START>' in decoded:
                si = decoded.find('<START>') + len('<START>')
                body = decoded[si:]
            else:
                body = decoded

            # normalize whitespace percent-encoding etc. (optional)
            body = body.strip()

            # if generated lacks scheme+host, prepend one from training data
            if '://' not in body and scheme_hosts:
                body = ensure_prefix(body, scheme_hosts)

            # final length checks
            if len(body) < min_len:
                continue
            if len(body) > max_len_effective:
                body = body[:max_len_effective]

            samples.append(body)
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as fo:
        for p in samples[:n_samples]:
            fo.write(p + '\n')

    print(f"[Saved] {len(samples[:n_samples])} samples to {out_file} (len {min_len}-{max_len_effective})")
    return samples[:n_samples]

# --- entrypoint ------------------------------------------------------------

def main(args):
    tok_path = os.path.join(args.save_dir, 'tokenizers', f"{args.attack}_tokenizer.pkl")
    model_dir = os.path.join(args.save_dir, 'models', args.attack)
    if not os.path.exists(tok_path):
        raise SystemExit(f"Tokenizer not found: {tok_path}")
    if not os.path.exists(model_dir):
        raise SystemExit(f"Model dir not found: {model_dir}")

    tokenizer = load_tokenizer(tok_path)
    print("[OK] Loaded tokenizer. Vocab size:", len(tokenizer.word_index) + 1)

    gen = build_and_load_generator(tokenizer, model_dir, seq_length=args.seq_length,
                                   emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                                   start_token=args.start_token)

    eos_id = tokenizer.word_index.get('<END>')

    # extract scheme+host candidates from training file if available
    train_path = os.path.join('data', args.attack, f'{args.attack}.txt')
    scheme_hosts = extract_scheme_hosts_from_file(train_path) if os.path.exists(train_path) else None
    if scheme_hosts:
        print(f"[INFO] Found {len(scheme_hosts)} scheme+host prefixes from training data (examples): {scheme_hosts[:8]}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"{args.attack}_samples.txt")

    generate_and_save_variable_length(
        gen, tokenizer, out_file, scheme_hosts=scheme_hosts,
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
    parser.add_argument("--seq_length", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--start_token", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
