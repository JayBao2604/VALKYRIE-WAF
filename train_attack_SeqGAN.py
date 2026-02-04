# train_attack_tf2.py
import os, argparse, pickle, numpy as np, tensorflow as tf
from SeqGAN.dataloader import SimpleDataLoader
from SeqGAN.generator_xss import Generator
from SeqGAN.discriminator import Discriminator
from SeqGAN.rollout_xss import ROLLOUT
from tensorflow.keras.optimizers.legacy import Adam

def save_generator_weights(generator, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'generator_weights')
    generator.save_weights(path)
    print("[Saved] generator weights to", path)

def load_generator_weights(generator, out_dir):
    path = os.path.join(out_dir, 'generator_weights')
    generator.build((None, generator.seq_length))
    generator.load_weights(path)
    print("[Loaded] generator weights from", path)

# def pretrain_generator(generator, data_loader, epochs=20, lr=1e-3):
#     optimizer = Adam(learning_rate=lr)
#     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
#     for ep in range(epochs):
#         data_loader.create_batches()
#         losses = []
#         for b in range(data_loader.num_batch):
#             batch = data_loader.next_batch()  # [batch, seq]
#             x_in = batch  # teacher forcing input
#             # target is same sequence
#             with tf.GradientTape() as tape:
#                 logits = generator(x_in, training=True)  # [batch, seq, vocab]
#                 # shift target? For SeqGAN original they predict next token; here use same alignment
#                 loss = loss_fn(batch, logits)  # [batch, seq]
#                 mask = tf.cast(tf.not_equal(batch, 0), tf.float32)
#                 loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
#             grads = tape.gradient(loss, generator.trainable_variables)
#             optimizer.apply_gradients(zip(grads, generator.trainable_variables))
#             losses.append(loss.numpy())
#         print(f"[Pretrain gen] epoch {ep} loss {np.mean(losses):.4f}")
def pretrain_generator(generator, data_loader, epochs=20, lr=1e-3):
    optimizer = Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    for ep in range(epochs):
        data_loader.create_batches()
        losses = []
        for b in range(data_loader.num_batch):
            batch = data_loader.next_batch()  # [batch, seq]
            # teacher forcing: input = batch[:, :-1], target = batch[:, 1:]
            x_in = batch[:, :-1]
            y_true = batch[:, 1:]
            with tf.GradientTape() as tape:
                logits = generator(x_in, training=True)  # expected [batch, seq-1, vocab]
                # compute per-position loss
                loss_per_pos = loss_fn(y_true, logits)  # [batch, seq-1]
                mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)  # pad id is 0
                loss = tf.reduce_sum(loss_per_pos * mask) / (tf.reduce_sum(mask) + 1e-8)
            grads = tape.gradient(loss, generator.trainable_variables)
            # optional: clip grads
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, generator.trainable_variables))
            losses.append(loss.numpy())
        print(f"[Pretrain gen] epoch {ep} loss {np.mean(losses):.4f}")


def pretrain_discriminator(discriminator, real_loader, generator, epochs=3, generated_num=1000, lr=1e-4):
    opt = Adam(learning_rate=lr)
    bsize = real_loader.batch_size
    for ep in range(epochs):
        # generate negatives
        neg = []
        n_batches = int(np.ceil(generated_num / bsize))
        for _ in range(n_batches):
            samp = generator.generate(bsize).numpy()
            neg.append(samp)
        neg = np.vstack(neg)[:generated_num]
        # prepare dataset with labels
        real = []
        # we need as many real samples as neg
        real_loader.create_batches()
        while len(real) < len(neg):
            real.append(real_loader.next_batch())
        real = np.vstack(real)[:len(neg)]
        X = np.vstack([real, neg])
        y = np.concatenate([np.ones((len(real),1), dtype=np.float32), np.zeros((len(neg),1), dtype=np.float32)], axis=0)
        # shuffle
        idx = np.arange(len(X)); np.random.shuffle(idx)
        X = X[idx]; y = y[idx]
        # train
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(bsize)
        for xb, yb in dataset:
            with tf.GradientTape() as tape:
                preds = discriminator(xb, training=True)
                loss = tf.keras.losses.binary_crossentropy(yb, preds)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, discriminator.trainable_variables)
            opt.apply_gradients(zip(grads, discriminator.trainable_variables))
        print(f"[Pretrain disc] epoch {ep} done")

def adversarial_train(generator, discriminator, rollout, data_loader, total_batch=200, gen_lr=1e-4):
    gen_opt = Adam(learning_rate=gen_lr)
    bsize = data_loader.batch_size
    for total in range(total_batch):
        # 1) train generator by policy gradient
        # sample
        samples = generator.generate(bsize).numpy()  # [batch, seq]
        # get rewards: shape [batch, seq]
        rewards = rollout.get_reward(None, samples, n_rollout=10, discriminator=discriminator)
        # compute log_probs for taken actions by running generator in teacher forcing on samples
        _, log_probs = generator.get_logits_for_sequences(tf.convert_to_tensor(samples))
        # log_probs: [batch, seq, vocab], pick taken tokens
        idx = tf.stack([tf.range(bsize)[:,None]*0 + tf.range(generator.seq_length)[None,:]], axis=0)  # not used; simpler use gather
        token_indices = tf.cast(samples, tf.int32)
        # build indices to gather: for each (i,t) gather log_probs[i,t, token]
        # use tf.range trick
        batch_idx = tf.reshape(tf.range(bsize)[:,None] * 0 + tf.range(generator.seq_length)[None,:], [-1])
        time_idx = tf.reshape(tf.range(generator.seq_length)[None,:] * 0 + tf.range(generator.seq_length)[None,:], [-1])
        # simpler: use tf.gather for last axis using token_indices flattened
        flat_log_probs = tf.reshape(log_probs, [-1, generator.vocab_size])  # [batch*seq, vocab]
        flat_token = tf.reshape(token_indices, [-1])  # [batch*seq]
        chosen_logp = tf.gather(flat_log_probs, flat_token, axis=1, batch_dims=0)  # NOT supported; alternative:
        # Workaround: one-hot multiply then sum
        one_hot = tf.one_hot(flat_token, depth=generator.vocab_size)
        chosen_logp = tf.reduce_sum(flat_log_probs * one_hot, axis=1)  # [batch*seq]
        chosen_logp = tf.reshape(chosen_logp, [bsize, generator.seq_length])  # [batch, seq]
        # policy gradient loss: - sum_t logp_t * reward_t
        pg_loss = -tf.reduce_mean(tf.reduce_sum(chosen_logp * rewards, axis=1))
        with tf.GradientTape() as tape:
            # compute loss again but we need gradients wrt generator params.
            # To compute grads, we need chosen_logp as a function of generator params
            # So recompute logits & log_probs within tape (we already did above but must be inside tape)
            logits = generator.call(tf.convert_to_tensor(samples), training=True)
            log_probs2 = tf.nn.log_softmax(logits, axis=-1)
            flat_log_probs2 = tf.reshape(log_probs2, [-1, generator.vocab_size])
            one_hot2 = tf.one_hot(tf.reshape(samples, [-1]), depth=generator.vocab_size)
            chosen_logp2 = tf.reshape(tf.reduce_sum(flat_log_probs2 * one_hot2, axis=1), [bsize, generator.seq_length])
            loss = -tf.reduce_mean(tf.reduce_sum(chosen_logp2 * rewards, axis=1))
        grads = tape.gradient(loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(grads, generator.trainable_variables))

        # 2) train discriminator for a few steps
        for _ in range(3):
            neg = []
            n_batches = 5
            for _ in range(n_batches):
                neg.append(generator.generate(bsize).numpy())
            neg = np.vstack(neg)[:generated_num if 'generated_num' in globals() else bsize * n_batches]
            # real
            data_loader.create_batches()
            real = []
            while len(real) < len(neg):
                real.append(data_loader.next_batch())
            real = np.vstack(real)[:len(neg)]
            X = np.vstack([real, neg])
            y = np.concatenate([np.ones((len(real),1)), np.zeros((len(neg),1))], axis=0)
            idx = np.arange(len(X)); np.random.shuffle(idx)
            X = X[idx]; y = y[idx]
            dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(bsize)
            dopt = Adam(learning_rate=1e-4)
            for xb, yb in dataset:
                with tf.GradientTape() as tape:
                    preds = discriminator(xb, training=True)
                    loss_d = tf.reduce_mean(tf.keras.losses.binary_crossentropy(yb, preds))
                grads = tape.gradient(loss_d, discriminator.trainable_variables)
                dopt.apply_gradients(zip(grads, discriminator.trainable_variables))

        if total % 10 == 0:
            print(f"[Adv] step {total} done")

def main(args):
    attack = args.attack
    save_dir = args.save_dir
    tok_path = os.path.join(save_dir, 'tokenizers', f"{attack}_tokenizer.pkl")
    real_path = os.path.join(save_dir, f"{attack}_real_data.txt")
    out_model_dir = os.path.join(save_dir, 'models', attack)

    if not os.path.exists(tok_path) or not os.path.exists(real_path):
        print("Run preprocess first")
        return

    tokenizer = pickle.load(open(tok_path,'rb'))
    vocab_size = len(tokenizer.word_index) + 1

    # hyperparams - tune as you like
    EMB_DIM = 64
    HIDDEN_DIM = 64
    SEQ_LENGTH = args.seq_length
    BATCH_SIZE = args.batch_size

    data_loader = SimpleDataLoader(real_path, BATCH_SIZE, SEQ_LENGTH)
    data_loader.create_batches()

    generator = Generator(vocab_size, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, start_token=0)
    # build by calling once
    _ = generator(tf.zeros([BATCH_SIZE, SEQ_LENGTH], dtype=tf.int32))

    discriminator = Discriminator(vocab_size, emb_dim=64, num_filters=[64,64,64], filter_sizes=[1,2,3], seq_length=SEQ_LENGTH)
    _ = discriminator(tf.zeros([BATCH_SIZE, SEQ_LENGTH], dtype=tf.int32))

    # pretrain generator MLE
    pretrain_generator(generator, data_loader, epochs=args.pretrain_epochs, lr=1e-3)
    # pretrain discriminator
    pretrain_discriminator(discriminator, data_loader, generator, epochs=3, generated_num=2000, lr=1e-4)

    # rollout
    rollout = ROLLOUT(generator, update_rate=0.8)

    # adversarial training
    adversarial_train(generator, discriminator, rollout, data_loader, total_batch=args.adversarial_steps)

    # save weights
    save_generator_weights(generator, out_model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True)
    parser.add_argument("--save_dir", default="save")
    parser.add_argument("--seq_length", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--adversarial_steps", type=int, default=100)
    args = parser.parse_args()
    main(args)
