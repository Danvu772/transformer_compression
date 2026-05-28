# %% [markdown]
# # Prior Manipulation Experiment
# Tests whether a transformer trained on Beta(alpha, beta) prior converges to
# the corresponding Bayes-optimal estimator.
#
# Sweep: Beta(1,1), Beta(0.5,0.5), Beta(5,5), Beta(5,2), Beta(2,5)
# Estimators compared at each N:
#   - Laplace         : Bayes-optimal under wrong prior Beta(1,1)
#   - CTW             : prior-agnostic universal coder
#   - Transformer     : prior baked into weights from training
#   - Bayes-optimal   : oracle floor for the true prior

# %%
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle
import os
import sys

sys.path.append('./')
from mingpt.model import GPT
from mingpt.trainer import Trainer
from ctw.ctw import CTW  # gabeschamberg implementation

# %%
# -------------------------
# Config
# -------------------------
SEQ_LEN     = 500
NUM_SAMPLES = 10000
ITERS       = 500
LEARN_RATE  = 1e-4
DEVICE      = 'mps'
N_VALUES    = [10, 25, 50, 100, 200, 499]
NUM_TEST    = 500
CTW_DEPTH   = 8

PRIORS = [
    (1.0, 1.0),   # Beta(1,1)   - uniform; Bayes-optimal == Laplace here
    (0.5, 0.5),   # Beta(0.5,0.5) - bimodal, mass at extremes
    (5.0, 5.0),   # Beta(5,5)   - symmetric, concentrated toward 0.5
    (5.0, 2.0),   # Beta(5,2)   - asymmetric, skewed high
    (2.0, 5.0),   # Beta(2,5)   - asymmetric, skewed low
]

model_save_folder      = 'models/'
experiment_save_folder = 'experiments/'

# %%
# -------------------------
# Dataset
# -------------------------
class BetaIIDDataset(Dataset):
    """
    IID binary sequences where p ~ Beta(alpha, beta) is drawn fresh per sequence.
    Transformer must learn to adapt in-context to the unknown p.
    """
    def __init__(self, alpha, beta, seq_len=500, num_samples=10000):
        self.block_size = seq_len - 1
        self.alpha = alpha
        self.beta  = beta
        self.data  = []
        for _ in range(num_samples):
            p   = np.random.beta(alpha, beta)
            seq = np.random.binomial(1, p, seq_len)
            x   = torch.tensor(seq[:-1], dtype=torch.long)
            y   = torch.tensor(seq[1:],  dtype=torch.long)
            self.data.append((x, y))

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# %%
# -------------------------
# Model infrastructure
# -------------------------
def model_configure(dataset, iters=500, learn_rate=1e-4):
    model_config              = GPT.get_default_config()
    model_config.model_type   = 'gpt-nano'
    model_config.vocab_size   = 2
    model_config.block_size   = dataset.get_block_size()
    model                     = GPT(model_config).to(DEVICE)
    train_config              = Trainer.get_default_config()
    train_config.learning_rate = learn_rate
    train_config.max_iters    = iters
    train_config.num_workers  = 0
    train_config.device       = DEVICE
    trainer                   = Trainer(train_config, model, dataset)
    return model, trainer

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"  iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

def train_run(model, trainer, output_name):
    save_dir = model_save_folder + output_name
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    torch.save(
        {'state_dict': model.state_dict(), 'block_size': model.block_size},
        save_dir
    )
    print(f'Saved to {save_dir}')
    return model

def load_model(model_dir):
    checkpoint          = torch.load(model_dir, map_location=DEVICE, weights_only=False)
    model_config        = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = 2
    model_config.block_size = checkpoint['block_size']
    model               = GPT(model_config)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(DEVICE).eval()

def model_name(alpha, beta):
    a_str = str(alpha).replace('.', 'p')
    b_str = str(beta).replace('.', 'p')
    return f'beta_{a_str}_{b_str}_model.pt'

# %%
# -------------------------
# Probability estimators
# -------------------------
def bayes_optimal_p_array(sequence, alpha=1.0, beta=1.0):
    """
    Sequential Bayes-optimal estimator under Beta(alpha, beta) prior.
    At position i having seen k ones: p_hat = (k + alpha) / (i + alpha + beta)
    alpha=beta=1 recovers Laplace.
    """
    p_array = []
    k = 0
    for i, bit in enumerate(sequence):
        p_array.append((k + alpha) / (i + alpha + beta))
        k += bit
    return p_array

def transformer_p_array(sequence, model, device=DEVICE):
    """
    Transformer probability estimates at each position.
    Truncates input to model's block_size if necessary.
    Prior is implicitly encoded in model weights from training.
    """
    model.eval()
    seq = sequence[:model.block_size]
    x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)
        probs     = F.softmax(logits, dim=-1)
    # pad with 0.5 for any positions beyond block_size
    p_list = [0.5] + probs[0, :-1, 1].tolist()
    if len(sequence) > model.block_size:
        p_list += [0.5] * (len(sequence) - model.block_size)
    return p_list

def ctw_p_array(sequence, depth=CTW_DEPTH):
    """
    CTW sequential probability estimates.
    predict_sequence returns distributions[:,n] AFTER updating on seq[depth+n],
    so distributions[1, n] is the prediction for seq[depth+n+1], not seq[depth+n].
    Correct causal alignment: pad depth+1 flat priors, then use p_ones[:-1].
    Returns p_array of length N where p_array[i] = P(X_i = 1 | X_0,...,X_{i-1}).
    """
    tree = CTW(depth=depth, symbols=2)
    probs = tree.predict_sequence(sequence)
    p_ones = probs[1, :].tolist()
    return [0.5] * (depth + 1) + p_ones[:-1]

def sequential_bps(seq, p_array):
    """
    Bits per symbol given sequential probability estimates.
    """
    total_bits = 0.0
    for i, bit in enumerate(seq):
        p1 = np.clip(p_array[i], 1e-10, 1 - 1e-10)
        p0 = 1.0 - p1
        total_bits += -np.log2(p1 if bit == 1 else p0)
    return total_bits / len(seq)

# %%
# -------------------------
# Evaluation
# -------------------------
def evaluate_prior(alpha, beta, model, n_values=N_VALUES, num_test=NUM_TEST):
    """
    Evaluate four estimators at each N for a model trained on Beta(alpha, beta).

    Estimators:
      laplace       : Beta(1,1) Bayes-optimal — wrong prior, universal baseline
      ctw           : prior-agnostic universal coder
      transformer   : learned prior via training
      bayes_optimal : oracle floor for Beta(alpha, beta)
    """
    results = {
        N: {'laplace': [], 'ctw': [], 'transformer': [], 'bayes_optimal': []}
        for N in n_values
    }

    for _ in range(num_test):
        p        = np.random.beta(alpha, beta)
        full_seq = np.random.binomial(1, p, max(n_values) + 1).tolist()

        # compute p_arrays over the full sequence once, then slice
        laplace_full     = bayes_optimal_p_array(full_seq, alpha=1.0,  beta=1.0)
        bayes_full       = bayes_optimal_p_array(full_seq, alpha=alpha, beta=beta)
        transformer_full = transformer_p_array(full_seq, model)
        ctw_full         = ctw_p_array(full_seq)

        for N in n_values:
            seq = full_seq[:N]
            results[N]['laplace'].append(      sequential_bps(seq, laplace_full[:N]))
            results[N]['ctw'].append(          sequential_bps(seq, ctw_full[:N]))
            results[N]['transformer'].append(  sequential_bps(seq, transformer_full[:N]))
            results[N]['bayes_optimal'].append(sequential_bps(seq, bayes_full[:N]))

    summary = {}
    for N in n_values:
        summary[N] = {}
        for k, v in results[N].items():
            arr = np.array(v)
            summary[N][k] = {
                'mean': np.mean(arr),
                'std':  np.std(arr),
                'ci':   1.96 * np.std(arr) / np.sqrt(len(arr))
            }
    return summary

# %%
# -------------------------
# Main sweep
# -------------------------
all_results = {}

for alpha, beta in PRIORS:
    name       = model_name(alpha, beta)
    model_path = model_save_folder + name

    print(f'\n{"="*55}')
    print(f'Prior: Beta({alpha}, {beta})')
    print(f'{"="*55}')

    dataset = BetaIIDDataset(alpha, beta, seq_len=SEQ_LEN, num_samples=NUM_SAMPLES)

    if os.path.exists(model_path):
        print(f'Loading model from {model_path}')
        model = load_model(model_path)
    else:
        print(f'Training model...')
        os.makedirs(model_save_folder, exist_ok=True)
        model, trainer = model_configure(dataset, iters=ITERS, learn_rate=LEARN_RATE)
        model = train_run(model, trainer, name)

    summary = evaluate_prior(alpha, beta, model)
    all_results[(alpha, beta)] = summary

    print(f'\n  {"N":>5}  {"Laplace":>16}  {"CTW":>16}  {"Transformer":>18}  {"Bayes-Opt":>16}')
    print(f'  {"-"*79}')
    for N, vals in summary.items():
        print(
            f'  {N:>5}  '
            f'{vals["laplace"]["mean"]:>8.4f}±{vals["laplace"]["std"]:.4f}  '
            f'{vals["ctw"]["mean"]:>8.4f}±{vals["ctw"]["std"]:.4f}  '
            f'{vals["transformer"]["mean"]:>10.4f}±{vals["transformer"]["std"]:.4f}  '
            f'{vals["bayes_optimal"]["mean"]:>8.4f}±{vals["bayes_optimal"]["std"]:.4f}'
        )

# %%
# -------------------------
# Save results
# -------------------------
os.makedirs(experiment_save_folder, exist_ok=True)
save_path = experiment_save_folder + 'prior_manipulation_results.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(all_results, f)
print(f'\nResults saved to {save_path}')