# Companion Notebook Outline

Lives at `notebooks/companion.ipynb`. Mirrors the existing directory structure:
`IID/`, `ML/`, `Markov/`, `train_bounds/` — each with `models/`, `experiments/`, `figures/`
subdirectories. Created automatically if absent.

---

## Cell 0: Imports

```python
import sys, os, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib, matplotlib.pyplot as plt, matplotlib.ticker as mticker
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.append('../')
from mingpt.model import GPT
from mingpt.trainer import Trainer
from ctw.ctw import CTW
```

---

## Cell 1: Global Config

All hyperparameters defined once. Nothing below this cell touches a magic number.

```python
DEVICE        = 'mps'
SEQ_LEN       = 500          # training sequence length; block_size = SEQ_LEN - 1 = 499
NUM_SAMPLES   = 10000        # training sequences per model
ITERS         = 1000         # training iterations
LEARN_RATE    = 1e-4
BATCH_SIZE    = 64
CTW_DEPTH     = 8
NUM_TEST      = 100          # evaluation trials (standard)
NUM_TEST_BETA = 500          # evaluation trials for beta sweep (Exp 4)

N_VALUES      = [10, 25, 50, 100, 200, 300, 499]   # sequence lengths for evaluation
P_VALUES      = [0.1, 0.3, 0.5, 0.7, 0.9]          # IID bias sweep (Exp 1 & 2)
P_STAY_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]    # Markov p_stay sweep (Exp 5)
BETA_PRIORS   = [(1.0,1.0), (0.5,0.5), (5.0,5.0), (5.0,2.0), (2.0,5.0)]
DIRICHLET_ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
M_VALUES      = [10, 25, 50, 75, 100, 150, 200, 300, 500, 1000]  # training sizes (Exp 3)

FSM_EMISSION  = [0.1, 0.35, 0.65, 0.9]
FSM_T0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
FSM_T1 = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])

# Colour palette (consistent across all plots)
C_SPECIALIZED = '#002d4d'
C_IID         = '#5B9BD5'
C_LAPLACE     = '#177F6B'
C_CTW         = '#9C4732'
C_BAYES       = '#E07B39'
C_ENTROPY     = '#888888'
```

---

## Cell 2: Directory Setup

Creates the full directory tree if it doesn't exist. Uses existing directories as-is.

```python
DIRS = {
    'IID':         '../notebooks/IID',
    'ML':          '../notebooks/ML',
    'Markov':      '../notebooks/Markov',
    'train_bounds':'../notebooks/train_bounds',
}
for base in DIRS.values():
    for sub in ('models', 'experiments', 'figures'):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
```

---

## Cell 3: Source Generators

```python
def sample_iid(p, n):
    """Bernoulli IID sequence."""

def sample_markov(p_stay, n):
    """Binary symmetric Markov chain."""

def sample_dirichlet_markov(alpha, emissions, n, n_states=4):
    """4-state Markov with transition matrix ~ Dirichlet(alpha)."""

def generate_fsm_sequence(emission_probs, T0, T1, n, initial_state=None):
    """Deterministic 4-state FSM (Mealy machine)."""
```

---

## Cell 4: Dataset Classes

```python
class FixedModelDataset(Dataset):
    """Pre-generated IID sequences. p can be scalar, list, or callable."""

class BetaIIDDataset(Dataset):
    """p ~ Beta(alpha, beta) drawn fresh per sequence."""

class MarkovDataset(Dataset):
    """Binary symmetric Markov chain sequences."""

class DirichletMarkovDataset(Dataset):
    """4-state Markov with Dirichlet-random transition matrix per instance."""

class FSMDataset(Dataset):
    """Deterministic 4-state FSM sequences."""
```

---

## Cell 5: Model Infrastructure

```python
def model_configure(dataset, iters=ITERS, learn_rate=LEARN_RATE):
    """Returns (model, trainer) for gpt-nano on binary vocab."""

def train_run(model, trainer, path):
    """Train and save checkpoint to path."""

def load_model(path):
    """Load gpt-nano checkpoint from path."""

def train_or_load(path, dataset, iters=ITERS, learn_rate=LEARN_RATE):
    """Load from path if checkpoint exists, otherwise train and save."""
```

---

## Cell 6: Probability Estimators

```python
def laplace_p_array(seq):
    """(k+1)/(t+2) at each position."""

def bayes_optimal_p_array(seq, alpha, beta):
    """(k+alpha)/(t+alpha+beta) — oracle for Beta(alpha,beta) prior."""

def transformer_p_array(seq, model, device=DEVICE):
    """Single forward pass; p[0]=0.5, p[t]=P(x_t=1|x_0..x_{t-1})."""

def ctw_p_array(seq, depth=CTW_DEPTH):
    """CTW with causal alignment fix: pad depth+1 flat priors, use p_ones[:-1]."""

def sequential_bps(seq, p_array):
    """Cross-entropy codelength: sum(-log2 p(x_t)) / len(seq)."""
```

---

## Cell 7: Evaluation Helpers

```python
def evaluate_iid(model_dict, p_values, n_values, num_test):
    """
    Returns results[p][N][estimator] = {'mean': float, 'se': float}.
    model_dict keys: estimator names -> model objects (or None for analytic).
    """

def evaluate_markov(model_dict, p_stay_values, n_values, num_test):
    """Same structure, source is Markov."""

def evaluate_fsm_sweep(alpha_values, model_dir, n_values, num_test):
    """Dirichlet Markov sweep. Loads specialized models per alpha."""

def evaluate_fsm_benchmark(model, emission_probs, T0, T1, n_values, num_test):
    """Single FSM model vs Laplace vs CTW."""
```

---

## Cell 8: Plot Style + Helpers

```python
matplotlib.rcParams.update({...})  # shared style: sans-serif, no top/right spine, log x-ticks

def style_ax(ax): ...
def set_log_xticks(ax, n_values): ...
def plot_bps_grid(results, param_values, param_name, estimator_keys, colors, title, save_path): ...
def plot_bps_single(results, n_values, estimator_keys, colors, entropy, title, save_path): ...
def plot_redundancy_loglog(benchmark_results, laplace_val, m_values, save_path): ...
def plot_r2_bars(probe_results, model_labels, colors, title, save_path): ...
```

---

## Experiment 1: General Transformer Learns Laplace (IID)

**Paper:** both (Exp 1 / Exp A)  
**Models:** `IID/models/iid_model.pt` — trained on p ~ Uniform(0,1)  
**Experiment:** `IID/experiments/iid_experiment.pkl`  
**Figure:** `IID/figures/iid_bps_per_p.pdf`

```
train_or_load  →  iid_model.pt
evaluate at p ∈ P_VALUES, N ∈ N_VALUES, NUM_TEST trials
estimators: transformer (general), laplace
save pkl, plot 5-panel grid
```

---

## Experiment 2: Specialized Transformer Beats Laplace at Short Sequences

**Paper:** both (Exp 2 / Exp B)  
**Models:** `IID/models/p03_model.pt`, `IID/models/p07_model.pt`  
**Experiments:** `IID/experiments/p03_experiment.pkl`, `IID/experiments/p07_experiment.pkl`  
**Figures:** `IID/figures/iid_p03.pdf`, `IID/figures/iid_p07.pdf`

```
train_or_load p03, p07
evaluate at fixed p=0.3 and p=0.7, N ∈ N_VALUES
estimators: specialized transformer, general transformer (loaded from Exp 1), laplace
save pkl per p, plot single panel per p
```

---

## Experiment 3: Redundancy Scales as 1/m

**Paper:** EE 496 only (Exp 3)  
**Models:** `train_bounds/models/benchmark_{m}.pt` for m ∈ M_VALUES  
**Experiment:** `train_bounds/experiments/benchmark_results.pkl`  
**Figure:** `train_bounds/figures/redundancy_vs_training.pdf`

```
for m in M_VALUES: train_or_load benchmark_{m}.pt on p=0.3, num_samples=m
evaluate each at l=100, NUM_TEST=500 trials; redundancy = BPS - H(0.3)
laplace redundancy as flat baseline
save pkl, plot log-log redundancy vs m
```

---

## Experiment 4: Beta Prior Generalization

**Paper:** both (Exp C / Exp 2.5)  
**Models:** `ML/models/beta_{a}_{b}_model.pt` for each prior in BETA_PRIORS  
**Experiment:** `ML/experiments/prior_sweep_results.pkl`  
**Figure:** `ML/figures/fig1_bps_overview.pdf`

```
for (alpha, beta) in BETA_PRIORS: train_or_load beta_{a}_{b}_model.pt on BetaIIDDataset
evaluate at N ∈ N_VALUES, NUM_TEST_BETA trials
estimators: transformer, laplace, ctw, bayes_optimal
save pkl, plot 5-panel grid (one panel per prior)
```

---

## Experiment 5: Binary Symmetric Markov Chain

**Paper:** both (Exp D / Exp 4)  
**Models:** `Markov/models/markov_{p_stay}_model.pt` for each p_stay; reuses `IID/models/iid_model.pt`  
**Experiments:** `Markov/experiments/specialized_experiment_markov.pkl`, `Markov/experiments/iid_experiment_markov.pkl`  
**Figure:** `Markov/figures/markov_bps_grid_comparison.png`

```
for p_stay in P_STAY_VALUES: train_or_load markov_{p_stay}_model.pt on MarkovDataset
load iid_model.pt from IID/models/
evaluate at N ∈ N_VALUES, NUM_TEST trials
estimators: specialized transformer, iid transformer, laplace, ctw
save two pkls (iid results, specialized results), plot 6-panel comparison grid
```

---

## Experiment 6: Dirichlet Markov Sweep

**Paper:** EE 496 only (Exp 5)  
**Models:** `Markov/models/fsm_{alpha}_model.pt` for each alpha; reuses `IID/models/iid_model.pt`  
**Experiments:** `Markov/experiments/fsm_{alpha}_experiment.pkl` per alpha; `Markov/experiments/iid_model_fsm_alpha_sweep.pkl`  
**Figures:** `Markov/figures/fsm_bps_grid_specialized.pdf`, `Markov/figures/fsm_bps_grid_iid.pdf`

```
for alpha in DIRICHLET_ALPHAS: train_or_load fsm_{alpha}_model.pt on DirichletMarkovDataset
evaluate specialized + iid at N ∈ N_VALUES, NUM_TEST trials
estimators: specialized, iid, laplace, ctw
save pkls, plot two 6-panel grids (specialized, iid)
```

---

## Experiment 7: Deterministic FSM Benchmark

**Paper:** EE 496 only (Exp 6)  
**Model:** `Markov/models/fsm_final_model.pt`  
**Experiment:** `Markov/experiments/fsm_final_benchmark.pkl`  
**Figure:** `Markov/figures/fsm_benchmark.pdf`

```
train_or_load fsm_final_model.pt on FSMDataset(FSM_EMISSION, FSM_T0, FSM_T1)
evaluate at N ∈ N_VALUES, NUM_TEST trials
estimators: specialized transformer, laplace, ctw
save pkl, plot single BPS vs length panel
```

---

## Experiment 8: Attention Head Probe

**Paper:** ML 445 only (Exp E)  
**Models:** loads all 5 beta models from `ML/models/`  
**Experiment:** computed in-memory (no pkl needed — fast Ridge regression)  
**Figures:** `ML/figures/attention_r2.pdf`, `ML/figures/attention_r2_bayes.pdf`

```
load all 5 beta models
generate 50 fixed test seqs: p=0.5, len=499, seed=42
extract attention outputs: shape [50, 3_layers, 3_heads, 499, 16]
  requires _attn_output hook on CausalSelfAttention (add if not present)
compute targets: k_t/t and (k_t+alpha)/(t+alpha+beta) per model
fit Ridge(alpha=1e-3) probes, compute R² per layer per head
average over heads per layer
plot two grouped bar charts side by side
```

---

## Notes

- All `train_or_load` calls print whether they loaded or trained.
- Experiment cells check for the `.pkl` file before running; if found, load and skip.
- Plot cells always re-render from pkl (no guard) so figures regenerate on re-run.
- The CTW causal alignment fix (pad `depth+1` flat priors, use `p_ones[:-1]`) is
  applied consistently in `ctw_p_array` and must not be reverted.
