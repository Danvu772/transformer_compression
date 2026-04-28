# Transformer Based Lossless Source Coding

Research project exploring whether a transformer trained offline on sequences from a known source can beat a universal source coder, and if so, how much training data it actually needs to do that.

Advised by Prof. Anders Host-Madsen at University of Hawaii at Manoa.

---

## The Core Question

Universal source coders like the Laplace estimator work for any source without any prior knowledge. The obvious tradeoff is that they pay a redundancy penalty, which for a sequence of length N is roughly 0.5 * log(N) / N bits per symbol. A learned coder that was trained on sequences from the true source should theoretically do better since it already has a model of the distribution.

The question is how many training sequences m are actually needed before the learned coder starts winning. Naive information theoretic reasoning would suggest exponentially many. 
---

## Setup

The probability estimator is a GPT transformer (gpt-nano from minGPT, about 0.11M parameters). It takes a binary context sequence and outputs a probability estimate for the next bit. That estimate gets plugged into a standard arithmetic coder.

The key design choice is a frozen coder: the model is trained once on a fixed dataset and then never updated on the test sequences. This keeps things theoretically clean and avoids any data leakage.

The baseline is the Laplace estimator, which just counts observed bits and adds a Laplace prior. No training required.

Performance is measured in bits per symbol (BPS). Shannon entropy is the theoretical floor, so redundancy is BPS minus entropy.

---

## Experiments

### IID Sequences

The first set of experiments uses IID binary sequences at various bias levels (p = 0.1, 0.3, 0.5, 0.7, 0.9).

**General model** trained on a mixture across all p values:

This model does not beat Laplace. At p=0.1, N=10: transformer gets 0.6116 BPS vs Laplace at 0.6085. They converge together as N grows, both tracking the 0.5 * log(N) / N curve. Learning a general distribution does not give enough of an edge to overcome the universal coder on any specific one.

**Specialized models** trained only on one value of p:

This is where it gets interesting. A model trained only on p=0.3 sequences:

| N | Transformer BPS | Laplace BPS |
|---|---|---|
| 10 | 0.9140 | 0.9744 |
| 25 | 0.8748 | 0.9282 |
| 100 | 0.8837 | 0.9067 |

Entropy at p=0.3 is 0.8813 BPS. The transformer is within a few thousandths of entropy by N=100, while Laplace still has meaningful overhead. Same pattern holds for p=0.7.

The transformer advantage narrows as N grows, which is expected: Laplace converges to entropy eventually, just slower.

### Scaling with Training Data

How many training sequences does it take before the transformer model stops being worse than random and starts being useful?

Running a sweep over m = 10, 25, 50, 100, 200, 500, 1000, 5000, 10000 training sequences, evaluating on p=0.3 with N=100:

| Training sequences | Redundancy |
|---|---|
| 10 | ~0.042 |
| 100 | ~0.009 |
| 1000 | ~0.001 |
| 10000 | ~0.0001 |

Redundancy is decaying by roughly one order of magnitude for each order of magnitude increase in training data, which is consistent with the 1/m prediction from theory. This is probably the cleanest result in the whole project.

### Markov Chains

Binary Markov chains with varying stay probabilities (p_stay from 0 to 0.99). The transformer used here is the general IID model, so it was not trained on Markov data at all.

At high autocorrelation (p_stay = 0.99), the transformer still beats Laplace:

| N | Transformer BPS | Laplace BPS |
|---|---|---|
| 10 | 0.3836 | 0.4008 |
| 50 | 0.3456 | 0.3566 |
| 100 | 0.5137 | 0.5159 |

The causal attention mechanism seems to be picking up on the sequential correlation implicitly, even without being trained on Markov data. At p_stay = 0.5 or p_stay = 0 the advantage mostly disappears, as expected.

### Finite State Machines

4-state FSMs with emission probabilities [0.1, 0.35, 0.65, 0.9] and Dirichlet concentration parameter alpha controlling how structured the transitions are. High alpha means more concentrated, more predictable transitions.

This set of experiments also includes CTW (Context Tree Weighting) as a third comparison point. CTW is a more sophisticated universal compressor that uses context trees.

Results at alpha = 1.0:

| N | Laplace | Transformer | CTW |
|---|---|---|---|
| 10 | 1.0875 | 1.0104 | 1.0188 |
| 100 | 1.0190 | 1.0011 | 1.0263 |
| 499 | 1.0022 | 0.9950 | 0.9976 |

The transformer is beating both Laplace and CTW here. At high alpha (alpha = 10, very structured transitions), the transformer maintains its advantage throughout all sequence lengths. At low alpha (alpha = 0.1, near uniform transitions) the three methods converge and the differences are within noise.

---

## Project Structure

```
notebooks/         main experiment notebooks
  model_train.ipynb        training IID and specialized models
  experiment_runs.ipynb    main benchmarking suite
  markov_benchmark.ipynb   Markov and FSM experiments
  training_benchmark.ipynb scaling experiments
  plots.ipynb              result visualization

mingpt/            transformer implementation (minGPT)
  model.py         GPT architecture
  trainer.py       training loop

ctw/               context tree weighting baseline
  ctw.py

models/            saved model checkpoints
  iid_model.pt
  p03_model.pt
  p07_model.pt
  fsm_*.pt
  benchmark_*.pt

experiments/       pickled result data from all runs

paper/             LaTeX manuscript in progress
```

---

## Notes

The Laplace estimator is a strong baseline for pure IID sequences, which is part of why the general transformer does not beat it there. The specialized models work because they can concentrate probability mass around the true p, but that requires knowing the source distribution in advance, which is exactly the assumption we are allowed to make in the offline training regime.

The FSM results are probably the most interesting from a practical standpoint since real data has more structure than IID sequences. The fact that the transformer beats CTW there is a decent empirical argument for learned compression on structured sources.
