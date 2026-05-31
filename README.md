# Transformer-Based Lossless Source Coding

Lossless compression reduces the data compression task to modeling the source distribution. The better the probability model, the shorter the average code. This project explores recent advancements with transformers and uses its' prediction capabilities to develop a learned coder.

We test on binary IID and Markov sequences, benchmarking the transformer against non-learned methods including the Laplace estimator and Context Tree Weighting (CTW). On IID data the transformer does not beat Laplace, which is already near optimal and requires no training. But on structured data with longer range dependencies, learned coding pays off. The core metric throughout is bits per symbol (BPS), where lower is better and the theoretical floor is Shannon entropy.

The paper is in [paper/](paper/). Experiments live in [notebooks/](notebooks/), the CTW implementation in [ctw/](ctw/), and the transformer (minGPT) in [mingpt/](mingpt/).
