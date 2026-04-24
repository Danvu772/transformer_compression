# %% [markdown]
# # Transformer based compression research project toolkit
# 
# This is a notebook with many numerical examples of concepts found in *Elements of Information Theory*
# 
# These functions may be useful in future experiments exploring universal source coding and transformer based coding. For now, they are used as an educational tool to reinforce concepts in the book.

# %%

import numpy as np
import matplotlib.pyplot as plt
from math import comb 
from itertools import combinations 

# %%
# Concepts: basic entropy calculations

x = [0, 0, 1, 1, 1, 2]

def convert_sample_to_pmf(sample):
    values, counts = np.unique(sample, return_counts=True)
    probs = counts/counts.sum()
    return dict(zip(values, probs))

def clean_pmf(pmf):
    return {k: v for k, v in pmf.items() if v > 0.0}

def generate_bin_pmf():
    p = np.random.rand()
    dict_out = {
        0: p,
        1: 1-p
    }
    return dict_out

def iid_generate(p_dict, N):
    p0 = p_dict[0]
    p1 = p_dict[1]
    return np.random.choice([0,1], size=N, p=[p0, p1])

def exact_binary_sample(p, N):
    count0 = int(round(p[0] * N))
    count1 = N - count0  

    samples = np.array([0]*count0 + [1]*count1)
    np.random.shuffle(samples)

    return samples

def entropy_calc(pmf):
    pmf = clean_pmf(pmf)
    pmf_array = np.asarray(list(pmf.values()))
    return -np.sum(pmf_array * np.log2(pmf_array))

# %%
# Concept: grouping values together means that entropy cannot increase

X = [1, 2, 3, 4, 5, 6, 7, 8]
X_pmf = convert_sample_to_pmf(X)

def coarsening_func(pmf_dict):
    pmf_fdict = {}
    for k, v in pmf_dict.items():
        if k in [1]:
            new_key = 'A'
        else:
            new_key = 'B'
        pmf_fdict[new_key] = pmf_fdict.get(new_key, 0) + v
    return pmf_fdict

print(f'H(f(X)) = {entropy_calc(coarsening_func(X_pmf))} is less than H(X) = {entropy_calc(X_pmf)}')

# %%
# Concepts: Conditional probability, chain rule

p_joint = {
    (0,0): 0.1,
    (0,1): 0.2,
    (0,2): 0.1,
    (1,0): 0.05,
    (1,1): 0.3,
    (1,2): 0.25
}

def p_X_marg(p_joint):
    p_X = {}
    for (x, y), p in p_joint.items():
        p_X[x] = p_X.get(x, 0) + p
    return p_X

def p_Y_marg(p_joint):
    p_Y = {}
    for (x, y), p in p_joint.items():
        p_Y[y] = p_Y.get(y, 0) + p
    return p_Y

def p_X_given_Y_convert(p_joint):
    p_Y = p_Y_marg(p_joint)
    p_X_given_Y = {}
    for (x, y), p in p_joint.items():
        cond = p / p_Y[y]
        p_X_given_Y[(x, y)] = cond
    return p_X_given_Y

def p_Y_given_X_convert(p_joint):
    p_X = p_X_marg(p_joint)
    p_Y_given_X = {}
    for (x, y), p in p_joint.items():
        cond = p / p_X[x]
        p_Y_given_X[(x, y)] = cond
    return p_Y_given_X

def cond_entropy_Y_given_X(p_joint):
    h_Y_X = 0
    p_X = p_X_marg(p_joint)
    for x in p_X:
        cond_pmf = {}
        for (x2, y), p in p_joint.items():
            if x2 == x:
                cond_pmf[y] = p / p_X[x]
        slice_entropy = entropy_calc(cond_pmf) 
        h_Y_X += p_X[x] * slice_entropy
    return h_Y_X

def cond_entropy_X_given_Y(p_joint):
    h_X_Y = 0
    p_Y = p_Y_marg(p_joint)
    for y in p_Y:
        cond_pmf = {}
        for (x, y2), p in p_joint.items():
            if y2 == y:
                cond_pmf[x] = p / p_Y[y]
        slice_entropy = entropy_calc(cond_pmf) 
        h_X_Y += p_Y[y] * slice_entropy
    return h_X_Y

# implementing chain rule checks
# verify H(X,Y) = H(X|Y) + H(Y)

H_XY = entropy_calc(p_joint)
H_Y_given_X = cond_entropy_Y_given_X(p_joint)
H_X = entropy_calc(p_X_marg(p_joint))

print(H_XY, H_Y_given_X + H_X)
print(f'error: {H_XY - (H_Y_given_X + H_X)}')


# %% [markdown]
# ## Jensen's inequality + concavity
# 
# The following code implements a concavity check of the entropy calc function for the binary case.  

# %%
# implement something like f(lambda * x1 + (1-lambda)*x2) <= lambda * f(x1) + (1-lambda) * f(x2)

def numerical_jansen_check(p, q, func):
    lams = np.linspace(0, 1, 50)
    results = []
    for lam in lams:
        m = {k: lam*p[k] + (1-lam)*q[k] for k in p.keys()}
        lhs = func(m)
        rhs = lam*func(p) + (1-lam)*func(q)
        results.append(lhs >= rhs)
    return all(results) 

pmf1 = generate_bin_pmf()
pmf2 = generate_bin_pmf()

# Verify the concavity of entropy function
entropy_concave_check = numerical_jansen_check(pmf1, pmf2, entropy_calc)
print(entropy_concave_check)


# %% [markdown]
# ### Kullback-Leibler divergence
# In the following python exercises we develop functions to calculate the KL divergence of two binary distributions 

# %%
def kl_div(p, q):
    p_val = np.asarray(list(p.values()))
    q_val = np.asarray(list(q.values()))
    mask = p_val > 0
    p_val = p_val[mask]
    q_val = q_val[mask]
    result = np.sum(p_val * np.log2(p_val/q_val))
    return result

probability = np.linspace(0, 1, 100)


kl_values = []

for prob in probability:
    p = {0: prob, 1:1-prob}
    q = {0: 0.5, 1:0.5}
    kl_values.append(kl_div(p, q))

plt.figure(figsize=(10, 4), dpi=150)
plt.plot(probability, kl_values)
plt.xlabel("p")
plt.ylabel("KL(p || q)")
plt.title("KL Divergence of Bernoulli p vs q=0.5")
plt.grid(True)
plt.show()




# %% [markdown]
# ## Shannon average code length calculation for realizable codes
# 
# When code lengths are expected to be integers which are an unescapable reality in real compression schemes, the word lengths must be rounded up to the nearest integer. 
# 
# Therefore, we calculate $ L = \sum_i p_i \, l_i $ using $ l_i = \lceil \log_2 \frac{1}{p_i} \rceil $

# %%

def shannon_average_code_length_calc_integer(pmf):
    output = 0
    for digit in pmf.keys():
        output += pmf[digit] * np.ceil(-np.log2(pmf[digit]))
    return output

p = generate_bin_pmf()
L = shannon_average_code_length_calc_integer(p)
H = entropy_calc(p)

print(f'p: {p}')
print(f'L: {L}')
print(f'H: {H}')
print(f'Test for if H < L < H+1: {H < L < H + 1}')

# %% [markdown]
# ## Kraft inequality 
# Below we implement a Kraft inequality verification function
# 
# We are implementing the following inequality:
# 
# $ \sum_{x \in \mathcal{X}} D^{-l(x)} \le 1 $
# 
# 

# %%
def kraft_inequality_verify(pmf):
    pmf = np.asarray(list(pmf.values()))
    return np.sum(2 ** -(np.ceil(-np.log2(pmf)))) <= 1


# %% [markdown]
# ### Universal source coding
# 
# We start to get into the meat of what we're really here for: getting average code lengths using what we know about universal source coding!
# 
# The Csiszar-Korner universal code length calculation comes from the Elements of Information theory book in chapter 13 (eq. 13.32):
# 
# $
# \ell(x^n) \le \log_2 (n + 1) + \log_2 \binom{n}{k} + 2
# $
# 
# 
# 

# %%
seq_test = exact_binary_sample(generate_bin_pmf(), 1000) 

pmf_find = convert_sample_to_pmf(seq_test)

def csiszar_korner_universal_code_length(seq):
    n = len(seq)
    k = np.sum(seq)
    ltype = np.ceil(np.log2(n+1)) 
    if k == 0 or k == n:
        lseq = 0
    else:
        lseq = np.ceil(np.log2(comb(n, k)))
    return ltype + lseq 

def expected_ck(seq, n):
    total = 0
    blocks = 0

    for i in range(0, len(seq), n):
        block = seq[i:i+n]
        if len(block) < n:
            block += [0]*(n - len(block))
        b_len = csiszar_korner_universal_code_length(block)
        total += b_len
        blocks += n

    return total / blocks
        

# %% [markdown]
# ## Encoding algorithm
# 
# The universal source coding calculation above only calculates the code length bit size
# 
# We will implement a type based universal source compressor to more concretely and empirically match our theoretical length calculation 

# %%

def int_to_bits(n, length):
    return [int(b) for b in bin(n)[2:].zfill(length)]

def lex_rank(seq):
    rank = 0
    ones_left = sum(seq)
    k = sum(seq)
    n = len(seq)
    bit_length = int(np.ceil(np.log2(comb(n, k))))

    if ones_left == 0:
        return int_to_bits(rank, bit_length)

    for i, bit in enumerate(seq):
        if bit == 1:
            rank += comb(n - i - 1, ones_left)
            ones_left -= 1
    return int_to_bits(rank, bit_length)


def type_encode(block):
    n = len(block)
    k = sum(block)

    bit_length_k = np.ceil(np.log2(n + 1))
    k_bits = int_to_bits(int(k), int(bit_length_k))
    return k_bits

def full_encode(full_seq, n):
    blocks = []
    full_array = []

    for i in range(0, len(full_seq), n):
        block = full_seq[i:i+n]
        if len(block) < n:
            block += [0]*(n - len(block))
        blocks.append(block)

    for block in blocks:
        typ = type_encode(block)
        idx = lex_rank(block)

        full_array.append(typ)
        full_array.append(idx)

    flat_stream = [bit for bits in full_array for bit in list(bits)]
    return flat_stream 


block_size = 20
compress_seq = iid_generate(generate_bin_pmf(), 10000) 
generated_p = convert_sample_to_pmf(compress_seq)
p = generated_p[1]
bps_empirical = len(full_encode(compress_seq, block_size))/len(compress_seq)
bps_shanon_real = shannon_average_code_length_calc_integer(generated_p)
bps_ex_ck = expected_ck(compress_seq, block_size)
bps_lower_limit = entropy_calc(generated_p)

print(f'Calculating with p = {p}')
print(f'bps_empirical: {bps_empirical}')
print(f'bps_ex_ck {bps_ex_ck}')
print(f'bps_shanon_real {bps_shanon_real}')
print(f'bps_lower_limit {bps_lower_limit}')






# %% [markdown]
# ## Encoding Algorithm 2
# 
# We develop another algorithm for universal source coding. This algorithm will simply sequentially calculate the bit length sequentially

# %%
def sequential_universal_source_coding(seq):
    total_bits = 0
    k = 0
    i = 0

    for bit in seq:
        p1 = (k+1) / (i+2)
        p0 = 1-p1

        if bit == 1:
            total_bits += -np.log2(p1)
        else:
            total_bits += -np.log2(p0)
        k += bit
        i += 1

    bits_per_symbol = total_bits / len(seq)
    return bits_per_symbol


bps_sequential = sequential_universal_source_coding(compress_seq) 
bps_lower_limit = entropy_calc(generated_p)

print(f'Calculating with p = {p}')
print(f'bps_sequential: {bps_sequential}')
print(f'bps_empirical: {bps_empirical}')
print(f'bps_ex_ck {bps_ex_ck}')
print(f'bps_lower_limit {bps_lower_limit}')

# %% [markdown]
# # N sweep
# 
# This figure examines the relationship between bitlength and size of the sequence 
# 
# Before doing that let me ask:
# - How should we perform a sweep when dealing with probabilities
# - I want to change N or sequence length and experiment with how that changes compression rates. Each experiment will have different results though, so I need guidance on how to run probabalistic experiments

# %% [markdown]
# ## minGPT
# 
# Here we start getting into minGPT. We will perform basic setup and run a few experiments. 

# %%
import minGPT



