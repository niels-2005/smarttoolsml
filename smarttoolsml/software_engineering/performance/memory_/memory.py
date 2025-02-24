from collections import Counter

import numpy as np


def mode_using_counter(n_integers):
    random_integers = np.random.randint(1, 100000, n_integers)
    c = Counter(random_integers)
    return c.most_common(1)[0][0]


if __name__ == "__main__":
    print(mode_using_counter(10000000))


# terminal: memray run memory.py
# output:
# Writing profile results into memray-memory.py.149719.bin
# Memray WARNING: Correcting symbol for aligned_alloc from 0x7cca2b857c50 to 0x7cca2c2a5c60
# 56573
# [memray] Successfully generated profile results.

# You can now generate reports from the stored allocation records.
# Some example commands to generate reports:

# /home/ubuntu/miniconda3/envs/ml/bin/python -m memray flamegraph memray-memory.py.149719.bin

# for generating plot: python -m memray flamegraph memray-memory.py.149719.bin
