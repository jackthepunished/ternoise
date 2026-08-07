import numpy as np

def save_tensor(path, arr):
    arr = np.asarray(arr, dtype=np.int64)
    with open(path, "w") as f:
        f.write(str(arr.ndim) + " " + " ".join(str(d) for d in arr.shape) + "\n")
        f.write(" ".join(str(v) for v in arr.reshape(-1)) + "\n")

def load_tensor(path):
    toks = open(path).read().split()
    rank = int(toks[0])
    dims = [int(t) for t in toks[1:1 + rank]]
    vals = np.array([int(t) for t in toks[1 + rank:]], dtype=np.int64)
    return vals.reshape(dims)
