import argparse
import scipy.io as sio
import scipy.sparse as ss
import numpy as np

def main(fname, size=50000, k=4):
    nnz = k * size
    data = np.ones(nnz)
    cols = np.random.randint(1, size, size=nnz)
    rows = np.repeat(np.arange(size), k)
    A = ss.csr_matrix((data, (rows, cols)))
    A.sort_indices()
    print("A.shape = ", A.shape)
    sio.mmwrite(fname, A)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fname")
    parser.add_argument("size")
    parser.add_argument("k")
    args = parser.parse_args()

    main(args.fname, size=int(args.size), k=int(args.k))
