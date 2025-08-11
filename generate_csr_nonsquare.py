import argparse
import numpy as np
import scipy.sparse as ss
import scipy.io as sio

def main(args):
    nrows = int(args.rows)
    ncols = int(args.cols)
    nnz = int(args.nnz_per_row)
    M = ss.csr_matrix((nrows, ncols))
    for r in range(nrows):
        i = np.random.randint(1, ncols, nnz)
        v = np.random.randn(nnz)
        M[r, i] = v
        if r % 1000 == 0:
            print(r)
    sio.mmwrite(args.out_file + ".mtx", M)
    sio.mmwrite(args.out_file + "-transpose.mtx", M.T)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rows")
    parser.add_argument("cols")
    parser.add_argument("nnz_per_row")
    parser.add_argument("out_file")
    args = parser.parse_args()

    main(args)
