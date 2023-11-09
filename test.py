import sys
import math
import scipy.io as sio


MAX_ELEMENT = 999999999

def tournament_tree_kth_largest(A, b, w_k, k):
    m = len(A)
    pot = int(math.ceil(math.log(m) / math.log(2.0)))
    T = []
    for i in range(pot+1):
        T.append([(-MAX_ELEMENT, -1)] * 2**i)

    # Initial push an element from all lists up
    # Fill the bottom row
    for i in range(m):
        if b[i] == 0:
            continue
        T[pot][i] = (A[i][b[i]-1], i)

    # Now propagate up the tree
    for l in range(pot-1, -1, -1):
        for j in range(2**l):
            # Need to propagate which list won too
            if T[l+1][j*2][0] > T[l+1][j*2+1][0]:
                T[l][j] = T[l+1][j*2]
            else:
                T[l][j] = T[l+1][j*2+1]

    winner = T[0][0][1]
    b[winner] -= w_k

    # So we don't have to pad the lists...
    if b[winner] == 0:
        T[pot][winner] = (-MAX_ELEMENT, winner)
    else:
        T[pot][winner] = (A[winner][b[winner]-1], winner)

    # Now just propagate the winning list
    for i in range(k-1):
        j = winner
        for l in range(pot-1, -1, -1):
            if T[l+1][(j // 2) * 2][0] > T[l+1][(j // 2) * 2 + 1][0]:
                T[l][j // 2] = T[l+1][(j // 2) * 2]
            else:
                T[l][j // 2] = T[l+1][(j // 2) * 2 + 1]
            j = j // 2
        winner = T[0][0][1]
        b[winner] -= w_k

        # So we don't have to pad the lists
        if b[winner] == 0:
            T[pot][winner] = (-MAX_ELEMENT, winner)
        else:
            T[pot][winner] = (A[winner][b[winner]-1], winner)


def tournament_tree_kth(A, b, w_k, k):
    m = len(A)
    pot = int(math.ceil(math.log(m) / math.log(2.0)))
    T = []
    for i in range(pot+1):
        T.append([(MAX_ELEMENT, -1)] * 2**i)

    # Initial push an element from all lists up
    # Fill the bottom row
    for i in range(m):
        if b[i] + w_k > len(A[i]):
            T[pot][i] = (MAX_ELEMENT, -1)
            continue
        T[pot][i] = (A[i][b[i]+w_k-1], i)

    # Now propagate up the tree
    for l in range(pot-1, -1, -1):
        for j in range(2**l):
            # Need to propagate which list won too
            if T[l+1][j*2][0] < T[l+1][j*2+1][0]:
                T[l][j] = T[l+1][j*2]
            else:
                T[l][j] = T[l+1][j*2+1]

    winner = T[0][0][1]
    b[winner] += w_k

    # So we don't have to pad the lists...
    if b[winner] + w_k > len(A[winner]):
        T[pot][winner] = (MAX_ELEMENT, winner)
    else:
        T[pot][winner] = (A[winner][b[winner]+w_k-1], winner)

    # Now just propagate the winning list
    for i in range(k-1):
        j = winner
        for l in range(pot-1, -1, -1):
            if T[l+1][(j // 2) * 2][0] < T[l+1][(j // 2) * 2 + 1][0]:
                T[l][j // 2] = T[l+1][(j // 2) * 2]
            else:
                T[l][j // 2] = T[l+1][(j // 2) * 2 + 1]
            j = j // 2
        winner = T[0][0][1]
        b[winner] += w_k

        # So we don't have to pad the lists
        if b[winner] + w_k > len(A[winner]):
            T[pot][winner] = (MAX_ELEMENT, winner)
        else:
            T[pot][winner] = (A[winner][b[winner]+w_k-1], winner)


def compute_lmax(A, b):
    return max([A[i][b[i]-1] for i in range(len(b)) if b[i] > 0])


# A is list of lists to partition
# p is the splitting point (number on left) (fN)
def variable_split(A, p, tournaments, debug=False):
    # Preprocessing: First compute padding and r
    m = len(A)
    b = [0] * m

    if p < m:
        tournament_tree_kth(A, b, 1, p)
        return b

    r = int(math.ceil(math.log(p / m) / math.log(2.0)))
    two_r = 2**r
    n_max = max([len(x) for x in A])
    alpha = math.floor(n_max / two_r)
    n = two_r * (alpha + 1) - 1
    if debug:
        print("m = %d, N = %d, r = %d, two_r = %d, n_max = %d, alpha = %d, n = %d" % (m, N, r, two_r, n_max, alpha, n))

    # Base case: determine f-partition of S(0) (f = p / (m * n))
    k = math.ceil(p / n * alpha)
#    k = int((p / n * alpha) + 0.5)
    if debug:
        print("k = ", k)

    tournament_tree_kth(A, b, two_r, k)
    lmax = compute_lmax(A, b)
    if debug:
        print("b = ", b, "lmax = ", lmax)

    # r iterative steps
    for k in range(r):
        Lsize = 0
        w_k = 2**(r - k - 1)
        target_size = math.ceil(p * (n // w_k) / n)
#        target_size = int((p * (n // w_k) // n) + 0.5)
        if debug:
            print("w_k = %d, target_size = %d" % (w_k, target_size))
        # add the decided elements
        for i in range(len(b)):
            Lsize += b[i] // w_k       # original formulation is power-of-2 so b_k is always even
        if debug:
            print("initial loop Lsize after adding decided: ", Lsize)
        for i in range(m):
            if b[i] + w_k > len(A[i]):
                continue
            undecided = A[i][b[i] + w_k - 1]
            if undecided < lmax:
                b[i] += w_k
                Lsize += 1
        if debug:
            print("initial loop Lsize after adding undecided: ", Lsize, b)
            print("pre-boundary (after moving undecided")
            print([A[i][x-1] for i, x in enumerate(b)])
            print("Lsize = %d, target_size = %d" % (Lsize, target_size))

        if Lsize == target_size:
            if debug:
                print("have f-partition")
        elif Lsize > target_size:
            if debug:
                print("moving %d largest elements from L to H" % (Lsize - target_size))
            tournament_tree_kth_largest(A, b, w_k, Lsize - target_size)
            tournaments.append((m, Lsize - target_size))
        else:
            if debug:
                print("move %d smallest elements from H to L" % (target_size - Lsize))
            tournament_tree_kth(A, b, w_k, target_size - Lsize)
            tournaments.append((m, target_size - Lsize))

        lmax = compute_lmax(A, b)
        if debug:
            print("new lmax = %d" % lmax, compute_lmax(A, b))

        if debug:
            print("post-boundary")
            print([A[i][x-1] for i, x in enumerate(b)])

    return b


def test_split(A, b, p):
    L = []
    for a in A:
        L += list(a)
    ref = sorted(L)[:p]
    L = []
    for i, b_i in enumerate(b):
        L += list(A[i][:b_i])
    test = sorted(L)
    for i in range(p):
        if ref[i] != test[i]:
            return False
    return True


if __name__ == '__main__':
    '''
    A = [
        [ 5, 8, 11, 15, 17, 20, 25, 29, 33, 34, 36, 45, 49, 55, 58, 60, 65 ],
        [ 3, 21, 25, 29, 30, 33, 38, 41, 44, 48, 52, 55, 60, 75, 81, 90 ],
        [ 2, 5, 7, 10, 13, 18, 22, 29, 30, 35, 40 ],
        [ 1, 2, 3, 8, 9, 12, 15, 22, 31, 32, 36, 44, 48, 49, 52, 56, 62, 66, 68, 71, 79, 80, 88, 90, 100 ],
        [ 1, 9, 22, 23, 24, 29, 39, 45, 48 ]
    ]
    B = []
    for a in A:
        B += a
    split_point = 26
    print(sorted(B)[:split_point])
    variable_split(A, split_point)
    '''

    '''
    A = [[ 1, 2, 6, 7, 9, 11, 15 ],
         [ 2, 8, 9, 17, 23, 24, 25 ],
         [ 6, 7, 9, 12, 23, 24, 25 ],
         [ 3, 8, 10, 13, 14, 17, 19 ]]
    B = []
    for a in A:
        B += a
    split_point = 14
    print(A)
    print(sorted(B)[:split_point])
    variable_split(A, split_point)
    '''

    M = sio.mmread(sys.argv[1])
    M = M.tocsr()
    M.sort_indices()

    # prefix sum row lengths and split on 2048
    row_sizes = []

    BLOCK_SIZE = 2048
    total = 0
    shorts = 0      # Number of short splits where k <= m (no splitting, just TT)
    longs = 0       # Number of long splits where f > 1/2
    long_rows = 0   # Number of splits with more than 128 lists
    singles = 0     # Number of splits with a single list (no split)
    multi_rows = 0  # Number of splits with row size > BLOCK_SIZE such that we have to split the same row twice
    
    tournaments = []
    tries = 0
    next_block = BLOCK_SIZE
    for i in range(M.shape[0]):
        row_size = 0
        for j in M.getrow(i).indices:
            total += M.getrow(j).nnz
            row_size += M.getrow(j).nnz 
        if total > next_block:
            if M.getrow(i).nnz > 32:
                long_rows += 1
            if M.getrow(i).nnz == 1:
                singles += 1
            if row_size > BLOCK_SIZE:
                multi_rows += 1
            print("split at %d, %d / %d, m = %d" % (i, total - next_block, row_size, M.getrow(i).nnz))
#            if (total - next_block) <= M.getrow(i).nnz:
#                shorts += 1
            if (total - next_block) / row_size > 0.5:
                longs += 1
            else:
                tries += 1
                # collect into A 
                A = []
                for j in M.getrow(i).indices:
                    A.append(M.getrow(j).indices)
                b = variable_split(A, total - next_block, tournaments)
                res = test_split(A, b, total - next_block)
                if not res:
                    print("ERROR: bad split at %d, %d" % (i, total - next_block))
                    break
            next_block += BLOCK_SIZE
        row_sizes.append(total)
    print("shorts: %d, longs: %d, long_rows: %d, singles: %d, multi_rows: %d" % (shorts, longs, long_rows, singles, multi_rows))
    print("tries: %d, tournaments: %d" % (tries, len(tournaments)))

    # Vectorized sorted search instead (merge, merge-path)
