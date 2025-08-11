import sys
import math
import struct
import scipy.io as sio
import numpy as np


MAX_ELEMENT = 9999999999

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
        if b[i] > len(A[i]):
            T[pot][i] = (MAX_ELEMENT-1, i)
        else:
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
    if winner == -1:
        print("FATAL: winner = -1")
        sys.exit(1)
    b[winner] -= w_k

    # So we don't have to pad the lists...
    if b[winner] == 0 or b[winner] > len(A[winner]):
        T[pot][winner] = (-(MAX_ELEMENT-1), winner)
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
        if winner == -1:
            print("FATAL: winner = -1")
            sys.exit(1)
        b[winner] -= w_k

        # So we don't have to pad the lists
        if b[winner] == 0 or b[winner] > len(A[winner]):
            T[pot][winner] = (-(MAX_ELEMENT-1), winner)
        else:
            T[pot][winner] = (A[winner][b[winner]-1], winner)


def tournament_tree_kth(A, b, w_k, k, debug=False):
    m = len(A)
    pot = int(math.ceil(math.log(m) / math.log(2.0)))
    T = []
    if debug:
        print("pot = ", pot, w_k, k)
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
#            print("checking (%d, %d) : (%d, %d)" % (l, j, T[l+1][j*2][0], T[l+1][j*2+1][0]), T[pot][0], T[pot][1], [len(a) for a in A])
            # Need to propagate which list won too
            if T[l+1][j*2][0] < T[l+1][j*2+1][0]:
                T[l][j] = T[l+1][j*2]
            else:
                T[l][j] = T[l+1][j*2+1]

    winner = T[0][0][1]
    if winner == -1:
        print("FATAL(1): winner = -1")
        sys.exit(1)
    b[winner] += w_k
    if debug:
        print("winner = ", winner, b[winner], len(A[winner]))

    # So we don't have to pad the lists...
    if b[winner] + w_k > len(A[winner]):
        T[pot][winner] = (MAX_ELEMENT-1, winner)
    else:
        T[pot][winner] = (A[winner][b[winner]+w_k-1], winner)

    # Now just propagate the winning list
    for i in range(k-1):
        j = winner
        for l in range(pot-1, -1, -1):
            if debug:
                print("checking (%d, %d, %d) : (%d, %d)" % (i, l, (j//2)*2, T[l+1][(j//2)*2][0], T[l+1][(j//2)*2+1][0]))
            if T[l+1][(j // 2) * 2][0] < T[l+1][(j // 2) * 2 + 1][0]:
                T[l][j // 2] = T[l+1][(j // 2) * 2]
            else:
                T[l][j // 2] = T[l+1][(j // 2) * 2 + 1]
            j = j // 2
        winner = T[0][0][1]
        if winner == -1:
            print("FATAL(2): winner = -1, ", T[0][0])
            sys.exit(1)
        b[winner] += w_k
        if debug:
            print("winner = ", winner, b[winner], len(A[winner]))

        # So we don't have to pad the lists
        if b[winner] + w_k > len(A[winner]):
            T[pot][winner] = (MAX_ELEMENT-1, winner)
        else:
            T[pot][winner] = (A[winner][b[winner]+w_k-1], winner)


def tournament_tree_kth_reverse(A, b, w_k, k):
    m = len(A)
    pot = int(math.ceil(math.log(m) / math.log(2.0)))
    T = []
    for i in range(pot+1):
        T.append([(MAX_ELEMENT, -1)] * 2**i)

    # Initial push an element from all lists up
    # Fill the bottom row
    for i in range(m):
        if b[i] - w_k < 0:
            T[pot][i] = (MAX_ELEMENT, -1)
            continue
        T[pot][i] = (-A[i][b[i]-w_k], i)

    # Now propagate up the tree
    for l in range(pot-1, -1, -1):
        for j in range(2**l):
            # Need to propagate which list won too
            if T[l+1][j*2][0] < T[l+1][j*2+1][0]:
                T[l][j] = T[l+1][j*2]
            else:
                T[l][j] = T[l+1][j*2+1]

    winner = T[0][0][1]
    if winner == -1:
        print("FATAL: winner = -1")
        sys.exit(1)
    b[winner] -= w_k

    # So we don't have to pad the lists...
    if b[winner] - w_k < 0:
        T[pot][winner] = (MAX_ELEMENT-1, winner)
    else:
        T[pot][winner] = (-A[winner][b[winner]-w_k], winner)

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
        if winner == -1:
            print("FATAL: winner = -1")
            sys.exit(1)
        b[winner] -= w_k

        # So we don't have to pad the lists
        if b[winner] - w_k < 0:
            T[pot][winner] = (MAX_ELEMENT-1, winner)
        else:
            T[pot][winner] = (-A[winner][b[winner]-w_k], winner)


def tournament_tree_kth_largest_reverse(A, b, w_k, k):
    m = len(A)
    pot = int(math.ceil(math.log(m) / math.log(2.0)))
    T = []
    for i in range(pot+1):
        T.append([(-MAX_ELEMENT, -1)] * 2**i)

    # Initial push an element from all lists up
    # Fill the bottom row
    for i in range(m):
        if b[i] == len(A[i]):
            continue
        T[pot][i] = (-A[i][b[i]], i)

    # Now propagate up the tree
    for l in range(pot-1, -1, -1):
        for j in range(2**l):
            # Need to propagate which list won too
            if T[l+1][j*2][0] > T[l+1][j*2+1][0]:
                T[l][j] = T[l+1][j*2]
            else:
                T[l][j] = T[l+1][j*2+1]

    winner = T[0][0][1]
    if winner == -1:
        print("FATAL: winner = -1")
        sys.exit(1)
    b[winner] += w_k

    # So we don't have to pad the lists...
    if b[winner] >= len(A[winner]):
        T[pot][winner] = (-(MAX_ELEMENT-1), winner)
    else:
        T[pot][winner] = (-A[winner][b[winner]], winner)

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
        if winner == -1:
            print("FATAL: winner = -1")
            sys.exit(1)
        b[winner] += w_k

        # So we don't have to pad the lists
        if b[winner] >= len(A[winner]):
            T[pot][winner] = (-(MAX_ELEMENT-1), winner)
        else:
            T[pot][winner] = (-A[winner][b[winner]], winner)


def compute_lmax(A, b):
    s = []
    for i in range(len(b)):
        if b[i] > 0:
            if b[i] > len(A[i]):
                s.append(MAX_ELEMENT-1)
                continue
            s.append(A[i][b[i]-1])
    return max(s)
#    return max([A[i][b[i]-1] for i in range(len(b)) if b[i] > 0])

def compute_lmax_reverse(A, b):
    return max([-A[i][b[i]] for i in range(len(b)) if b[i] < len(A[i])])

def compute_carry(A, b, lmax):
    for i, x in enumerate(b):
        if x < len(A[i]) and A[i][x] == lmax:
            return True
    return False

def compute_carry_reverse(A, b, lmax):
    for i, x in enumerate(b):
        if x > 0 and -A[i][x-1] == lmax:
            return True
    return False


# A is list of lists to partition
# p is the splitting point (number on left) (fN)
def variable_split(A, p, tournaments, debug=False):
    # Preprocessing: First compute padding and r
    m = len(A)
    b = [0] * m

    if p == 0:
        return b, False

    if p < m:
        tournament_tree_kth(A, b, 1, p)
        return b, compute_carry(A, b, compute_lmax(A, b))

    r = int(math.ceil(math.log(p / m) / math.log(2.0)))
    two_r = 2**r
    n_max = max([len(x) for x in A])
    alpha = math.floor(n_max / two_r)
    n = two_r * (alpha + 1) - 1
    if debug:
        print("p = %d, m = %d, r = %d, two_r = %d, n_max = %d, alpha = %d, n = %d" % (p, m, r, two_r, n_max, alpha, n))

    # Base case: determine f-partition of S(0) (f = p / (m * n))
    k = math.ceil(p / n * alpha)
#    k = int((p / n * alpha) + 0.5)
#    if debug:
#        print("k = ", k)

    tournament_tree_kth(A, b, two_r, k)
#    if debug:
#        print("b = ", b)
    lmax = compute_lmax(A, b)
#    if debug:
#        print("b = ", b, "lmax = ", lmax)

    # The paper doesn't address this case and their description of the base case with variable size lists
    # doesn't seem to work in the case of huge imbalance
    if lmax == MAX_ELEMENT-1:
        print("FATAL lmax == MAX_ELEMENT")
        sys.exit(1)
        b = [0] * m
        tournament_tree_kth(A, b, 1, p, True)
        lmax = compute_lmax(A, b)
        return b, compute_carry(A, b, lmax)

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
#            print("pre-boundary (after moving undecided")
#            print([A[i][x-1] for i, x in enumerate(b)])
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

#        if debug:
#            print("post-boundary")
#            print([A[i][x-1] for i, x in enumerate(b)])

    return b, compute_carry(A, b, lmax)


# A is list of lists to partition
# p is the splitting point (number on left) (fN)
# debug probably doesn't work as it's copy-pasted from non-reverse version
def variable_split_reverse(A, p, tournaments, debug=False):
    # Preprocessing: First compute padding and r
    m = len(A)
    b = []
    for a in A:
        b.append(len(a))

    if debug:
        print("p: %d, m: %d" % (p, m))

    if p == 0:
        return b, False

    if p < m:
        tournament_tree_kth_reverse(A, b, 1, p)
        return b, compute_carry_reverse(A, b, compute_lmax_reverse(A, b))

    r = int(math.ceil(math.log(p / m) / math.log(2.0)))
    two_r = 2**r
    n_max = max([len(x) for x in A])
    alpha = math.floor(n_max / two_r)
    n = two_r * (alpha + 1) - 1
    if debug:
        print("m = %d, r = %d, two_r = %d, n_max = %d, alpha = %d, n = %d" % (m, r, two_r, n_max, alpha, n))

    # Base case: determine f-partition of S(0) (f = p / (m * n))
    k = math.ceil(p / n * alpha)
#    k = int((p / n * alpha) + 0.5)
    if debug:
        print("k = ", k)

    tournament_tree_kth_reverse(A, b, two_r, k)
    lmax = compute_lmax_reverse(A, b)
    if lmax == -(MAX_ELEMENT-1):
        print("FATAL: lmax == MAX_ELEMENT-1")
        sys.exit(1)
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
            Lsize += (len(A[i]) - b[i]) // w_k       # original formulation is power-of-2 so b_k is always even
        if debug:
            print("initial loop Lsize after adding decided: ", Lsize)
        for i in range(m):
            if b[i] - w_k < 0:
                continue
            undecided = -A[i][b[i] - w_k]
            if undecided < lmax:
                b[i] -= w_k
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
            tournament_tree_kth_largest_reverse(A, b, w_k, Lsize - target_size)
            tournaments.append((m, Lsize - target_size))
        else:
            if debug:
                print("move %d smallest elements from H to L" % (target_size - Lsize))
            tournament_tree_kth_reverse(A, b, w_k, target_size - Lsize)
            tournaments.append((m, target_size - Lsize))

        lmax = compute_lmax_reverse(A, b)
        if debug:
            print("new lmax = %d" % lmax, compute_lmax_reverse(A, b))

        if debug:
            print("post-boundary")
            print([A[i][x-1] for i, x in enumerate(b)])

    return b, compute_carry_reverse(A, b, lmax)


def test_split(A, b, p, carry_check, debug=False):
    L = []
    for a in A:
        L += list(a)
    s = sorted(L)
    ref = s[:p]
    if p == len(s):
        next_val = MAX_ELEMENT
    else:
        next_val = s[p]
    L = []
    for i, b_i in enumerate(b):
        L += list(A[i][:b_i])
    test = sorted(L)
    if debug:
        print(carry_check)
        print(test)
        print(ref)
    for i in range(p):
        if ref[i] != test[i]:
            return False, False

    carry_pass = False
    if p == 0:
        return True, carry_check == False

    if ref[-1] == next_val and carry_check:
        carry_pass = True
    if ref[-1] != next_val and not carry_check:
        carry_pass = True

    return True, carry_pass


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
    split_point = 60
    print(sorted(B)[:split_point])
#    variable_split(A, split_point)
    print(len(B))
    b = variable_split_reverse(A, len(B) - 60, [], debug=True)
    print(b)
    res = test_split(A, b, 60)
    print(res)
    sys.exit(1)
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

    MB = sio.mmread(sys.argv[2])
    MB = MB.tocsr()
    MB.sort_indices()

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
    lb_data_file = open("lb_data.bin", "wb")
    lb_block_ptrs_file = open("lb_block_ptrs.bin", "wb")
    out_loc = 0

    lb_block_ptrs_file.write(struct.pack('I', out_loc))
    data = [0, len(M.getrow(0).indices), 0]
    data += [0] * len(M.getrow(0).indices)
    ploc = out_loc
    out_loc += 3 + len(M.getrow(0).indices)
    
    splits = 1
    splits_file = open("splits.txt", "w")
    splits_bin_file = open("splits.bin", "wb")
    bin_loc = 0
    max_cut_size = 0
    last_i = 0
    max_nrows = 0
    for i in range(M.shape[0]):
        row_size = 0
        for j in M.getrow(i).indices:
            total += MB.getrow(j).nnz
            row_size += MB.getrow(j).nnz 
        while next_block <= total:
            '''
            if M.getrow(i).nnz > 32:
                long_rows += 1
            if M.getrow(i).nnz == 1:
                singles += 1
            if row_size > BLOCK_SIZE:
                multi_rows += 1
            '''
            split_pt = next_block - (total - row_size)
            nrows = i - last_i
            print("%d: split at %d, %d / %d, m = %d, bin_loc = %d, rows = %d" % (splits, i, split_pt, row_size, M.getrow(i).nnz, bin_loc, nrows))
            if nrows > max_nrows:
                max_nrows = nrows
            A = []
            for j in M.getrow(i).indices:
                A.append(MB.getrow(j).indices)

            if split_pt / row_size > 0.5:
                fN = row_size - split_pt
                b, carry = variable_split_reverse(A, fN, tournaments, debug=False)
                splits_file.write("%d %d %d 1\n" % (i, fN, bin_loc))
            else:
                d = False
#                if splits == 1381:
#                    d = True
                b, carry = variable_split(A, split_pt, tournaments, debug=d)
                splits_file.write("%d %d %d 0\n" % (i, split_pt, bin_loc))

            splits_bin_file.write(np.array(b, dtype='u4').tobytes())
            bin_loc += len(b)

            lb_block_ptrs_file.write(struct.pack('I', out_loc))
            data += [i, len(M.getrow(i).indices), 0]
            if len(M.getrow(i).indices) > max_cut_size:
                max_cut_size = len(M.getrow(i).indices)
            data += b
            data[ploc + 2] = carry
            ploc = out_loc
            out_loc += 3 + len(b)
            splits += 1
            last_i = i

            # Check correct
            res, cres = test_split(A, b, split_pt, carry, debug=False)
            if not res:
                print("ERROR: bad split at %d, %d" % (i, split_pt))
#                test_split(A, b, split_pt, carry, debug=True)
                sys.exit(1)
            if not cres:
                print("ERROR: carry check failed at %d, %d" % (i, split_pt))
                sys.exit(1)
            next_block += BLOCK_SIZE
        row_sizes.append(total)
    print("shorts: %d, longs: %d, long_rows: %d, singles: %d, multi_rows: %d" % (shorts, longs, long_rows, singles, multi_rows))
    print("tries: %d, tournaments: %d" % (tries, len(tournaments)))
    print("max cut size: %d" % max_cut_size)
    print("max nrows: %d" % max_nrows)
    splits_file.close()
    splits_bin_file.close()
    lb_data_file.write(np.array(data, dtype='u4').tobytes())
    lb_data_file.close()
    lb_block_ptrs_file.close()

    # Vectorized sorted search instead (merge, merge-path)
