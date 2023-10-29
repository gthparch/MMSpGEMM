import math

def tournament_tree_kth(A, b, w_k, k):
    m = len(A)
    pot = int(math.ceil(math.log(m) / math.log(2.0)))
    T = []
    for i in range(pot+1):
        T.append([(9999999999, -1)] * 2**i)

    # Initial push an element from all lists up
    # Fill the bottom row
    for i in range(m):
        T[pot][i] = (A[i][b[i]+w_k-1], i)

    # Now propagate up the tree
    for l in range(pot-1, -1, -1):
        for j in range(2**l):
            # Need to propagate which list won too
            if T[l+1][j*2][0] < T[l+1][j*2+1][0]:
                T[l][j] = T[l+1][j*2]
            else:
                T[l][j] = T[l+1][j*2+1]

    print(T[0][0])
    winner = T[0][0][1]
    max_winner = T[0][0][0]
    b[winner] += w_k
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
        print(T[0][0])
        winner = T[0][0][1]
        if T[0][0][0] > max_winner:
            max_winner = T[0][0][0]
        b[winner] += w_k
        T[pot][winner] = (A[winner][b[winner]+w_k-1], winner)

    return max_winner



# A is list of lists to partition
# p is the splitting point (number on left) (fN)
def variable_split(A, p):
    # Preprocessing: First compute padding and r
    m = len(A)
    N = sum([len(a) for a in A])
    r = int(math.ceil(math.log(p / m) / math.log(2.0)))
    two_r = 2**r
    n_max = max([len(x) for x in A])
    alpha = math.floor(n_max / two_r)
    n = two_r * (alpha + 1) - 1
    print("m = %d, N = %d, r = %d, two_r = %d, n_max = %d, alpha = %d, n = %d" % (m, N, r, two_r, n_max, alpha, n))
#    print(m, N, r, two_r, n_max, alpha, n, p / (m * n))

    print("padding to %d" % n)
    for a in A:
        a += [9999999999] * (n - len(a))
    print(A)

    # Base case: determine f-partition of S(0) (f = p / (m * n))

    # Select set
    S = []
    for a in A:
        i = two_r
        while i < len(a) and a[i-1] < 9999999999:
            S.append(a[i-1])
            i += two_r
    print(S, len(S))
#    print("k = %d" % k)
    print(sorted(S))

    k = math.ceil(p / n * alpha)
    print("k = ", k)

    b = [0, 0, 0, 0, 0]
    lmax = tournament_tree_kth(A, b, two_r, k)
    print("b = ", b, "lmax = ", lmax)

    # r iterative steps
    for k in range(r):
        Lsize = 0
        w_k = 2**(r - k - 1)
#        target_size = math.ceil(p / w_k)
        target_size = math.ceil(p * (n // w_k) / n)
        print("w_k = %d, target_size = %d" % (w_k, target_size))
        # add the decided elements
        for i in range(len(b)):
            Lsize += b[i] // w_k       # original formulation is power-of-2 so b_k is always even
        print("initial loop Lsize after adding decided: ", Lsize)
        for i in range(m):
            undecided = A[i][b[i] + w_k - 1]
            if undecided < lmax:
                b[i] += w_k
                Lsize += 1
        print("initial loop Lsize after adding undecided: ", Lsize, b)
        print("pre-boundary (after moving undecided")
        print([A[i][x-1] for i, x in enumerate(b)])
        print([A[i][x+w_k-1] for i, x in enumerate(b)])
        print("Lsize = %d, target_size = %d" % (Lsize, target_size))

        if Lsize == target_size:
            print("have f-partition")
            continue
        elif Lsize > target_size:
            print("moving %d largest elements from L to H" % (Lsize - target_size))
            # Use tournament tree or something faster
            for i in range(Lsize - target_size):
                max_v = -99999999
                max_j = None
                for j in range(len(A)):
                    # skip empty lists (boundary outside list)
                    if b[j] == 0:
                        continue
                    x = A[j][b[j]-1]
                    if x > max_v:
                        max_v = x
                        max_j = j
                # move element by adjusting boundary
                b[max_j] -= w_k
                print("moving %d from list %d from L to H" % (max_v, max_j), b)
        else:
            print("move %d smallest elements from H to L" % (target_size - Lsize))
            tournament_tree_kth(A, b, w_k, target_size - Lsize)
        Lsize = target_size

        print("post-boundary")
        print([A[i][x-1] for i, x in enumerate(b)])
        print([A[i][x+w_k-1] for i, x in enumerate(b)])



def main():
    # Example from the paper
    A = [[ 1, 2, 6, 7, 9, 11, 15, 99999 ],
         [ 2, 8, 9, 17, 23, 24, 25, 99999 ],
         [ 6, 7, 9, 12, 23, 24, 25, 99999 ],
         [ 3, 8, 10, 13, 14, 17, 19, 99999 ]]
    m = len(A)
    n = len(A[0])-1
    N = m * n
    r = int(math.log2(n + 1))

    k = 0
    w_k = 2**(r - k -1)

    print("N = %d, n = %d, r = %d, k = %d, w_k = %d\n" % (N, n, r, k, w_k))

    f = 0.5
#    print(math.ceil(f * N))

    # Find the first f-partition of S(0)
    S_0 = []
    for i in range(m):
        S_0.append(A[i][w_k-1])
    S_0.sort()
    L_0 = S_0[:int(f * m)]     # Handle when m is not even
    H_0 = S_0[int(f * m):]
    lmax = L_0[-1]

    print(S_0, L_0, H_0, lmax)

    # given boundary, just update?
    b = [4, 0, 4, 0]
    Lsize = 0

    # Iterative process
    for k in range(1, 3):
        w_k = 2**(r - k - 1)
        size_S = (2**(k+1) - 1) * m
        # add the decided elements
        for i in range(len(b)):
            Lsize += b[i] // 2       # original formulation is power-of-2 so b_k is always even
        target_size = math.ceil(f * size_S)
        for i in range(m):
            undecided = A[i][b[i] + w_k - 1]
            if undecided < lmax:
                b[i] += w_k
                Lsize += 1
        print("pre-boundary (after moving undecided")
        print([A[i][x-1] for i, x in enumerate(b)])
        print([A[i][x+w_k-1] for i, x in enumerate(b)])
        print("Lsize = %d, target_size = %d" % (Lsize, target_size))
        if Lsize == target_size:
            print("have f-partition")
            continue
        elif Lsize > target_size:
            print("moving %d largest elements from L to H" % (Lsize - target_size))
            # Use tournament tree or something faster
            for i in range(Lsize - target_size):
                max_v = -99999999
                max_j = None
                for j in range(len(A)):
                    x = A[j][b[j]-1]
                    if x > max_v:
                        max_v = x
                        max_j = j
                # move element by adjusting boundary
                print("moving %d from list %d from L to H" % (max_v, max_j))
                b[max_j] -= w_k
        else:
            print("move %d smallest elements from H to L" % (target_size - Lsize))
            tournament_tree_kth(A, b, w_k, target_size - Lsize)
            '''
            for i in range(target_size - Lsize):
                min_v = 99999999
                min_j = None
                for j in range(len(A)):
                    x = A[j][b[j]+w_k-1]
                    if x < min_v:
                        min_v = x
                        min_j = j
                # move element by adjusting boundary
                print("moving %d from list %d from H to L" % (min_v, min_j))
                b[min_j] += w_k
            '''
        Lsize = target_size

        print("post-boundary")
        print([A[i][x-1] for i, x in enumerate(b)])
        print([A[i][x+w_k-1] for i, x in enumerate(b)])

#    print(size_S, Lsize)
#    print(b)
#    print([A[i][x-1] for i, x in enumerate(b)])
#    print([A[i][x+w_k-1] for i, x in enumerate(b) if i > 0])



if __name__ == '__main__':
    #main()
    
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
    split_point = 35
    print(sorted(B)[:split_point])
    variable_split(A, split_point)
