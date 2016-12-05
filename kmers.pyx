from cython.parallel import parallel, prange
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport exp, log


ctypedef np.uint8_t UINT8_t
ctypedef np.uint32_t UINT32_t
ctypedef np.uint64_t UINT64_t
ctypedef np.float32_t FLOAT32_t
ctypedef np.float64_t FLOAT64_t

cdef inline UINT8_t letter_to_bits(UINT8_t n):
    if n == 'A':
        return 0
    elif n == 'C':
        return 1
    elif n == 'G':
        return 2
    elif n == 'T':
        return 3
    elif n == 'U':
        return 3
    if n == 'a':
        return 0
    elif n == 'c':
        return 1
    elif n == 'g':
        return 2
    elif n == 't':
        return 3
    elif n == 'u':
        return 3
    else:
        #raise ValueError("encountered non ACGT nucleotide '{0}'".format(n))
        return 255
    
@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def seq_to_bits(unsigned char *seq):
    cdef UINT32_t x, L
    cdef UINT8_t n
    
    L = len(seq)
    cdef np.ndarray[UINT8_t] _res = np.zeros(L, dtype=np.uint8)
    cdef UINT8_t [::1] res = _res
    
    for x in range(L):
        n = seq[x]
        res[x] = letter_to_bits(n)

    return _res

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def read_raw_seqs(src, str pre="", str post="", UINT32_t n_max=0, UINT32_t n_skip=0):
    cdef char* l
    cdef UINT32_t i, N=0, L
    cdef list seqs = list()
    cdef np.ndarray[UINT8_t] _buf
    cdef UINT8_t [::1] buf
    
    for line in src:
        N += 1
        if n_skip and N <= n_skip:
            continue

        line = line.rstrip() # remove trailing new-line characters
        if pre or post:
            line = pre + line + post

        L = len(line)
        _buf = np.empty(L, dtype=np.uint8)
        buf = _buf # initialize the view
        
        l = line # extract raw string content
        for i in range(0,L):
            buf[i] = letter_to_bits(l[i])
        
        seqs.append(_buf)
        if N >= n_max + n_skip and n_max:
            break

    return np.array(seqs)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def read_raw_seqs_chunked(src, str pre="", str post="", UINT32_t n_max=0, UINT32_t n_skip=0, chunklines=1000000):
    cdef char* l
    cdef UINT64_t i, N=0, n=0, L=0
    cdef UINT32_t chunkbytes = 0
    cdef list chunks = list()
    cdef bytes line
    cdef bytes _pre = <bytes>pre
    cdef bytes _post = <bytes>post
    cdef np.ndarray[UINT8_t] _buf
    cdef UINT8_t [::1] buf # fast memoryview into current buffer

    for line in src:
        if n_skip and N <= n_skip:
            continue

        if 'N' in line:
            continue

        line = line.rstrip() # remove trailing new-line characters

        if pre or post:
            line = _pre + line + _post
        
        if not L:
            L = len(line)
            chunkbytes = chunklines*L
        
        if N % chunklines == 0:
            if N: chunks.append(_buf)
            _buf = np.empty(L*chunklines, dtype=np.uint8)
            buf = _buf # initialize the view
            n = 0

        l = line # extract raw string content
        for i in range(0,L):
            x = letter_to_bits(l[i])
            buf[n] = x
            n += 1

        N += 1
            
        if N >= n_max + n_skip and n_max:
            break

    chunks.append(_buf[:n])
    cat = np.concatenate(chunks)

    #for c in chunks:
        #print c.shape
        
    #print "concatenation", cat.shape
    #print "want", N,L, N*L
    return cat.reshape((N,L))

from libc.stdio cimport FILE, fopen, fclose, getline
from libc.stdlib cimport malloc, free
 
 
@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def read_raw_seqs_C(filename, UINT32_t n_max=0, UINT32_t n_skip=0):

    cdef UINT32_t i, N=0
    cdef list seq_num, seqs = list()
    cdef np.ndarray[UINT8_t] _buf
    cdef UINT8_t [::1] buf
 
    fname_bytes = filename.encode("UTF-8")
    cdef char* fname = fname_bytes
    cdef FILE* cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise IOError(2, "No such file or directory: '%s'" % filename)
 
    # ask getline to allocate buffer for us
    cdef char* l = NULL
    cdef size_t n = 0
    cdef ssize_t read 
    cdef bytes line
    cdef ssize_t L = 0
    
    while True:
        read = getline(&l, &n, cfile)
        if read == -1: 
            break
 
        N += 1
        if n_skip and N <= n_skip:
            continue

        if not L:
            while L < read:
                if not l[L] in ['A','C','G','T','a','c','g','t']:
                    break
                L += 1
                
        seq_num = []
        #_buf = np.empty(L, dtype=np.uint8)
        #buf = _buf # initialize the view
        
        for i in range(0,L):
            seq_num.append( letter_to_bits(l[i]) )
        
        seqs.append(seq_num)
        if N >= n_max + n_skip and n_max:
            break

    free(l)
    fclose(cfile)

    return np.array(seqs, dtype=np.uint8)

        
@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def read_fastq(src, str pre="", str post="", UINT32_t n_max=0, UINT32_t n_skip=0):
    cdef char* l
    cdef UINT32_t i, N, L, line_num = -1
    cdef list seqs = list()
    cdef np.ndarray[UINT8_t] _buf
    cdef UINT8_t [::1] buf
    
    N = 0
    for line in src:
        line_num += 1
        if line_num % 4 != 1:
            continue
        
        N += 1
        if n_skip and N <= n_skip:
            continue
        
        line = pre + line.rstrip() + post
        L = len(line)
        _buf = np.empty(L, dtype=np.uint8)
        buf = _buf # initialize the view
        
        l = line # extract raw string content
        for i in range(0,L):
            buf[i] = letter_to_bits(l[i])
        
        seqs.append(_buf)
        if N >= n_max + n_skip and n_max:
            break
        
    return np.array(seqs)
            

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
cdef inline UINT32_t kbits_to_index(UINT8_t [:] kbits, UINT32_t k):
    cdef UINT32_t i, index = 0
    
    #assert kbits.dtype == np.uint8
    
    for i in range(k):
        index += kbits[i] << 2 * (k - i - 1)
    
    return index

def seq_to_index(seq):
    k = len(seq)
    bits = seq_to_bits(seq)
    return kbits_to_index(bits, k)

def index_to_seq(index, k):
    nucs = ['a','c','g','t']
    seq = []
    for i in range(k):
        j = index >> ((k-i-1) * 2)
        seq.append(nucs[j & 3])

    return "".join(seq)


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def seq_set_kmer_count(np.ndarray[UINT8_t, ndim=2] seq_matrix, UINT32_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1

    # store k-mer counts here
    _counts = np.zeros(4**k, dtype = np.uint32)
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT32_t [::1] counts = _counts

    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t index, i, j
    
    for j in range(N):
        seq_bits = _seq_matrix[j*L:(j+1)*L]
        # compute index of first k-mer by bit-shifts
        index = kbits_to_index(seq_bits, k) 
        # count first k-mer
        counts[index] += 1
        # iterate over remaining k-mers
        for i in range(0, L-k):
            # get next "letter"
            s = seq_bits[i+k]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            # count
            counts[index] += 1
            
    return _counts


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def seq_set_kmer_count_weighted(np.ndarray[UINT8_t, ndim=2] seq_matrix, np.ndarray[FLOAT64_t] _weights, UINT32_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1

    # store k-mer counts here
    _counts = np.zeros(4**k, dtype = np.float64)
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef FLOAT64_t [::1] counts = _counts
    cdef FLOAT64_t [::1] weights = _weights

    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t index, i, j
    cdef FLOAT64_t w
    
    for j in range(N):
        seq_bits = _seq_matrix[j*L:(j+1)*L]
        # compute index of first k-mer by bit-shifts
        index = kbits_to_index(seq_bits, k) 
        # count first k-mer
        w = weights[j]
        counts[index] += w
        # iterate over remaining k-mers
        for i in range(0, L-k):
            # get next "letter"
            s = seq_bits[i+k]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            # count
            counts[index] += w
            
    return _counts


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def seq_set_SKA(np.ndarray[UINT8_t, ndim=2] seq_matrix, np.ndarray[FLOAT32_t] _weights, np.ndarray[FLOAT32_t] _background, UINT32_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1

    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])

    # store k-mer indices here
    _mer_indices = np.zeros(L-k+1, dtype=np.uint32)
    
    # store current k-mer weights here
    _mer_weights = np.zeros(L-k+1, dtype=np.float32)
    
    
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT32_t [::1] mer_indices = _mer_indices
    cdef FLOAT32_t [::1] mer_weights = _mer_weights
    cdef FLOAT32_t [::1] weights = _weights
    cdef FLOAT32_t [::1] background = _background

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t index, i, j
    cdef FLOAT32_t w, total_w
    
    for j in range(N):
        seq_bits = _seq_matrix[j*L:(j+1)*L]

        # compute index of first k-1 mer by bit-shifts
        index = kbits_to_index(seq_bits, k-1)
        
        total_w = 0
        
        # iterate over k-mers
        for i in range(0, L-k+1):
            # get next "letter"
            s = seq_bits[i+k-1]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            mer_indices[i] = index
            w = weights[index] / background[index]
            mer_weights[i] = w
            total_w += w

        # update weights
        for i in range(0, L-k+1):
            weights[mer_indices[i]] += mer_weights[i]/total_w

    # normalize such that all weights sum up to 4**k
    w = (MAX_INDEX+1)/_weights.sum()
    _weights *= w
    return _weights


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def GC_count(np.ndarray[UINT8_t, ndim=2] seq_matrix):
    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])

    # store k-mer counts here
    _counts = np.zeros(N, dtype = np.uint32)
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT32_t [::1] counts = _counts

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t index, i, j
    
    for j in range(N):
        seq_bits = _seq_matrix[j*L:(j+1)*L]
        for i in range(0, L):
            # get next "letter"
            s = seq_bits[i]
            if s == 1 or s == 2:
                counts[j] += 1

    return _counts


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_energies(np.ndarray[FLOAT64_t] _pos_energies, UINT32_t k):
    # _pos_energies is a (k,4) matrix with per-position energy contributions. 
    # We need to compute the energy values for each k-mer quickly
    _kmer_energies = np.zeros(4**k, dtype=np.float64)
    #print "called cython"
    # views
    cdef FLOAT64_t [::1] kmer_energies = _kmer_energies
    cdef FLOAT64_t [::1] E = _pos_energies
    
    # typed helper variables
    cdef UINT64_t index = 0, l,n, i
    cdef FLOAT64_t e_kmer = 0
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX = 4**k - 1
    
    for index in range(0, MAX_INDEX+1):
        e_kmer = 0
        for l in range(0,k):
            n = (index >> (k - l - 1)*2) & 3
            i = l*4 + n
            e_kmer += E[i]
            #if index == 0b1110110000001100: # tgtaaata
                #print "base at {0} is {1}, e={2}".format(l,n,E[i])
            
        kmer_energies[index] = e_kmer
        
    return _kmer_energies
    

    
#@cython.boundscheck(True)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
#@cython.cdivision(True)
#@cython.overflowcheck(False)
#def kmer_energies_nearest_neighbor(FLOAT64_t [::1] kmer_energies, FLOAT64_t [::1] E_nn, UINT32_t k):
    ## _pos_energies is a (k,4) matrix with per-position energy contributions. 
    ## We need to compute the energy values for each k-mer quickly
    
    ## typed helper variables
    #cdef int index = 0, l,nn, i
    ## largest index in array of DNA/RNA k-mer counts
    #cdef UINT64_t MAX_INDEX = 4**k - 1

    #with nogil, parallel(num_threads=8):
        #for index in prange(0, MAX_INDEX+1):
            #kmer_energies[index] = 0
            #for l in range(0,k-1):
                #nn = (index >> (k - l - 2)*2) & 15
                #i = l*16 + nn
                #kmer_energies[index] += E_nn[i]


#@cython.boundscheck(True)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
#@cython.cdivision(True)
#@cython.overflowcheck(False)
#def k_eff_seq_openen(FLOAT64_t [::1] kmer_energy, FLOAT64_t [::1] open_energy, UINT8_t [::1] discard, UINT8_t [::1] seqs, UINT64_t k, UINT64_t N, UINT64_t L):
    #cdef UINT64_t L_openen = L - k

    ## largest index in array of DNA/RNA k-mer counts
    #cdef UINT64_t MAX_INDEX = 4**k - 1

    ## store effective dissociation constants here
    #_k_eff = np.zeros(N, dtype = np.float64)
    ## make fast, unstrided memoryviews for the results
    #cdef FLOAT64_t [::1] k_eff = _k_eff
    
    ## helper variables to tell cython the types
    #cdef UINT8_t s
    #cdef UINT64_t index
    #cdef int i, j
    #cdef FLOAT64_t dG, dF
    #cdef FLOAT64_t boltzmann_sum
    
    #with nogil, parallel(num_threads=8):
        #for j in prange(N, schedule='static'):
            ## compute index of first k-mer by bit-shifts
            #index = 0
            #for i in range(k):
                #index += seqs[j*L + i] << 2 * (k - i - 1)
            
            #if not discard[index]:
                #dG = kmer_energy[index]
                #dF = open_energy[j*L_openen]
                #k_eff[j] += exp(-(dG + dF))
            
            ## iterate over remaining k-mers
            #for i in range(0, L-k):
                ##if j == 5634 or j == 5635:
                    ##print i, kmer_energy[index], exp(-kmer_energy[index]), k_eff[j], 1e6/k_eff[j]
                ## get next "letter"
                #s = seqs[j*L + i + k]
                ## compute next index from previous by shift + next letter
                #index = ((index << 2) | s ) & MAX_INDEX
                
                #if not discard[index]:
                    #dG = kmer_energy[index]
                    #dF = open_energy[(j*L_openen) + i]
                    #k_eff[j] += exp(-(dG + dF))

        #for j in prange(N, schedule='static'):
            ## convert to dissociation constant in nM
            #k_eff[j] = 1e6/k_eff[j]
        
    #return _k_eff



@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def k_eff_seq_openen_record(FLOAT64_t [::1] kmer_energy, FLOAT64_t [::1] open_energy, UINT8_t [::1] discard, UINT8_t [::1] seqs, UINT64_t k, UINT64_t N, UINT64_t L, UINT64_t struct_pad):
    cdef UINT64_t L_openen = L - (k + 2*struct_pad) + 1

    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX = 4**k - 1

    # store effective dissociation constants here
    _k_eff = np.zeros(N, dtype = np.float64)
    
    # store k_eff gradient matrix elements in here
    cdef int l_grad = k*4
    _k_grad = np.zeros(N*l_grad, dtype = np.float64)
    
    # keep index of best hit index in sequence
    _best_i = np.zeros(N, dtype = np.uint64)
    _best_index = np.zeros(N, dtype = np.uint32)
    _best_dG = np.zeros(N, dtype = np.float64)
    _best_dF = np.zeros(N, dtype = np.float64)
    _best_dE = np.zeros(N, dtype = np.float64) + 100000

    # make fast, unstrided memoryviews for the results
    cdef FLOAT64_t [::1] k_eff = _k_eff
    cdef FLOAT64_t [::1] k_grad = _k_grad
    cdef UINT64_t [::1] best_i = _best_i
    cdef UINT32_t [::1] best_index = _best_index
    cdef FLOAT64_t [::1] best_dG = _best_dG
    cdef FLOAT64_t [::1] best_dF = _best_dF
    cdef FLOAT64_t [::1] best_dE = _best_dE
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, grad_index
    cdef int i, j, l, m
    cdef FLOAT64_t dG, dF, dE, w
    
    #with nogil, parallel(num_threads=8):
        #for j in prange(N, schedule='static'):

    for j in range(N):
        #print j
        # compute index of first k-mer by bit-shifts
        index = 0
        for i in range(k):
            index += seqs[j*L + i] << 2 * (k - i - 1)
        
        i = 0
        if not discard[index]:
            dG = kmer_energy[index]
            dF = open_energy[j*L_openen]
            dE = dG + dF
            #print j, i, index_to_seq(index, k), dG
            w = exp(-(dE))
            k_eff[j] += w
            
            if dE < best_dE[j]:
                best_dE[j] = dE
                best_dG[j] = dG
                best_dF[j] = dF
                best_i[j] = i
                best_index[j] = index
        
            # update gradient matrix elements
            grad_index = index
            for l in range(k-1,-1,-1):
                m = grad_index & 3
                k_grad[j*l_grad + l*4 + m] += w
                grad_index >>= 2
                
                #if j == 0:
                    #print i,l,m,w,dE
                    #print "curr k_grad", _k_grad[j*l_grad:j*l_grad+l_grad]
        
        # iterate over remaining k-mers
        for i in range(0, L-k):
            #if j == 5634 or j == 5635:
                #print i, kmer_energy[index], exp(-kmer_energy[index]), k_eff[j], 1e6/k_eff[j]
            # get next "letter"
            s = seqs[j*L + i + k]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            
            if not discard[index]:
                dG = kmer_energy[index]
                l = min(0, i+1 - struct_pad)
                dF = open_energy[(j*L_openen) + l]
                dE = dG + dF
                
                #print j, i+1, index_to_seq(index, k), dG
                w = exp(-(dE))
                k_eff[j] += w

                if dE < best_dE[j]:
                    best_dE[j] = dE
                    best_dG[j] = dG
                    best_dF[j] = dF
                    best_i[j] = i + 1
                    best_index[j] = index

                # update gradient matrix elements
                grad_index = index
                for l in range(k-1,-1,-1):
                    m = grad_index & 3
                    #print j,l,m,j*l_grad + l*4 + m, N*l_grad
                    k_grad[j*l_grad + l*4 + m] += w
                    grad_index >>= 2

                    #if j == 0:
                        #print i+1,l,m,w,dE
                        #print "curr k_grad", _k_grad[j*l_grad:j*l_grad+l_grad]

            #print i+1, dE, exp(-dE), k_eff[j]

    #for j in prange(N, schedule='static'):
    for j in range(N):
        # convert to dissociation constant in nM
        k_eff[j] = 1e9/k_eff[j] # Z does not include 1 bc we condition on the oligo being bound!
        
        # d k_eff/ d G_lm = k_eff^2 * [sum_i=1^L \delta(s_i+l,m) exp^-dE ]
        _k_grad[j*l_grad:(j+1)*l_grad] *= k_eff[j] * k_eff[j] /1e9
        #if j == 0:
            #print "multiplied by k_eff^2", k_eff[j] * k_eff[j] /1e9
            #print _k_grad[j*l_grad:(j+1)*l_grad]
        
    return _k_eff, _k_grad.reshape((N,l_grad),order='C'), _best_i, _best_dE, _best_dF, _best_dG, _best_index




#@cython.boundscheck(True)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
#@cython.cdivision(True)
#@cython.overflowcheck(False)
#def compute_update_matrix(
    #FLOAT64_t [::1] R_obs, 
    #FLOAT64_t [::1] R_pred, 
    #FLOAT64_t [::1] kmer_energy, 
    #FLOAT64_t [::1] open_energy, 
    #UINT8_t [::1] discard, 
    #UINT8_t [::1] seqs, 
    #UINT64_t k, 
    #UINT64_t N, 
    #UINT64_t L
    #):
    
    ## largest index in array of DNA/RNA k-mer counts
    #cdef UINT64_t MAX_INDEX = 4**k - 1
    
    ## number of windows that can be placed on each sequence
    #cdef UINT64_t L_openen = L - k + 1

    ## accumulate the update matrix in here
    #_U = np.zeros(k*4, dtype=np.float64)
    
    ## accumulate update matrix weights here
    ## cognate
    #_c  = np.zeros(k*4, dtype=np.float64)
    ## non-cognate
    #_nc = np.zeros(k*4, dtype=np.float64)
    
    ## make fast, unstrided memoryviews for the results
    #cdef FLOAT64_t [::1] U = _U
    #cdef FLOAT64_t [::1] c = _c
    #cdef FLOAT64_t [::1] nc = _nc

    ## helper variables to tell cython the types
    #cdef UINT8_t s
    #cdef UINT64_t index
    #cdef int i, j, l, b
    #cdef FLOAT64_t dG, dF, dE, dR, wdG = 0
    #cdef FLOAT64_t w =0, v=0

    ## accumulate Boltzmann weights here 
    #cdef FLOAT64_t Z=1, Z_total=0, R_obs_total =0
    
    ##with nogil, parallel(num_threads=8):
        ##for j in prange(N, schedule='static'):

    #for j in range(N):
        ##print j
        ## compute index of first k-mer by bit-shifts
        #index = 0
        #for i in range(k):
            #index += seqs[j*L + i] << 2 * (k - i - 1)
        
        #i = 0
        #if not discard[index]:
            #dG = kmer_energy[index]
            #dF = open_energy[j*L_openen]
            #dE = dG + dF
            ##print j, i, index_to_seq(index, k), dG
            #w = exp(-(dE))
            #v = w/3.
            #Z += w
            
            ## add w to (non-)cognate base update-matrix elements
            #for l in range(0,k):
                #for b in range(0,4):
                    #s = seqs[j*L+i+l]
                    #if s == b:
                        #c[(l << 2) + b] += w
                    #else:
                        #nc[(l << 2) + b] += w
            
        ##print i,w
        ## iterate over remaining k-mers
        #for i in range(0, L-k):
            ##if j == 5634 or j == 5635:
                ##print i, kmer_energy[index], exp(-kmer_energy[index]), k_eff[j], 1e6/k_eff[j]
            ## get next "letter"
            #s = seqs[j*L + i + k]
            ## compute next index from previous by shift + next letter
            #index = ((index << 2) | s ) & MAX_INDEX
            
            #if not discard[index]:
                #dG = kmer_energy[index]
                #dF = open_energy[(j*L_openen) + i + 1]
                #dE = dG + dF
                
                #w = exp(-(dE))
                #v = w/3.
                ##print i+1, w, dE, dG, dF, index
                #Z += w
                
                ## add w to cognate base update-matrix elements
                #for l in range(0,k):
                    #s = seqs[j*L+i+l+1]
                    #for b in range(0,4):
                        #if s == b:
                            #c[(l << 2) + b] += w
                        #else:
                            #nc[(l << 2) + b] += w

        #Z_total += Z
        #R_obs_total += R_obs[j]
        
        #dR = log(R_pred[j]/R_obs[j])
        
        ##print dR
        ##print "mask", _u.reshape((6,4))
        ##if dR < 0:
            ##_u = _u *dR - (1-_u)*dR
        ##else:
            ##_u.fill(dR/4./k) #(1 - _u) * dR #- _u * dR

        ##print _u.reshape((6,4))

        #if j == 0:
            #print j, dR, Z
            #print ((_c - _nc) * dR/k).reshape((k,4))
        
        #_U += (_c - _nc) * dR/k * R_obs[j]
        #Z = 1
        ## reset
        #_c.fill(0.)
        #_nc.fill(0.)
        #c = _c
        #nc = _nc

    #return 100.*(_U/Z_total/R_obs_total).reshape((k,4))
    ##return (_U/Z_total).reshape((k,4))




#@cython.boundscheck(True)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
#@cython.cdivision(True)
#@cython.overflowcheck(False)
#def update_matrix_from_oligo(
    #FLOAT64_t R_obs, 
    #FLOAT64_t R_pred, 
    #FLOAT64_t [::1] kmer_energy, 
    #FLOAT64_t [::1] open_energy, 
    #UINT8_t [::1] seqs, 
    #UINT64_t k, 
    #UINT64_t L,
    #FLOAT64_t E_best,
    #debug=False
    #):
    
    ## largest index in array of DNA/RNA k-mer counts
    #cdef UINT64_t MAX_INDEX = 4**k - 1
    
    ## accumulate update matrix weights here
    #_u  = np.zeros(k*4, dtype=np.float64)
    ## cognate
    #_c  = np.zeros(k*4, dtype=np.float64)
    ## non-cognate
    #_nc = np.zeros(k*4, dtype=np.float64)
    
    ## make fast, unstrided memoryviews for the results
    #cdef FLOAT64_t [::1] c = _c
    #cdef FLOAT64_t [::1] nc = _nc

    ## helper variables to tell cython the types
    #cdef UINT8_t s
    #cdef UINT64_t index
    #cdef int i, j, l, b
    #cdef FLOAT64_t dG, dF, dE, dR, wdG # weighted dG
    #cdef FLOAT64_t w =0, w_best =0, a

    ## accumulate Boltzmann weights here 
    #cdef FLOAT64_t Z=0, Z_total=0, R_obs_total =0
    
    #index = 0
    #for l in range(k):
        #index += seqs[l] << 2 * (k - l - 1)

    #i = 0
    #dG = kmer_energy[index]
    #dF = open_energy[i]
    #dE = dG + dF
    #w_best = exp(E_best)
    ##print j, i, index_to_seq(index, k), dG
    #w = exp(-(dE))

    #wdG = dG*w

    #Z += w
    
    ## weight to distrubute between cognate and non-cognate
    #a = w_best * w
    
    ## add w to (non-)cognate base update-matrix elements
    #for l in range(0,k):
        #for b in range(0,4):
            #s = seqs[i+l]
            #if s == b:
                #c[(l << 2) + b] += w
            #else:
                #nc[(l << 2) + b] += w
        
    ## iterate over remaining k-mers
    #for i in range(0, L-k):
        ##if j == 5634 or j == 5635:
            ##print i, kmer_energy[index], exp(-kmer_energy[index]), k_eff[j], 1e6/k_eff[j]
        ## get next "letter"
        #s = seqs[j*L + i + k]
        ## compute next index from previous by shift + next letter
        #index = ((index << 2) | s ) & MAX_INDEX
        
        #dG = kmer_energy[index]
        #dF = open_energy[i + 1]
        #dE = dG + dF
        
        #w = exp(-(dE))
        ##print i+1, w, dE, dG, dF, index
        #wdG += w*dG
        #Z += w
        #a = w_best * w
        ## add w to cognate base update-matrix elements
        #for l in range(0,k):
            #s = seqs[j*L+i+l+1]
            #for b in range(0,4):
                #if s == b:
                    #c[(l << 2) + b] += w
                #else:
                    #nc[(l << 2) + b] += w

        #if debug:
            #print i+1,w,a,dE
    #dR = log(R_pred/R_obs)
    
    #return _c.reshape((k,4)) / Z, _nc.reshape((k,4)) / Z, wdG/Z



#def debug_score_seq(FLOAT64_t [::1] kmer_energy, FLOAT64_t [::1] open_energy, UINT8_t [::1] discard, UINT8_t [::1] seqs, UINT64_t k, UINT64_t N, UINT64_t L, UINT64_t j):
    #cdef UINT64_t L_openen = L - k + 1

    ## largest index in array of DNA/RNA k-mer counts
    #cdef UINT64_t MAX_INDEX = 4**k - 1

    ## store effective dissociation constants here
    #_k_eff = np.zeros(N, dtype = np.float64)
    
    ## keep index of best hit index in sequence
    #_best_i = np.zeros(N, dtype = np.uint64)
    #_best_index = np.zeros(N, dtype = np.uint32)
    #_best_dG = np.zeros(N, dtype = np.float64)
    #_best_dF = np.zeros(N, dtype = np.float64)
    #_best_dE = np.zeros(N, dtype = np.float64) + 100000

    ## make fast, unstrided memoryviews for the results
    #cdef FLOAT64_t [::1] k_eff = _k_eff
    #cdef UINT64_t [::1] best_i = _best_i
    #cdef UINT32_t [::1] best_index = _best_index
    #cdef FLOAT64_t [::1] best_dG = _best_dG
    #cdef FLOAT64_t [::1] best_dF = _best_dF
    #cdef FLOAT64_t [::1] best_dE = _best_dE
    
    ## helper variables to tell cython the types
    #cdef UINT8_t s
    #cdef UINT64_t index
    #cdef int i
    #cdef FLOAT64_t dG, dF, dE
    #cdef FLOAT64_t boltzmann_sum
    
    ##with nogil, parallel(num_threads=8):
        ##for j in prange(N, schedule='static'):

        ##print j
    ## compute index of first k-mer by bit-shifts
    #index = 0
    #for i in range(k):
        #index += seqs[j*L + i] << 2 * (k - i - 1)
    
    #i = 0
    #print j,i, index_to_seq(index, k), discard[index], kmer_energy[index]
    #if not discard[index]:
        #dG = kmer_energy[index]
        #dF = open_energy[j*L_openen]
        #dE = dG + dF
        #print dG, dF, dE
        #k_eff[j] += exp(-(dE))
        
        #if dE < best_dE[j]:
            #best_dE[j] = dE
            #best_dG[j] = dG
            #best_dF[j] = dF
            #best_i[j] = i
            #best_index[j] = index
    
    ## iterate over remaining k-mers
    #for i in range(0, L-k):
        ##if j == 5634 or j == 5635:
            ##print i, kmer_energy[index], exp(-kmer_energy[index]), k_eff[j], 1e6/k_eff[j]
        ## get next "letter"
        #s = seqs[j*L + i + k]
        ## compute next index from previous by shift + next letter
        #index = ((index << 2) | s ) & MAX_INDEX

        #print j,i+1, index_to_seq(index, k), discard[index], kmer_energy[index]
        #if not discard[index]:
            #dG = kmer_energy[index]
            #dF = open_energy[(j*L_openen) + i + 1]
            #dE = dG + dF
            
            #print dG, dF, dE
            #k_eff[j] += exp(-(dE))

            #if dE < best_dE[j]:
                #best_dE[j] = dE
                #best_dG[j] = dG
                #best_dF[j] = dF
                #best_i[j] = i + 1
                #best_index[j] = index
                
                #print "best_hit so far"

    #print "best", index_to_seq(best_index[j], k), best_dG[j]



#from cpython.array cimport array
#from libc.stdlib cimport malloc, free

#@cython.boundscheck(True)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
#@cython.cdivision(True)
#@cython.overflowcheck(False)
#def predict_weighted_counts_from_seq_and_openen(
        #FLOAT64_t [::1] kmer_energy, 
        #FLOAT64_t [::1] open_energy, 
        #FLOAT64_t dG_threshold, 
        #UINT8_t [::1] seqs, 
        #UINT64_t k, UINT64_t N, UINT64_t L, FLOAT64_t P, FLOAT64_t B,  
        #FLOAT64_t [::1] k_eff,
        #FLOAT64_t [::1] counts,
    #):
   
   
    #### constants, really. read-only -> thread-shared
    
    ## row-length in unfolding energy array
    #cdef UINT64_t L_openen = L - k
    ## largest index in array of DNA/RNA k-mer counts
    #cdef UINT64_t MAX_INDEX = 4**k - 1
   
   
    #### Helper variables to tell cython the types.
    #### Everything here should be thread-local!
    
    ## the RBNS sequence at hand ([A,C,G,T]=[0,1,2,3])
    #cdef UINT8_t s 
    ## kmer index: 0 <= index < 4^k
    #cdef UINT64_t index 
    ## j = index of sequence s in all sequences, i=kmer position
    #cdef int i, j 
    ## sequence binding and unfolding energies
    #cdef FLOAT64_t dG, dF 
    ## running sum of Boltzmann weights to accumulate 
    ## bound states along each sequence
    #cdef FLOAT64_t b_sum 
    
    #cdef FLOAT64_t Kd, Theta
    
    ## here we keep the list of kmer-indices per sequence, so we can quickly
    ## update the weighted counts after k_eff is determined
    #cdef UINT64_t *update_indices # thread-local allocation and release. 
    ## even faster would be one-time allocation and index-trick with threadid()?
    
    #with nogil, parallel(num_threads=8):
        ## zero out the counts
        #for i in prange(MAX_INDEX+1, schedule='static'):
            #counts[i] = 0
        
        #for j in prange(N, schedule='static'):
            #update_indices = <UINT64_t*> malloc(sizeof(UINT64_t) * L_openen)
            ## need to abuse this for b_sum due to cython parallelization constraints
            #k_eff[j] = 0

            ## compute index of first k-mer by bit-shifts
            #index = 0
            #for i in range(k):
                #index += seqs[j*L + i] << 2 * (k - i - 1)

            ## record for count updates
            #update_indices[0] = index
            
            #dG = kmer_energy[index]
            ## ignore very weak binding above dG_threshold to save some time
            #if dG < dG_threshold:
                #dF = open_energy[j*L_openen]
                #k_eff[j] += exp(-(dG + dF)) # should be b_sum, but need to trick reduction variable limitation :(
            
            ## iterate over remaining k-mers
            #for i in range(0, L-k):
                ## get next "letter"
                #s = seqs[j*L + i + k]
                ## compute kmer index from previous by shift + next letter
                #index = ((index << 2) | s ) & MAX_INDEX
                ## record for count updates
                #update_indices[i+1] = index

                #dG = kmer_energy[index]
                ## ignore very weak binding above dG_threshold to save some time
                #if dG < dG_threshold:
                    #dF = open_energy[(j*L_openen) + i]
                    #k_eff[j] += exp(-(dG + dF))

            ## compute K_eff in nM
            #Kd = 1e6/k_eff[j]
            ## Michaelis-Menten type binding
            #Theta = P / (Kd + P)
            ## remember k_eff in case we need it elsewhere
            #k_eff[j] = Kd
            
            ## update counts for all kmers encountered in this sequence
            #for i in range(L_openen):
                #counts[update_indices[i]] += Theta

            #free(update_indices)
            

    # non-parallel version of seq-loop with debug output (if needed)
    ## zero out the counts
    #for i in range(MAX_INDEX+1):
        #counts[i] = 0
    
    #for j in range(N):
        #update_indices = <UINT64_t*> malloc(sizeof(UINT64_t) * L_openen)
        ## need to abuse this for b_sum due to cython parallelization constraints
        #k_eff[j] = 0

        ## compute index of first k-mer by bit-shifts
        #index = 0
        #for i in range(k):
            #index += seqs[j*L + i] << 2 * (k - i - 1)

        ## record for count updates
        #update_indices[0] = index
        
        #dG = kmer_energy[index]
        ## ignore very weak binding above dG_threshold to save some time
        #if dG < dG_threshold:
            #dF = open_energy[j*L_openen]
            #k_eff[j] += exp(-(dG + dF)) # should be b_sum, but need to trick reduction variable limitation :(
        
        ## iterate over remaining k-mers
        #for i in range(0, L-k):
            ##if j == 5634 or j == 5635:
                ##print i, kmer_energy[index], exp(-kmer_energy[index]), k_eff[j], 1e6/k_eff[j]
            
            ## get next "letter"
            #s = seqs[j*L + i + k]
            ## compute kmer index from previous by shift + next letter
            #index = ((index << 2) | s ) & MAX_INDEX
            ## record for count updates
            #update_indices[i+1] = index

            #dG = kmer_energy[index]
            ## ignore very weak binding above dG_threshold to save some time
            #if dG < dG_threshold:
                #dF = open_energy[(j*L_openen) + i]
                #k_eff[j] += exp(-(dG + dF))

        ## compute K_eff in nM
        #Kd = 1e6/k_eff[j]
        ## Michaelis-Menten type binding
        #Theta = P / (Kd + P)
        ## remember k_eff in case we need it elsewhere
        #k_eff[j] = Kd
        
        ## update counts for all kmers encountered in this sequence
        #for i in range(L_openen):
            #counts[update_indices[i]] += Theta

        #free(update_indices)
            
            
    
