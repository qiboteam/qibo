import numpy as np
import numba as nb
import tensorflow as tf
from statistics import mean
import random
import math
import time

# mininum threshold for the eigenvalues
EIG_EPS = 1E-11

class timer:
    """ A trivial timer class. """
    def __init__(self):
        pass

    def start(self):
        self.t0 = time.time()
        self.t1 = time.process_time()

    def print(self, message):
        d0 = time.time() - self.t0
        d1 = time.process_time() - self.t1
        print("%fs (wall) | %fs (cpu) - %s" % (d0, d1, message))

@nb.njit(parallel=True)
def random_state(qubits, M):
    """Generate a random state.
    Parameters
    ----------
        qubits (int): the number of qubits.
        M (int): the number of integer states
    Returns
    -------
        The random state.
    """
    zeros = np.zeros(shape=(2**qubits-M), dtype=np.ubyte)
    ones = np.ones(shape=(M), dtype=np.ubyte)
    state = np.concatenate((zeros, ones))
    np.random.shuffle(state)
    return state

def Norm(state):
    """Normalization of the state.
    Parameters
    ----------
        state: the input state
    Returns
    -------
        The normalized state.
    """
    n=np.linalg.norm(state)
    if n==0: 
       return state
    return state/n

def QFT(state):
    """Quantum fourier transform.
    Parameters
    ----------
        state: the input state
    Returns
    -------
        The normalized QFT.
    """
    return Norm(np.fft.fft(state))

@nb.njit(parallel=True)
def build_state(state,qubits,M):
    """ Computes random state matrix from the vector.

    This algoritm works only for even n.

    Parameters
    ----------
        qubits: the number of qbits for the system
        M: the number of integers
        state: the random state
    Returns
    -------
        The random state matrix
    """
    limit = 2**qubits
    norm = np.int64(2**(qubits/2))
    pp = np.zeros(shape=(norm, norm), dtype=np.int32)
    for i in range(limit):
        if state[i] == 1:
            b = np.int64(i % norm)
            a = np.int64((i - b) / norm)
            pp[a, b] = 1
    return pp, M

@tf.function
def make_density_matrix(state, M):
    """ Computes the density matrix (rho) for a given random state.

    Parameters
    ----------
        state: the random state matrix
        M : the number of integers

    Returns
    -------
        The normalized density matrix of the random state
    """
    return tf.linalg.matmul(tf.transpose(state), state)/M

@nb.njit(parallel=True)
def build_state2(state,qubits):
    """ Computes random state matrix from the vector.

    This algoritm works only for even n.
    It works all for normailized state. Not just random state
    Parameters
    ----------
        qubits: the number of qbits for the system
        state: the random state
    Returns
    -------
        The random state matrix
    """
    limit = 2**qubits
    norm = int(2**(qubits/2))
    pp = np.zeros((norm, norm),dtype=np.complex128)
    for i in range(limit):
        b = np.int64(i % norm)
        a = np.int64((i - b) / norm)
        pp[a, b] = state[i]
    return pp

@tf.function
def make_density_matrix2(state):
    """ Computes the density matrix (rho).
    To use with build_state2.
    It works all for normailized state.
    Parameters
    ----------
        state: the random state matrix
        M : the number of integers

    Returns
    -------
        The normalized density matrix of the all state.
    """
    return tf.linalg.matmul(tf.transpose(state,conjugate=True), state)

@tf.function
def get_eigenvalues(rho, eps=EIG_EPS):
    """ Computes the eigenvalues of the density matrix
    and applies a threshold filtering condition.

    Parameters
    ----------
        `rho`: the normalized density matrix
        `eps`: minimum threshold to accept the eigenvalue

    Returns
    -------
        The filtered eigenvalue array

    """
    # compute eigenvalues
    w = tf.linalg.eigvalsh(rho)
    w=tf.math.real(w)
    # apply eigenvalues filtering
    wf = tf.gather(w, tf.where(w > eps))
    
    return wf

@tf.function
def compute_entropy(w):
    """ Computes the entropy for the eigenvalues of the
    density matrix.

    Parameters
    ----------
        `w`: the eigenvalues of the density matrix

    Returns
    -------
        The entropy of the state.
    """
    return -tf.reduce_sum(w*tf.math.log(w)) / np.log(2)

def entropy(state,qubits,M):
    """ Computes the random state entropy.

    Parameters
    ----------
        qubits: the number of qbits for the system
        M: the number of integers
        state: the random state
    Return
    ------
        The entropy
    """
    #t = timer()
    #t.start()
    pp, size = build_state(state,qubits,M)
    #t.print('randum state generation [1/4]')
    
    #t.start()
    rho = make_density_matrix(pp, size)
    #t.print('density construction [2/4]')

    #t.start()
    w = get_eigenvalues(rho, 0)
    #t.print('eigenvalues [3/4]')

    #t.start()
    entropy = compute_entropy(w)
    #t.print('entropy [4/4]')
    return entropy.numpy()

def entropy2(state,qubits):
    """ Computes the random state entropy.

    Parameters
    ----------
        qubits: the number of qbits for the system
        M: the number of integers
        state: the random state
    Return
    ------
        The entropy
    """
    
    pp = build_state2(state,qubits)
    
    rho = make_density_matrix2(pp)
   
    w = get_eigenvalues(rho, 0)
    
    entropy = compute_entropy(w)
 
    return entropy.numpy()

def scale(qubits,a,b,c,s):
    """ Scales the random state entropy along with M in the position state.

    Parameters
    ----------
        qubits for number of qubits
        M for number of integer
        (a,b) the intervalle we want to scale
        c the number of samples of M we want to take
        s for number of samples for each M
    Return
    ------
        Two lists: M and the mean entropy
    """
    A=[x for x in range(a,b,int((b-a)/c))]
    L=[]
    for M in A:
        Ent=[]
        for i in range(s):
            Ent.append(entropy(random_state(qubits, M),qubits,M))
        L.append(mean(Ent))
    return A,L

def scalelog(qubits,c,s):
    """ Scales the random state entropy along with M in the position state.
        Globally (from 1-2**qubits) and in log scale
    Parameters
    ----------
        qubits for number of qubits
        M for number of integer
        c the number of samples of M we want to take
        s for number of samples for each M
    Return
    ------
        Two lists: M and the mean entropy
    """
    A=[]
    for i in range(c):
        A.append(int(2**(qubits*(i+1)/c)))
    L=[]
    for M in A:
        Ent=[]
        for i in range(s):
            Ent.append(entropy(random_state(qubits, M),qubits,M))
        L.append(mean(Ent))
    return A,L

def scaleQFT(qubits,a,b,c,s):
    """ Scales the random state entropy along with M in the momentum space.

    Parameters
    ----------
        qubits for number of qubits
        M for number of integer
        (a,b) the intervalle we want to scale
        c the number of samples of M we want to take
        s for number of samples for each M
    Return
    ------
        Two lists: M and the mean entropy
    """
    A=[x for x in range(a,b,int((b-a)/c))]
    L=[]
    for M in A:
        Ent=[]
        for i in range(s):
            state=random_state(qubits, M)
            state=QFT(state)
            Ent.append(entropy2(state,qubits))
        L.append(mean(Ent))
    return A,L

def scaleTOTAL(qubits,a,b,c,s):
    """ Scales the random state entropy along with M.
    The average of entropy in momentum space and position space.
    Parameters
    ----------
        qubits for number of qubits
        M for number of integer
        (a,b) the intervalle we want to scale
        c the number of samples of M we want to take
        s for number of samples for each M
    Return
    ------
        Two lists: M and the mean entropy
    """
    A=[x for x in range(a,b,int((b-a)/c))]
    L=[]
    for M in A:
        Ent=[]
        for i in range(s):
            state=random_state(qubits, M)
            stateQFT=QFT(state)
            Ent.append((entropy(state,qubits,M)+entropy2(stateQFT,qubits))/2)
            #Ent.append((entropy(state,qubits,M)-entropy2(stateQFT,qubits)))
        L.append(mean(Ent))
    return A,L

def partition(state,qubits):
    """Change a given state for a different partition.
    Parameters
    ----------
        qubits : the number of qubits.
        state : the targeted state
    Returns
    -------
        The state in a different partition.
    """
    N=2**qubits
    mask=[0]*int(qubits/2)+[1]*int(qubits/2)
    random.shuffle(mask)
    result=np.zeros(N)
    ones=np.nonzero(state)[0]
    for i in ones:
        new=[]
        old=int_to_bin(i,qubits)
        for j in range(qubits):
           if mask[j]==1: new.append(old[j])
        for j in range(qubits):
           if mask[j]==0: new.append(old[j])
        newindex=bin_to_int(new)
        result[newindex]=1
    return result

def int_to_bin(integer,qubits):
    """
    changes the binary representation of integer to a list
    """
    integer=str(bin(integer))[2:]
    result=[]
    for i in range(len(integer)):
        result.append(integer[i])
    result=[0]*(qubits-len(integer))+result
    return result
    
def bin_to_int(binary):
    """
    changes the binary list back to integer
    """
    return int(''.join(map(str, binary)), 2)

def distribution(state,qubits,M,n):
    """The distribution of entropy for the same state with different partitions.
    Parameters
    ----------
        k for number of qubits
        M for number of integer
        state for unnormalized random state
        n the number of partitions to take draw the distribution
    Returns
    -------
        A list of entropy of different partition.
    """
    data=[]
    for i in range(n):
        #t = timer()
        t.start()
        statebis=partition(state,qubits)
        #t.print('partition')
        data.append(entropy(statebis,qubits,M))
    return data
