"""QUBO solver using simulated annealing.

Provides a SOLVEQUBO function that can be backed by:
1. dwave-neal/dimod SimulatedAnnealingSampler (preferred)
2. Hand-rolled Metropolis SA as fallback

Designed as an isolated module so a real QPU sampler can be
substituted later with no changes to other modules.
"""
import numpy as np
from typing import Optional
import random
import math

try:
    import dimod
    import neal
    HAS_DWAVE = True
except ImportError:
    HAS_DWAVE = False

def solve_qubo(Q: np.ndarray, num_reads: int = 200, seed: Optional[int] = None) -> np.ndarray:
    """Solve a QUBO problem: minimize z^T Q z over binary z.
    
    Args:
        Q: n x n symmetric QUBO matrix
        num_reads: number of independent annealing runs
        seed: random seed for reproducibility
    
    Returns:
        Best binary vector z (numpy array of 0/1 integers)
    """
    if HAS_DWAVE:
        return _solve_qubo_dwave(Q, num_reads, seed)
    else:
        return _solve_qubo_sa(Q, num_reads, seed)

def _solve_qubo_dwave(Q: np.ndarray, num_reads: int, seed: Optional[int]) -> np.ndarray:
    n = len(Q)
    # create upper triangular Q dict for dimod
    Q_dict = {}
    for i in range(n):
        Q_dict[(i, i)] = Q[i, i]
        for j in range(i + 1, n):
            Q_dict[(i, j)] = Q[i, j] + Q[j, i]
            
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = neal.SimulatedAnnealingSampler()
    
    kwargs = {'num_reads': num_reads}
    if seed is not None:
        kwargs['seed'] = seed
        
    sampleset = sampler.sample(bqm, **kwargs)
    best_sample = sampleset.first.sample
    
    z = np.zeros(n, dtype=int)
    for i in range(n):
        z[i] = best_sample[i]
    return z

def _solve_qubo_sa(Q: np.ndarray, num_reads: int, seed: Optional[int]) -> np.ndarray:
    """Fallback hand-rolled simulated annealing."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    n = len(Q)
    best_global_z = None
    best_global_E = float('inf')
    
    T_start = 10.0
    T_end = 0.01
    num_sweeps = 1000 * n
    
    cooling_factor = (T_end / T_start) ** (1.0 / num_sweeps)
    
    for _ in range(num_reads):
        z = np.random.randint(0, 2, size=n)
        T = T_start
        
        for _ in range(num_sweeps):
            i = random.randrange(n)
            
            sum_cross = np.dot(Q[i, :], z) - Q[i, i] * z[i]
            delta = (1 - 2 * z[i]) * (Q[i, i] + 2 * sum_cross)
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                z[i] = 1 - z[i]
                
            T *= cooling_factor
            
        # compute final energy
        E = np.dot(z, np.dot(Q, z))
        if E < best_global_E:
            best_global_E = E
            best_global_z = z.copy()
            
    return best_global_z
