import sys
import os
import logging
from src.experiment.config import RunConfig
from src.experiment.runner import run_single

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

# Run just Room 6x6-2a with qa, alpha=0.5, beta=0.5, seed=0
config = RunConfig("Room 6x6-2a", 0.5, 0.5, 0, "qa")
result = run_single(config, "debug_results")
print("Converged:", result.converged)
print("Outer iters:", result.iterations_outer)
