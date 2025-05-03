import numpy as np
import pandas as pd
import random
from src.GA_util import objective, mutation, crossover

def test_objective_identity():
    educt = np.eye(3)
    product = pd.DataFrame(np.eye(3))
    # perfect alignment => sum of diagonal = 3
    assert objective([0,1,2], educt, product) == pytest.approx(3.0)

def test_mutation_keeps_length_and_uniqueness():
    random.seed(0)
    p = [0, 1, 2, 3]
    ref = [[0,1], [0,1], [2,3], [2,3]]
    out = mutation(p.copy(), ref, mutation_rate=1.0)
    assert len(out) == 4
    assert len(set(out)) == 4

def test_crossover_swaps_segments():
    p1 = [0,1,2,3]
    p2 = [3,2,1,0]
    child1, child2 = crossover(p1, p2, r_cross=1.0)
    # they must still be same length and contain same elements
    assert set(child1) == set(p1)
    assert set(child2) == set(p2)
