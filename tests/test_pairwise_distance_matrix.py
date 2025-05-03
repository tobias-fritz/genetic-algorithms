import pandas as pd
import numpy as np
from src.pairwise_distance_matrix import pairwise_distance_matrix

def test_pairwise_distance_matrix_unit():
    df = pd.DataFrame([
        {'Element': 'A', 'x': 0.0, 'y': 0.0, 'z': 0.0},
        {'Element': 'B', 'x': 0.0, 'y': 0.0, 'z': 1.0},
    ])
    mat = pairwise_distance_matrix(df, inverse=False, unit=True)
    # distance between 0 and 1 is 1.0
    assert np.isclose(mat.iloc[0,1], 1.0)
    assert mat.shape == (2, 2)
