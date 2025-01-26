import pytest
import pandas as pd
from src.GA_util import gen_reference

def test_gen_reference():

    product_xyz = pd.DataFrame({'Element': ['H', 'O', 'O', 'H', 'H', 'H'],
                                'x': [1.65176, 1.10136, 2.49254, 0.81685, 2.77705, -0.03109],
                                'y': [5.11644, 3.88800, 4.49960, 3.73177, 4.65584, 4.27845],
                                'z': [0.11994, 0.08262, 0.08262, -0.83146, 0.99670, 0.18183]})
    
    result = ([[0, 3, 4, 5], [1, 2]],[[0, 3, 4, 5], [1, 2], [1, 2], [0, 3, 4, 5], [0, 3, 4, 5], [0, 3, 4, 5]])
    print(gen_reference(product_xyz))
    assert gen_reference(product_xyz) == result

def test_init_pop():
    educt = pd.DataFrame([[0.000000, 0.432013, 0.384594, 0.294334, 0.782609, 0.355837],
                        [0.448326, 0.000000, 0.398623, 0.416336, 0.368968, 0.754755],
                        [0.358703, 0.358259, 0.000000, 0.729897, 0.339634, 0.332968],
                        [0.259486, 0.353687, 0.689925, 0.000000, 0.253662, 0.355216],
                        [0.702385, 0.319097, 0.326821, 0.258234, 0.000000, 0.258220],
                        [0.331147, 0.676829, 0.332231, 0.374964, 0.267750, 0.000000]])
    product = pd.DataFrame([[0.000000, 0.729897, 0.339634, 0.332968, 0.358703, 0.358259],
                            [0.689925, 0.000000, 0.253662, 0.355216, 0.259486, 0.353687],
                            [0.326821, 0.258234, 0.000000, 0.258220, 0.702385, 0.319097],
                            [0.332231, 0.374964, 0.267750, 0.000000, 0.331147, 0.676829],
                            [0.384594, 0.294334, 0.782609, 0.355837, 0.000000, 0.432013],
                            [0.398623, 0.416336, 0.368968, 0.754755, 0.448326, 0.000000]])

    referece = [[0, 1, 2, 3], [4, 5]]
    pass
