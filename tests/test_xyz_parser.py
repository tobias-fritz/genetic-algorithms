import pandas as pd
import pytest
from src.xyz_parser import XYZ_reader

def test_xyz_reader(tmp_path):
    content = "\n".join([
        "3",
        "comment line",
        "H    0.0   0.0   0.0",
        "O    0.0   0.0   1.0",
        "H    1.0   0.0   0.0",
    ]) + "\n"
    f = tmp_path / "molecule.xyz"
    f.write_text(content)
    df = XYZ_reader(str(f))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['Element', 'x', 'y', 'z']
    assert df.shape == (3, 4)
    assert list(df['Element']) == ['H', 'O', 'H']
    assert pytest.approx(df.loc[1, 'z']) == 1.0
