import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.ou_research import fit_ou

def test_ou_estimation_recovers_mean_reversion():
    rng=np.random.default_rng(7); x=[0.0]; phi=0.9
    for _ in range(1000): x.append(phi*x[-1]+rng.normal(scale=0.3))
    p=fit_ou(pd.Series(x))
    assert 0 < p['phi'] < 1
    assert p['theta'] > 0
    assert p['half_life'] > 0
