from typing import Union, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from hmmlearn import hmm
from hmmquant import __version__
from pandas.core.frame import DataFrame
from scipy.stats import kstest
from scipy.stats.stats import mode


def test_version():
    assert __version__ == "0.1.0"
