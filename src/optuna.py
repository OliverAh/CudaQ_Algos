import optuna
import hhl
import numpy as np
import pandas as pd

from typing import List, Tuple, Callable

class ParamStudy_hhl():
    def __init__(self, objective:Callable[optuna.Trial]=None, directions:List[str]=None):
        self.objective = objective
        self.directions = directions
        self.study = optuna.create_study()

