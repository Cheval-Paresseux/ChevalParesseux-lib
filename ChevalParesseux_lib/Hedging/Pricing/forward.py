import numpy as np
import pandas as pd
from typing import Self

class Forward_Pricer:
    def __init__(
        self, 
        n_jobs: int
    ) -> Self:
        self.n_jobs = n_jobs
    
    