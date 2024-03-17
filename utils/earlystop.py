import torch
import torch.nn as nn

import numpy as np


class EarlyStopping():
    def __init__(self, patience):
        self.metric = np.array([])
        self.patience = patience
        self.earlystop = False

    def update(self, metric):
        self.metric = np.append(self.metric, metric)
        self._check()

    def _check(self):
        if len(self.metric) >= self.patience:
            tmp = self.metric[-self.patience]
            self.earlystop = np.all(tmp == self.metric[-1])
