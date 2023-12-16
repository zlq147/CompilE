from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class F2(Regularizer):
    def __init__(self, weight: float):
        super(F2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(f ** 2)
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]


class DURA(Regularizer):
    def __init__(self, weight: float):
        super(DURA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0

        # for factor in factors:
        h, r, t = factors

        norm += torch.sum(t**2 + h**2)
        norm += torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]

class DURA_W(Regularizer):
    def __init__(self, weight: float):
        super(DURA_W, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        # for factor in factors:
        h, r, t = factors

        norm += 0.5 * torch.sum(t**2 + h**2)
        norm += 1.5 * torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]

class DURA_RESCAL(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        
        h, r, t = factors
        norm += torch.sum(h ** 2 + t ** 2)
        norm += torch.sum(
            torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + torch.bmm(r, t.unsqueeze(-1)) ** 2)
        return self.weight * norm / h.shape[0]