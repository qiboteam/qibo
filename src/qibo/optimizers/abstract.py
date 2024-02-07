from abc import ABC, abstractmethod
from dataclasses import dataclass

from qibo.config import raise_error


@dataclass
class Optimizer(ABC):

    @abstractmethod
    def fit(self):
        """Compute the optimization strategy."""
        raise_error(NotImplementedError)
