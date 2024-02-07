from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from qibo.config import raise_error


@dataclass
class Optimizer(ABC):

    verbosity: bool = field(default=True)
    """Verbosity of the optimization process. If True, logging messages will be displayed."""

    @abstractmethod
    def fit(self):
        """Compute the optimization strategy."""
        raise_error(NotImplementedError)
