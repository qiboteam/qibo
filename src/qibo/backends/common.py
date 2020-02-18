# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
from abc import ABCMeta, abstractmethod


class Backend(object):
    """A common class with all the required abstract methods for the backend implementation"""

    __metaclass__ = ABCMeta

    def __init__(self):
        """Initializes the class attributes"""
        self._output = {'virtual_machine': None, 'wave_func': None, 'measure': []}

    @abstractmethod
    def CNOT(self, **args):
        """The Controlled-NOT gate."""
        pass

    @abstractmethod
    def H(self, **args):
        """The Hadamard gate."""
        pass

    @abstractmethod
    def X(self, **args):
        """The Pauli X gate."""
        pass

    @abstractmethod
    def Y(self, **args):
        """The Pauli Y gate."""
        pass

    @abstractmethod
    def Z(self, **args):
        """The Pauli Z gate."""
        pass

    @abstractmethod
    def Barrier(self, **args):
        """The barrier gate."""
        pass

    @abstractmethod
    def S(self, **args):
        """The swap gate."""
        pass

    @abstractmethod
    def T(self, **args):
        """The Toffoli gate."""
        pass

    @abstractmethod
    def Iden(self, **args):
        """The identity gate."""
        pass

    @abstractmethod
    def MX(self, **args):
        """The measure gate X."""
        pass

    @abstractmethod
    def MY(self, **args):
        """The measure gate Y."""
        pass

    @abstractmethod
    def MZ(self, **args):
        """The measure gate Z."""
        pass

    @abstractmethod
    def Flatten(self, **args):
        """Set wave function coefficients"""
        pass

    @abstractmethod
    def execute(self, model, shots):
        """Executes the circuit on a given backend.

        Args:
            model: (qibo.models.Circuit): The circuit to be executed.
            shots (int): number of trials.
        """
        pass