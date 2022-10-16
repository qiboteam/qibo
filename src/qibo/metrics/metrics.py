# -*- coding: utf-8 -*-
import numpy as np
import scipy


def trace_distance(state, target):
    if state.shape != target.shape:
        raise TypeError(f'State has dims {state.shape} while target has dims {target.shape}.')

    difference = state - target
    difference_sqrt, _ = scipy.linalg.sqrtm(np.dot(difference.T.conj(), difference))
    return np.trace(difference_sqrt) / 2


def hilbert_schmidt_distance(state, target):
    if state.shape != target.shape:
        raise TypeError(f'State has dims {state.shape} while target has dims {target.shape}.')
    
    return np.trace((state - target)**2)


def fidelity(state, target):

    if isinstance(state, list):
        state = np.asarray(state)
    if isinstance(target, list):
        target = np.asarray(target)

    if state.shape != target.shape:
        raise TypeError(f'State has dims {state.shape} while target has dims {target.shape}.')
    
    if len(state.shape) == 1 and len(target.shape) == 1:
        return np.abs(np.dot(state.conj(), target))**2
    elif len(state.shape) == 2 and len(target.shape) == 2:
        return np.trace(np.dot(state.T.conj(), target))
    else:
        raise TypeError(f'Both objects must have dims either (k,) or (k,l), but have dims {state.shape} and {target.shape}')



def process_fidelity(channel, target = None):
    if target:
        if channel.shape != target.shape:
            raise TypeError(f'Channels must have the same dims, but {channel.shape} != {target.shape}')
    d = channel.shape[0]
    if target is None:
        return np.trace(channel) / d**2
    else:
        return np.trace(np.dot(channel.T.conj(), target)) / d**2


def average_gate_fidelity(channel, target = None):
    d = channel.shape[0]
    return (d * process_fidelity(channel, target) + 1) / (d + 1)

