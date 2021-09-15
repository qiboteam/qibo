import numpy as np
import matplotlib.pyplot as plt
import copy
from qibo.config import EIGVAL_CUTOFF




class DegeneratedGap():

    def __init__(self, energy_levels):
        self._energy_levels, self._n_levels = self._get_degenerated_levels(energy_levels) # List[List[float(Energy_Value)]]
        self._n_steps = len(self._energy_levels[0])



    def _get_degenerated_levels(self, energy_levels):
        for i, level in enumerate(energy_levels):
            if level[-1]-energy_levels[0][-1] > EIGVAL_CUTOFF:
                return energy_levels[:i+1], i+1   

        raise ValueError('ERROR: All energy levels passed are degenerated with the ground state.')

    def _check_larger_gaps(self, current, step):
        larger_gaps = []
        for i in range(self._n_levels-1, current[1], -1):
            if (self._energy_levels[i][step] - self._energy_levels[i-1][step]) > (
                self._energy_levels[current[1]][step]-self._energy_levels[current[0]][step]):
                larger_gaps.append((i-1, i))           
        return larger_gaps
    


class Method_1(DegeneratedGap):

    def __init__(self, energy_levels):
        super(Method_1, self).__init__(energy_levels)

    def compute_gap(self):
        current = (0,1)
        gap = [[], []]
        
        for step in range(self._n_steps):
            larger_gaps = self._check_larger_gaps(current, step)
            if larger_gaps != []:
                current = larger_gaps[0]

            gap[0].append(self._energy_levels[current[1]][step]-self._energy_levels[current[0]][step])
            gap[1].append((current[0], current[1]))
        return gap, self._energy_levels  # List[List[float(gap_value)], List[Tuple(int(lower_level), int(upper_level))]]

class Method_2(DegeneratedGap):

    def __init__(self, energy_levels):
        super(Method_2, self).__init__(energy_levels)

    def compute_gap(self, mode = 'minimum'):
        gaps=[[[self._energy_levels[1][0]-self._energy_levels[0][0]], [(0,1)]]]
        
        for step in range(1, self._n_steps):
            prev = copy.deepcopy(gaps)

            for i, k in enumerate(prev):
                current = k[1][-1]
                to_add = self._check_larger_gaps(current, step)
                for new in to_add:
                    gaps.append([ k[0]+[self._energy_levels[new[1]][step]-self._energy_levels[new[0]][step]],
                                k[1]+[new] ])

                gaps[i][0].append(self._energy_levels[current[1]][step]-self._energy_levels[current[0]][step])
                gaps[i][1].append(current)

            prev = copy.deepcopy(gaps)
            to_delete=[]
            for i, k in enumerate(prev):
                for j, l in enumerate(prev[i+1:]):
                    if k[1][-1] == l[1][-1]:
                        if mode == 'minimum':
                            if min(k[0]) < min(l[0]):
                                to_delete.append(i)
                            else:
                                to_delete.append(j+i+1)

                        elif mode == 'integrate':
                            if sum(g**2 for g in k[0]) < sum(g**2 for g in l[0]):
                                to_delete.append(i)
                            else:
                                to_delete.append(j+i+1)


            to_delete = list(set(to_delete))
            to_delete.sort(reverse = True)
            for index in to_delete:
                del gaps[index]

        optimal_way = gaps[0]
        if mode == 'minimum':
            for way in gaps:
                if min(way[0]) > min(optimal_way[0]):
                    optimal_way = way

        elif mode == 'integrate':
            for way in gaps:
                if sum(way[0]) > sum(optimal_way[0]):
                    optimal_way = way

        return optimal_way, self._energy_levels  # List[List[float(gap_value)], List[Tuple(int(lower_level), int(upper_level))]]

class Method_3(DegeneratedGap):

    def __init__(self, energy_levels):
        super(Method_3, self).__init__(energy_levels)

    def _check_larger_gaps(self, current, step):
        raise NotImplementedError()

    def _check_ground_state(self, gap, get_gap_list=False):
        level = 0
        gap_list = [[], []]

        for step in range(self._n_steps):
            level = self._check_gap(level, step, gap)

            if get_gap_list:
                gap_list[0].append(self._energy_levels[level+1][step]-self._energy_levels[level][step])
                gap_list[1].append((level, level+1))

        if level == self._n_levels-1:
            return False, gap_list
        else:
            return True, gap_list

    def _check_gap(self, level, step, gap):
        if level == self._n_levels-1:
            return level

        elif self._energy_levels[level+1][step]-self._energy_levels[level][step] >= gap:
            return level
            
        else:
            level = level + 1
            level = self._check_gap(level, step, gap)
            return level

    def compute_gap(self, precision, estimation = None):
        if estimation == None:
            max_gap = self._energy_levels[-1][-1] - self._energy_levels[0][-1]
            min_gap = 0
        else:
            max_gap = estimation[1]
            min_gap = estimation[0]
            if (self._check_ground_state(min_gap) == False) or (
                self._check_ground_state(max_gap) == True):
                print('ERROR: The minimum gap is not in the estimation range.')

        max_iter = int(np.log2((max_gap-min_gap)/precision))+1
        for i in range(max_iter):
            new_gap = (max_gap - min_gap)/2. + min_gap

            kept_ground_state, gap_list = self._check_ground_state(new_gap)

            if kept_ground_state == True:
                min_gap = new_gap
            else:
                max_gap = new_gap

            if (max_gap - min_gap) < precision:
                _, gap_list = self._check_ground_state(min_gap, get_gap_list=True)
                return gap_list, self._energy_levels

        _, gap_list = self._check_ground_state(min_gap, get_gap_list=True)
        return gap_list, self._energy_levels # List[List[float(gap_value)], List[Tuple(int(lower_level), int(upper_level))]]


    
