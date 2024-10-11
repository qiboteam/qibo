

import os
import pickle
import multiprocessing
import numpy as np
import time

# ----------------------------------------- #
#   class method to deal with each label    #
# ----------------------------------------- #

class Measurement:
    """ to deal with only label measurement results
    """
    def __init__(self, label, count_dict):
        """ initialization of shot measurement results for a given label
            according to the given count dict

        Args:
            label (str): a Pauli string corresponding to the sampled Pauli operator
                        (e.g. XYZXX)
            count_dict (dict): counts of each possible binary output string (e.g. '000...00', '000...01', etc.)
            For example, count_dict = {'101': 617, '000': 581, '111': 574, '010': 628}

        Initializes Measurement class
        - label: a string representing a Pauli matrix (e.g. XYZXX)
        - count_dict: counts of each possible binary output string (e.g. '000...00', '000...01', etc.)
        - num_shots: number of shots measurement is taken to get empirical frequency through counts
        """

        akey = list(count_dict.keys())[0]
        assert len(label) == len(akey)

        self.label            = label
        self.count_dict       = self.zfill(count_dict)
        self.num_shots        = sum(count_dict.values())

    @staticmethod
    def zfill(count_dict):
        """ to pad 0 in the front of the keys of the count_dict such that
            their lengths are equal to the qubit number. 

        Args:
            count_dict (dict): counts of each possible binary output string (e.g. '000...00', '000...01', etc.)
            For example, count_dict = {'101': 617, '000': 581, '111': 574, '010': 628}

        Returns:
            dict: (result) same as count_dict but with the keys being filled with 0 in 
                the front to let their lengths be equal to the qubit number. 
        """
        n = len(list(count_dict.keys())[0])
        d = 2 ** n
        result = {}
        for idx in range(d):
            key = bin(idx)[2:]
            key = key.zfill(n)
            result[key] = count_dict.get(key, 0)
        return result



    @staticmethod
    def naive_parity(key):
        """ Calculates the number of '1' in the given key

        Args:
            key (str): the measurement outcome in the form of 0 or 1
                    i.e. possible output binary string
                    (eg)  '101' or '110' for the 3 qubit measurement

        Returns:
            int: the number of '1' in the key
        """
        return key.count('1')

    
    def effective_parity(self, key):
        """ Calculates the effective number of '1' in the given key

        Args:
            key (str): the measurement outcome in the form of 0 or 1
                    (eg)  '101' or '110' for the 3 qubit measurement
        Returns:
            int: the number of effective '1' in the key
        """
        indices    = [i for i, symbol in enumerate(self.label) if symbol == 'I']
        digit_list = list(key)
        for i in indices:
            digit_list[i] = '0'
        effective_key = ''.join(digit_list)

        return effective_key.count('1')

    def parity(self, key, parity_flavor='effective'):
        """ determine the (effective) number of '1' in the key
                according to parity_flavor
        Args:
            key (str): the measurement outcome in the form of 0 or 1
                    (eg)  '101' or '110' for the 3 qubit measurement
            parity_flavor (str, optional): Determine different ways of calculating parity. 
                    Defaults to 'effective'.

        Returns:
            int: the (effective) number of '1' in the key
                according to parity_flavor
        """
        if parity_flavor == 'effective':
            return self.effective_parity(key)
        else:
            return self.naive_parity(key)
            
    
    def get_pauli_correlation_measurement(self, beta=None, parity_flavor='effective'):
        """ Generate Pauli correlation measurement (expectation value of Pauli monomials).
        Note that summation of d=2^n Pauli basis measurement corresponds to one Pauli correlation measurement.

        Args:
            beta (float, optional): for shifting the counted number (not used now). Defaults to None.
            parity_flavor (str, optional): Determine different ways of calculating parity. Defaults to 'effective'.

        Returns:
            dict: dictionary with Pauli label as the key and its corresponding coefficient in the density matrix 
                expansion as the value of the dictionary
        """

        if beta == None:
        #    beta = 0.50922         #  original MiFGD usage
            beta = 0.0              #  this one looks more exact
        num_shots          = 1.0 * self.num_shots
        num_items          = len(self.count_dict)

        #frequencies        = {k : (v + beta) / (num_shots + num_items * beta) for k, v in self.count_dict.items()}
        #parity_frequencies = {k : (-1) ** self.parity(k, parity_flavor) * v for k, v in frequencies.items()}
        #correlation        = sum(parity_frequencies.values())
        #data = {self.label : correlation}

        freq2          = {k : (v ) / (num_shots ) for k, v in self.count_dict.items()}
        parity_freq2   = {k : (-1) ** self.parity(k, parity_flavor) * v for k, v in freq2.items()}
        correlation2   = sum(parity_freq2.values())
        data2 = {self.label : correlation2}

        return data2


    def get_pauli_basis_measurement(self, beta=None):
        """ Generate Pauli basis measurement. 
        Note that summation of d=2^n Pauli basis measurement corresponds to one Pauli correlation measurement.

        Args:
            beta (float, optional): for shifting the counted number. Defaults to None.

        Returns:
            dict: dictionary with Pauli label as the key and 
                the effective frequencies for each possible binary outcomes as the value of the dictionary
        """

        if beta == None:
            beta = 0.50922
        num_shots   = 1.0 * self.num_shots
        num_items   = len(self.count_dict)
        frequencies = {k : (v + beta) / (num_shots + num_items * beta) for k, v in self.count_dict.items()}
        data = {self.label: frequencies}
        return data



    def _pickle_save(self, fpath):
        """ to save the count_dict in the pickle format, where
            count_dict is a dictionary
            that counts of each possible binary output string (e.g. '000...00', '000...01', etc.)
            For example, count_dict = {'101': 617, '000': 581, '111': 574, '010': 628}
        
        Args:
            fpath (str): the path to store count_dict
        """
        with open(fpath, 'wb') as f:
            pickle.dump(self.count_dict, f)


    def _hdf5_save(self, fpath):
        """ to save the count_dict in the hdf5 format

        Args:
            fpath (str): the path to store count_dict
        """
        f = h5py.File(fpath, 'a')
        group = f.create_group(self.label)
        
        items = [[key, value] for key, value in self.count_dict.items()]
        keys   = np.array([item[0] for item in items], dtype='S')
        values = np.array([item[1] for item in items], dtype='int32')
        
        dataset = group.create_dataset('keys',   data = keys)
        dataset = group.create_dataset('values', data = values) 
        f.close()

    # Save the measurement to disk
    def save(self, path):
        """ to save the count_dict either in the pickle or hdf5 format

        Args:
            fpath (str): the path to store count_dict
        """
        if os.path.isdir(path):
            fpath = os.path.join(path, '%s.pickle' % self.label)
            self._pickle_save(fpath)
        elif path.endswith('.hdf5'):
            fpath = path
            self._hdf5_save(fpath)

            
    @classmethod
    def _pickle_load(cls, fpath):
        """ to load count_dict from the saved pickle file

        Args:
            fpath (str): the path of the stored count_dict
        Returns:
            dict: (count_dict) which is a dictionary
                that counts of each possible binary output string (e.g. '000...00', '000...01', etc.)
                For example, count_dict = {'101': 617, '000': 581, '111': 574, '010': 628}
        """
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        return data


    @classmethod
    def _hdf5_load(cls, fpath, label):
        """ to load count_dict from the saved hdf5 file

        Args:
            fpath (str): the path of the stored count_dict
            label (str): the Pauli operator label

        Returns:
            dict: (count_dict) which is a dictionary
                that counts of each possible binary output string (e.g. '000...00', '000...01', etc.)
                For example, count_dict = {'101': 617, '000': 581, '111': 574, '010': 628}
        """

        f = h5py.File(fpath, 'r')
        group = f[label]
        keys   = group['keys'][:]
        values = group['values'][:]

        data = {k: v for k, v in  zip(*[keys, values])}
        return data


    # Load a measurement from disk
    @classmethod
    def load(cls, path, label, num_leading_symbols=0):
        """ to load a count_dict from either pickle file or h5py file
            and convert it to the class Measurement object

        Args:
            path (str): the path of the stored count_dict
            label (str): the Pauli operator label
            num_leading_symbols (int, optional): number of effective qubit numbers in the labels. Defaults to 0.

        Returns:
            cls: the object of class Measurement
        """
        if os.path.isdir(path):
            if num_leading_symbols == 0:
                fpath = os.path.join(path, '%s.pickle' % label)
                count_dict  = cls._pickle_load(fpath)
            else:
                fragment_name = label[:num_leading_symbols]
                fpath         = os.path.join(path, fragment_name, '%s.pickle' % label)
                count_dict    = cls._pickle_load(fpath) 
        elif path.endswith('.hdf5'):
            fpath = path
            count_dict = cls._hdf5_load(fpath, label)
        measurement = cls(label, count_dict)
        return measurement

# --------------------------------------------- #
#    functions to calculate measurement_list    #
#       -->  good for parallel calc             #
# --------------------------------------------- #
def measurement_list_calc(label_list, count_dict_list):
    """ to calculate measurement_list corresponding to coefficients of the target density matrix
        basis expansion respect to label_list representing sampled Pauli operators

    Args:
        label_list (list): list of label names representing the sampled Pauli operators
                    i.e. each label is a Pauli string, e.g. XYZXX
        count_dict_list (list): list of dictionaries where each dictionary is a count_dict
            - count_dict: counts of each possible binary output string (e.g. '000...00', '000...01', etc.)
            For example, count_dict = {'101': 617, '000': 581, '111': 574, '010': 628}
    Returns:
        list: (measurement_list) list of coefficients corresponding to basis expansions
            in each sampled Pauli operator 
    """
    
    parity_flavor='effective'
    beta = None
    measurement_object_list = [Measurement(label, count_dict) for (label, count_dict) in 
                                    zip(*[label_list, count_dict_list])]

    measurement_list = [measurement_object.get_pauli_correlation_measurement(beta, parity_flavor)[label] for
                            (label, measurement_object) in zip(*[label_list,
                                                                measurement_object_list])]
    #print('label_list       = {}'.format(label_list))
    #print('measurement_list = {}'.format(measurement_list))
    return measurement_list


def measurement_list_calc_wID_wrap(argv):
    """ a wrapper function to call measurement_list_calc_wID

    Args:
        argv (tuple): all the necessary arguments for calling measurement_list_calc_wID

    Returns:
        dir: {ID: measurement_list} which is the output of measurement_list_calc_wID
    """

    out = measurement_list_calc_wID(*argv)
    return out


def measurement_list_calc_wID(ID, label_list, count_dict_list):
    """ same as measurement_list_calc but with given ID as a index for specifying different cpu in parallel computation.
    The effect is to calculate measurement_list corresponding to coefficients of the target density matrix
        basis expansion respect to label_list representing sampled Pauli operators

    Args:
        ID (int): an index for parallel computation usage
        label_list (list): list of label names representing the sampled Pauli operators
                    i.e. each label is a Pauli string, e.g. XYZXX
        count_dict_list (list): list of dictionaries where each dictionary is a count_dict
            - count_dict: counts of each possible binary output string (e.g. '000...00', '000...01', etc.)
            For example, count_dict = {'101': 617, '000': 581, '111': 574, '010': 628}

    Returns:
        dir: {ID: measurement_list}, where
            ID is the parallel computation index, and
            measurement_list is list of coefficients corresponding to basis expansions
                    in each sampled Pauli operator 
    """

    print('      start to calc    {}-th   label_list'.format(ID))

    parity_flavor='effective'
    beta = None
    measurement_object_list = [Measurement(label, count_dict) for (label, count_dict) in 
                                    zip(*[label_list, count_dict_list])]

    measurement_list = [measurement_object.get_pauli_correlation_measurement(beta, parity_flavor)[label] for
                            (label, measurement_object) in zip(*[label_list,
                                                                measurement_object_list])]

    return {ID: measurement_list}


# ----------------------------- #
#       for parallel CPU        #
# ----------------------------- #
def split_list(x, num_parts):
    """ to slice list x into num_parts separated elements

    Args:
        x (list): list of elements which are supposed to be Pauli operator labels
        num_parts (int): the number of parts we want to split x 

    Returns:
        list: list of separated list x into num_parts elments.
            (eg) x = ['YYY', 'XXZ', 'XYY', 'YIY', 'ZIZ', 'XXY', 'IXZ', 'IIX', 'ZXY', 'IIZ']
            then split_list(x, 3) = [['YYY', 'XXZ', 'XYY'], ['YIY', 'ZIZ', 'XXY'], ['IXZ', 'IIX', 'ZXY', 'IIZ']]
    """
    n = len(x)
    size = n // num_parts
    parts = [x[i * size: (i+1) * size] for i in range(num_parts - 1 )]
    parts.append(x[(num_parts - 1) * size:])
    return parts


# --------------------------------- #
#   class to deal with all labels   #
# --------------------------------- #

class MeasurementStore:
    """ to deal with measurement results for a list of labels
    """
    def __init__(self,
                 measurement_dict):
        """ to initialize the measurement results for a list of sampled Pauli labels

        Args:
            measurement_dict (dict): a dictionary of shot measurement results

            (eg) measurement_dict = 
                {'YXI': {'101': 300, '110': 306, '111': 282, '000': 307, '001': 319, '010': 283, '100': 291, '011': 312},
                 'ZZI': {'000': 1181, '111': 1219}, 
                 'YXX': {'101': 294, '010': 291, '110': 291, '100': 293, '001': 322, '111': 295, '011': 286, '000': 328}, 
                 'ZIZ': {'000': 1193, '111': 1207}, 
                 'ZYY': {'100': 313, '010': 299, '101': 289, '011': 277, '110': 308, '001': 319, '000': 302, '111': 293}, 
                 'XXZ': {'101': 292, '001': 321, '100': 288, '010': 284, '000': 298, '110': 299, '011': 328, '111': 290}, 
                 'ZYI': {'101': 617, '000': 581, '111': 574, '010': 628}, 
                 'YXZ': {'001': 295, '101': 317, '111': 328, '110': 300, '010': 306, '000': 288, '100': 279, '011': 287}, 
                 'YYI': {'001': 290, '101': 291, '010': 307, '111': 299, '100': 284, '000': 329, '110': 299, '011': 301}, 
                 'IIZ': {'000': 1220, '111': 1180}}

        Initializes MeasurementStore class
        - measurement_dict: the shot measurement result which is given when calling the class constructor
        - labels: list of strings of Pauli matrices (where each label is a Pauli matrix, e.g. XYZXX)
        - size  : number of labels
        """
        self.measurement_dict = measurement_dict
        
        self.labels = list(measurement_dict.keys())
        self.size   = len(self.labels)

    @classmethod
    def calc_measurement_list(cls, label_list, data_dict, method=0):
        """calculate the measurement list which is the list of coefficients corresponding to each label (Pauli projector)

        Args:
            label_list (list): list of labels of Pauli operators
            data_dict (dict): dictionary storing all shot measurement results of each label
            method (int, optional): method of converting data_dict to measurement_list. Defaults to 0.
                where the measurement_list is a list of numbers corresponding to coefficients of Pauli operators
                    in the basis expansion of the density matrix of interest

        Returns:
            list: (measurement_list) list of float numbers corresponding to coefficients of Pauli operators
        """

        print('    -------------------         start calc measurment_list        ------------------  \n')

        count_dict_list = [data_dict[label] for label in label_list]
        del data_dict


        if method == 0:
            print('       ****   directly calc & save measurement_list  ')
            print('       ****   len(label_list) = {},  len(data_dict) = {}\n'.format(len(label_list), len(count_dict_list)))
            measurement_list = measurement_list_calc(label_list, count_dict_list)

        elif method == 1:           #  parallel CPU

            num_CPU = multiprocessing.cpu_count()
            if num_CPU > len(label_list):
                num_CPU = 3
            print('       ****  use parallel #CPU = {} to calc measurement_list \n'.format(num_CPU))
                        
            ID_list         = np.arange(num_CPU)
            label_part      = split_list(label_list, num_CPU)            
            count_dict_part = split_list(count_dict_list, num_CPU)

            del count_dict_list

            pool = multiprocessing.Pool(num_CPU)

            Run_parallel = 2

            if Run_parallel == 1:
                L_pt = pool.starmap(measurement_list_calc_wID, zip(ID_list, label_part, count_dict_part))
                pool.close()
                pool.join()

                ml_dict = {}
                {ml_dict.update(xx) for xx in L_pt}         #  if return  {ID, measurement_list}

            elif Run_parallel == 2:

                ml_dict = {}
                mp_collect = []
                for ii, labels in enumerate(label_part):
                    print('       **************  {}-th label_part to be parallelly calc   *******'.format(ii))

                    mp_out = pool.apply_async(measurement_list_calc_wID_wrap, ([ii, labels, count_dict_part[ii]], ))
                    mp_collect.append(mp_out)
                pool.close()
                pool.join()
                del count_dict_part

                for xx in mp_collect:
                    out = xx.get()

                    ml_dict.update(out)


            measurement_list = []
            for xx in ID_list:
                measurement_list += ml_dict[xx]

        print('    -------------         DONE of calculating measurment_list       ------------------  \n')

        return measurement_list

    @classmethod
    def Save_measurement_by_data_dict(cls, path, label_list, data_dict, method=0, Name=None, ToSave=1):
        """ to obtain the measurement_list (numbers corresponding to coefficients of each Pauli operator)
            from the given data_dict (the direct shot measurements for each Pauli operator)

        Args:
            path (str): directory path of saving the measurement results
            label_list (list): list of Pauli operator labels
            data_dict (dict): dictionary of shot measurement results
            method (int, optional): method of coverting data_dict to measurement_list. Defaults to 0.
            Name (str, optional): the name appendix of the file saving the results. Defaults to None.
            ToSave (int, optional): To save the measurement_list into the file or not. Defaults to 1.

        Returns:
            list: (label_list) list of Pauli operator labels
            list: (measurement_list) list of numbers representing the coefficients corresponding to
                            each Pauli operator
        """

        tt1 = time.time()
        measurement_list = cls.calc_measurement_list(label_list, data_dict, method)
        tt2 = time.time()
        print('    ******   cls.calc_measurement_list   -->  time = {}   *****\n'.format(tt2-tt1))

        if ToSave == 1:
            tt3 = time.time()
            cls.Save_measurement_list_file(path, label_list, measurement_list, Name)
            tt4 = time.time()
            print('    ******   cls.Save_measurement_list_file   -->  time = {}   ****'.format(tt4-tt3))

        return label_list, measurement_list


    def Save_measurement_list(self, path, method=0, Name=None):
        """ to call self.calc_measurement_list function to convert 
                self.data_dict (shot measurement record) into measurement_list (coefficients of labels)
            & save the results by calling self.Save_measurement_list_file

        Args:
            path (str): directory path storing the measurements results
            method (int, optional): method passing to self.calc_measurement_list function. Defaults to 0.
            Name (str, optional): name appendix of files for storing the measurement results. Defaults to None.

        Returns:
            list: (label_list) list of Pauli operator labels
            list: (measurement_list) list of Pauli operator coefficients in the expansion of the density matrix 
                    calculated from self.data_dict (shot measurement data for each label).
        """
        print(' -------------  to calc & save measurement_list  -------------- \n')

        label_list = self.label_list
        data_dict  = self.data_dict

        measurement_list = self.calc_measurement_list(label_list, data_dict, method)

        self.Save_measurement_list_file(path, label_list, measurement_list, Name)

        return label_list, measurement_list

    @classmethod
    def Load_measurement_list(cls, path, Name=None):
        """ to load the measurement list of Pauli operator coefficients from the specified path

        Args:
            path (str): the path to the stored measurement list
            Name (str, optional): suffix of the file name. Defaults to None.

        Returns:
            list : (label_list) list of Pauli operator labels
            list : (measurement_list) list of coefficients of the density matrix
                in the basis expansion corresponding to the labels
        """
        
        if Name == None:
            ml_file = '{}/measurement_list.pickle'.format(path)
            print(' ml_file  = {}'.format(ml_file))

            with open(ml_file, 'rb') as f:
                label_list, measurement_list = pickle.load(f)

        else:
            ml_file = '{}/measureL_{}.pickle'.format(path, Name)

            with open(ml_file, 'rb') as f:
                Name, label_list, measurement_list = pickle.load(f)

        print('  --> loading measurement_list DONE from  ml_file = {}\n'.format(ml_file))
        return label_list, measurement_list

    @classmethod
    def Save_measurement_list_file(cls, path, label_list, measurement_list, Name=None):
        """ to save the calculated measurement_list into a file

        Args:
            path (str): directory path of saved measurement_list
            label_list (list): list of labels
            measurement_list (list): list of coefficients for each Pauli operator label 
                 calculated from shot measurements
            Name (str, optional): name appendix of files storing the results. Defaults to None.
        """

        print('\n')
        print('          path = {} \n'.format(path))
        print('          Name = {} \n'.format(Name))
        if Name == None:
            ml_file = '{}/measurement_list.pickle'.format(path)

            with open(ml_file, 'wb') as f:
                pickle.dump([label_list, measurement_list], f)

        else:
            ml_file = '{}/measureL_{}.pickle'.format(path, Name)

            with open(ml_file, 'wb') as f:
                pickle.dump([Name, label_list, measurement_list], f)

        print('          ***  measurement_list --> ml_file = {} is saved\n'.format(ml_file))


    def saveInOne(self, path, Name=None):
        """ to save all the Pauli operator labels and the corresponding data_dict into files,
            instead of saving each individual Pauli operator label and data_dict into a single file

        Args:
            path (str): the path to save the measurement results
            Name (str, optional): the suffix to the file name to save. Defaults to None.
        """
        format = 'pickle'
        if not os.path.exists(path):
            os.mkdir(path)

        label_list = sorted(self.labels)

        data_dict = {}
        for label in label_list:
            data_dict[label] = self.measurement_dict[label]

        self.label_list = label_list
        self.data_dict  = data_dict

        # --------  save files ---------------- #   
        if Name == None:
            # --------  saved file for labels  ---------------- #
            label_file = '{}/labels.pickle'.format(path)

            with open(label_file, 'wb') as f:
                pickle.dump(label_list, f)
            print('      ***   label list is stored in {} \n'.format(label_file))

            # --------  saved file for count_dict  ------------- #
            Fname = '{}/count_dict.pickle'.format(path)

            with open(Fname, 'wb') as f:
                pickle.dump(data_dict, f)
            print('      ***   count_dict file = {} is dumped (i.e. saved) \n'.format(Fname))

        else:
            label_count_file = '{}/labs_data_{}.pickle'.format(path, Name)

            with open(label_count_file, 'wb') as f:
                pickle.dump([label_list, data_dict], f)
            print('      ***   label_list  &  count_dict_list  is stored in {} \n'.format(label_count_file))
    
    @classmethod
    def load_labels_data(cls, path, Name):
        """ to load Pauli operator labels and shot measurement data from path

        Args:
            path (str): the path to save the measurement results
            Name (str, optional): the suffix to the file name to save. Defaults to None.

        Returns:
            list : (label_list) list of Pauli operator labels
            list : data_dict (dict): dictionary storing all shot measurement results of each label

        """
        label_count_file = '{}/labs_data_{}.pickle'.format(path, Name)
        print('    label_count_file = {}\n'.format(label_count_file))

        with open(label_count_file, 'rb') as f:
            label_list, data_dict = pickle.load(f)

        return label_list, data_dict


    # Load measurements previously saved under a disk folder
    @classmethod
    def load_OneDict(cls, path, labels=None):
        """ to load labels and shot measurement results from stored files

        Args:
            path (str): the path to save the measurement results
            labels (list, optional): list of Pauli operator labels. Defaults to None.

        Returns:
            dict: (measurement_dict) shot measurement results
        """

        if labels == None:
            labels = cls.load_labels_from_file(path)

        Fname = '{}/count_dict.pickle'.format(path)
        print('  --->   Loading count_dict from {}\n'.format(Fname))
        with open(Fname, 'rb') as f:
            measurement_dict = pickle.load(f)
        print('  --->   count_dict is loaded \n')

        return measurement_dict

    @classmethod
    def load_labels_from_file(cls, path):
        """ Load a list of labels from labels.pickle in the path

        Args:
            path (str): the path to save the measurement results

        Returns:
            list: list of sampled Pauli operator labels
        """

        label_file = '{}/labels.pickle'.format(path)
        with open(label_file, 'rb') as f:
            labels = pickle.load(f)
        print('  --->   label_file = {} is loaded \n'.format(label_file))

        return labels


    def save(self, path):
        """ to save each Pauli operator label and its corresponding shot measurement data in
            each individual file.

        Args:
            path (str): the path to save the measurement results
        """
        format = 'hdf5'
        if not path.endswith('.hdf5'):
            format = 'pickle'
            if not os.path.exists(path):
                os.mkdir(path)
        for label, count_dict in self.measurement_dict.items():
            Measurement(label, count_dict).save(path)

    @classmethod
    def load_labels(cls, path):
        """ to load all the Pauli labels from collecting all the stored files in the path
            representing each individual label 

        Args:
            path (str): the path to load the saved measurement results

        Returns:
            list: list of sampled Pauli operator labels
        """
        if path.endswith('.hdf5'):
            with h5py.File(path, 'r') as f:
                labels = f.keys()
        else:
            try:
                labels = [fname.split('.')[0] for fname in os.listdir(path)]
            except:
                fragment_paths = [os.path.join(path, fragment) for fragment in os.listdir(path)]
                labels = []
                for fragment_path in fragment_paths:
                    fragment_labels = [fname.split('.')[0] for fname in os.listdir(fragment_path)]
                    labels.extend(fragment_labels)
        return labels

    
    
    # Load measurements previously saved under a disk folder
    @classmethod
    def load(cls, path, labels=None):
        """ to load all the shot measurement results from all the stored files, where each file
            contains only one label information

        Args:
            path (str): the path to load the saved measurement results
            labels (list, optional): list of Pauli operator labels. Defaults to None.
            
        Returns:
            dict: (measurement_dict) collecting all the shot measurements results for all the labels
        """

        if labels == None:
            labels = cls.load_labels(path)

        # checking if the store is fragmented and compute num_leading_symbols
        names = os.listdir(path)
        aname = names[0]
        apath = os.path.join(path, aname)
        if os.path.isdir(apath):
            num_leading_symbols = len(aname)
        else:
            num_leading_symbols = 0

        # load the store
        measurements = [Measurement.load(path, label, num_leading_symbols) for label in labels]
        measurement_dict = {}
        for label, measurement in zip(*[labels, measurements]):
            measurement_dict[label] = measurement.count_dict
        return measurement_dict



