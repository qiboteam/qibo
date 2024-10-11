
from itertools import product
from functools import reduce
import pickle

import multiprocessing

import os
import numpy as np

import scipy.sparse as sparse
import time

# ------------------------------------------- #
#   below following MiFGD code                #
# ------------------------------------------- #

# Coordinates of non-zero entries in each of the X, Y, Z Pauli matrices...
ij_dict     = {'I' : [(0, 0), (1, 1)],
               'X' : [(0, 1), (1, 0)], 
               'Y' : [(0, 1), (1, 0)], 
               'Z' : [(0, 0), (1, 1)]}


# ... and the coresponding non-zero entries
values_dict = {'I' : [1.0, 1.0],
               'X' : [1.0, 1.0], 
               'Y' : [-1.j, 1.j], 
               'Z' : [1.0, -1.0]}

# X, Y, Z Pauli matrices
matrix_dict = {'I' : np.array([[1.0, 0.0],
                               [0.0, 1.0]]),
               'X' : np.array([[0.0, 1.0],
                               [1.0, 0.0]]),
               'Y' : np.array([[0.0, -1.j],
                               [1.j, 0.0]]),
               'Z' : np.array([[1.0, 0.0],
                               [0.0, -1.0]])}
# XXX Actually, from matrix_dict we can generate the above 


## ---------------------------------- ##
##    build projector labels          ##
## ---------------------------------- ##

def binarize(x):
    return int(''.join([str(y) for y in x]), 2)
    #return ''.join([str(y) for y in x])

def generate_random_label(n, symbols=['I', 'X', 'Y', 'Z']):
    num_symbols = len(symbols)
    label = ''.join([symbols[i] for i in np.random.randint(0, num_symbols, size=n)])
    return label


def generate_random_label_list(size,
                               n,
                               Rm_Id = 0,
                               symbols=['I', 'X', 'Y', 'Z'], factor=1.0, factor_step=0.1):

        factor      = 1.0
        factor_step = 0.1
        
        factor          = factor + factor_step
        effective_size  = int(size * factor)

        ## the essence of 'set' is that 'set does not contain repeated elements'
        labels = list(set([generate_random_label(n, symbols) for i in range(effective_size)]))

        while(len(labels) < size):
            #print(' ***  increasing length  ***')
            factor          = factor + factor_step
            effective_size  = int(size * factor)
            labels = list(set([generate_random_label(n, symbols) for i in range(effective_size)]))


        # ----------------------------------------------- #
        #   to remove Identity controlled by   Rm_Id      #
        # ----------------------------------------------- #
        if Rm_Id == 1:

            Iden = ''.join(['I' for i in range(n)])
        
            Go_Delete = 1
            while Go_Delete == 1:
                try:
                    labels.remove(Iden)
                except:
                    Go_Delete = 0

            # ------------------------------------- #
            #   to pad up labels up to num_label    #
            # ------------------------------------- #
            while (len(labels)<size):
                Add = [generate_random_label(n, symbols)]
                if Add != Iden:
                    labels = labels + Add

        # ----------------------------------------- #
        #   to keep just enough num_label  labels   #
        # ----------------------------------------- #
        labels = labels[:size]
        return labels


## ------------------------------------------ ##
##    constructing the matrix from labels     ##
## ------------------------------------------ ##

# Generate a projector by accumulating the Kronecker products of Pauli matrices
# XXX Used basically to make sure that our fast implementation is correct
def build_projector_naive(label, label_format='big_endian'):
    """ to directly generate a Pauli matrix from tensor products

    Args:
        label (str): label of the projection, e.g.  'XXZYZ'
        label_format (str, optional): the ordering of the label. Defaults to 'big_endian'.

    Raises:
        Exception: when the matrix size is too big (i.e. for qubit number > 6)

    Returns:
        ndarray: a matrix representing the Pauli operator 
    """
    if label_format == 'little_endian':
        label = label[::-1]
    if len(label) > 6:
        raise Exception('Too big matrix to generate!')
    projector = reduce(lambda acc, item: np.kron(acc, item), [matrix_dict[letter] for letter in label], [1])
    return projector


# Generate a projector by computing non-zero coordinates and their values in the matrix, aka the "fast" implementation
def build_projector_fast(label, label_format='big_endian'):
    """ to fastly generate a Pauli projection matrix in sparse matrix format

    Args:
        label (str): label of the projection, e.g.  'XXZYZ'
        label_format (str, optional): the ordering of the label. Defaults to 'big_endian'.

    Returns:
        sparse matrix: sparse matrix of the Pauli operator representing label
    """
    if label_format == 'little_endian':
        label = label[::-1]

    n = len(label)
    d = 2 ** n
    
    # map's result NOT subscriptable in py3, just tried map() -> list(map()) for py2 to py3
    ij     = [list(map(binarize, y)) for y in 
              [zip(*x) for x in product(*[ij_dict[letter] for letter in label])]] 
    values = [reduce(lambda z, w: z * w, y) for y in 
              [x for x in product(*[values_dict[letter] for letter in label])]]
    ijv    = list(map(lambda x: (x[0][0], x[0][1], x[1]), zip(ij, values))) 
    
    i_coords, j_coords, entries = zip(*ijv)

    projector  = sparse.coo_matrix((entries, (i_coords, j_coords)), 
                                                     shape=(d, d), dtype=complex)
    return projector
# XXX Here for redundancy and convenience: has also been added to a separate Projector class


# Choose implementation for the projector
def build_projector(label, label_format='big_endian', fast=True):
    if fast:
        return build_projector_fast(label, label_format)
    return build_projector_naive(label, label_format)





## --------------------------------- ##
##      only save the labels         ##
## --------------------------------- ##

# utilities for saving in Projector class
def _hdf5_saver(label, path, lock):
    lock.acquire()
    Projector(label).save(path)
    lock.release()

def _pickle_saver(label, path):
    Projector(label).save(path)

def _pickle_saveMap(label):
    """ to produce the Pauli operator object 

    Args:
        label (str): Pauli operator label

    Returns:
        object: Projector(label) = the class instance of a Pauli operator
    """
    return Projector(label)

## -------------------------------------------- #
##      [testing]   classes in projectors       #
## -------------------------------------------- #

# A projector as a class, hopefully with convenient methods :)
class Projector:
    """ to create a Pauli projector stored in an efficient way, 
    instead of writing in the form of matrix directly

    Returns:
        class instance: _description_
    """
    # Generate from a label or build from a dictionary represenation
    def __init__(self, arg, label_format='big_endian'):
        if isinstance(arg, str):
            self.label        = arg
            self.label_format = label_format
            self._generate()
        elif isinstance(arg, dict):
            data = arg
            self._build(data)    

    def _build(self, data):
        """ to build a sparse matrix according to the given data

        Args:
            data (dict): specification of the matrix for non-zero values
        """
        self.label        = data['label']
        self.label_format = data.get('label_format', 'big_endian')
        
        assert   data['num_columns'] == data['num_columns']
        entries  = data['values']
        i_coords = data['row_indices']
        j_coords = data['column_indices']
        d        = data['num_rows']
        dtype    = data['value_type']
        
        matrix  = sparse.coo_matrix((entries, (i_coords, j_coords)), 
                                    shape=(d, d), dtype=complex)
        self.matrix     = matrix
        self.csr_matrix = None


    # Here, injecting the "fast" implementation logic for our projector
    def _generate(self):
        if self.label_format == 'little_endian':
            label = self.label[::-1]
        else:
            label = self.label[:]
            
        n = len(label)
        d = 2 ** n
        
        # map's result NOT subscriptable in py3, just tried map() -> list(map()) for py2 to py3
        ij     = [list(map(binarize, y)) for y in 
                  [zip(*x) for x in product(*[ij_dict[letter] for letter in label])]]
        values = [reduce(lambda z, w: z * w, y) for y in 
                  [x for x in product(*[values_dict[letter] for letter in label])]]
        ijv    = list(map(lambda x: (x[0][0], x[0][1], x[1]), zip(ij, values)))
    
        i_coords, j_coords, entries = zip(*ijv)
        matrix  = sparse.coo_matrix((entries, (i_coords, j_coords)), 
                                    shape=(d, d), dtype=complex)
        self.matrix     = matrix
        self.csr_matrix = None

    # matvec'ing with a vector
    def dot(self, x):
        return self.csr().dot(x)

    # Get a sparse matrix representation of the projector in CSR format, 
    # i.e. ideal for matvec'ing it 
    def csr(self):
        if self.csr_matrix is None:
            self.csr_matrix = sparse.csr_matrix(self.matrix)

        return self.csr_matrix


    # Get a dict representation of the projector,
    # i.e. ideal for serializing it
    # XXX Currently with Python's pickle format in mind; moving to json format would add to portability
    def dict(self):
        """ to create a dict representation of the projector

        Returns:
            dict: data specifying non-zero values of the matrix
        """

        data = {
            'values'         : self.matrix.data, 
            'row_indices'    : self.matrix.row, 
            'column_indices' : self.matrix.col, 
            'num_rows'       : self.matrix.shape[0],
            'num_columns'    : self.matrix.shape[1],
            'value_type'     : self.matrix.dtype,
            'label'          : self.label
        }

        return data


    def _pickle_save(self, fpath):
        data = self.dict()
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)


    def _hdf5_save(self, fpath):
        f = h5py.File(fpath, 'a')
        group = f.create_group(self.label)

        data_dict = self.dict()
        for key in ['column_indices', 'row_indices', 'values']:
            dataset = group.create_dataset(key, data = data_dict[key])
        group.attrs['num_columns'] = data_dict['num_columns']
        group.attrs['num_rows']    = data_dict['num_rows']
        f.close()
        
            
    # Save the projector to disk
    def save(self, path):
        if os.path.isdir(path):
            fpath = os.path.join(path, '%s.pickle' % self.label)
            #print("in dir: ", fpath)
            self._pickle_save(fpath)
        elif path.endswith('.hdf5'):
            fpath = path
            print("hdf5: ", fpath)
            self._hdf5_save(fpath)

    @classmethod
    def _pickle_load(cls, fpath):
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        return data


    @classmethod
    def _hdf5_load(cls, fpath, label):

        f = h5py.File(fpath, 'r')
        group = f[label]
        
        data = {'label' : label }
        data['num_rows']    = group.attrs['num_rows']
        data['num_columns'] = group.attrs['num_columns']

        data['column_indices'] = group['column_indices'][:]
        data['row_indices']    = group['column_indices'][:]
        data['values']         = group['values'][:]
        data['value_type']     = data['values'].dtype
        return data


    # Load a projector from disk
    @classmethod
    def load(cls, path, label, num_leading_symbols=0):
        """ to load the data (dictionary representation of the Pauli projector)
        and then create s sparse matrix representation

        Args:
            path (str): path storing the Pauli projector
            label (str): the label of the Pauli projector 
            num_leading_symbols (int, optional): the effective number of alphabet in the label. Defaults to 0.

        Returns:
            class instance: the object representing the Pauli projector
        """

        if os.path.isdir(path):
            if num_leading_symbols == 0:
                fpath = os.path.join(path, '%s.pickle' % label)
                data  = cls._pickle_load(fpath)

            else:
                fragment_name = label[:num_leading_symbols]
                fpath         = os.path.join(path, fragment_name, '%s.pickle' % label)
                data          = cls._pickle_load(fpath)

        elif path.endswith('.hdf5'):
            fpath = path
            data  = cls._hdf5_load(fpath, label)

        projector = cls(data)
        return projector




class ProjectorStore:
    """ to deal with several different Pauli projectors at one time
    """
    def __init__(self, 
                 labels):
        self.labels = labels
        self.size   = len(labels)

    @classmethod
    def combine_proj_bulk(cls, proj_path, method_combine=1):
        """ to cobmine parallelly produced Pauli operator chunks

        Args:
            proj_path (str): the path to store the projectors
            method_combine (int, optional): method to do the combination. Defaults to 1.

        Returns:                    
            list: (label_sorted) list of sorted sampled Pauli operator labels
            dict: (Pj_combine) the dictionary for all produced Pauli operators
            int: (bulk_Pj) number of chunks storing different Pauli projectors
            int: (num_cpus) number of cpu for parallel computation
        """
        # --------------------------------------------- #
        #   loading projectors from each bulk           #
        # --------------------------------------------- #

        F_label_ALL = '{}/ALL_labels.pickle'.format(proj_path)
        with open(F_label_ALL, 'rb') as f:
            label_ALL = pickle.load(f)

        proj_lab_files = [xx for xx in os.listdir(proj_path) if xx.startswith("labels_")]

        bulk_Pj = len(proj_lab_files)
    
        if method_combine == 0:
            print('   ----------   direct combining Pj_list   (NOT parallel)  --------------- \n')

            label_combine = []
            Pj_combine    = {}
            for ii, lab_file in enumerate(proj_lab_files):
                td1 = time.time()

                ID_label   = int(lab_file.split('.')[0][7:])
                Fname_lab  = '{}/{}'.format(proj_path, lab_file)
                Fname_proj = '{}/Pj_list_{}.pickle'.format(proj_path, ID_label)


                ID_Pj, label_Pj, Pj_list = cls.Load_Pj_part(ID_label, Fname_lab, Fname_proj)

                label_combine += label_Pj
                Pj_combine.update(Pj_list)

                td2 = time.time()
                print('  ---------------  done of updating {}-th Pj from {}'.format(ii, Fname_proj))
                print('            --->   Time = {}    for {}-th Pj\n'.format(td2-td1, ii))
                del Pj_list
                del label_Pj

            num_cpus = 1

        elif method_combine == 1:

            num_cpus = multiprocessing.cpu_count()
            if bulk_Pj < num_cpus:
                num_cpus = bulk_Pj
            print('      ----------   parallelel #CPU = {}  for combining Pj_list -----\n'.format(num_cpus))

            pool = multiprocessing.Pool(num_cpus)
    
            res_list = []
            for lab_file in proj_lab_files:
                ID_label   = int(lab_file.split('.')[0][7:])
                Fname_lab  = '{}/{}'.format(proj_path, lab_file)
                Fname_proj = '{}/Pj_list_{}.pickle'.format(proj_path, ID_label)

                print('      ******  apply_async for ID = {}   *****'.format(ID_label))                
                res = pool.apply_async(cls.Load_Pj_part_wrap, ([ID_label, Fname_lab, Fname_proj],))
                res_list.append(res)

            pool.close()
            pool.join()

            print('  *******  After parallel loading Pj  --> len(res_list) = {}  ******* \n'.format(len(res_list)))
            if len(res_list) != bulk_Pj:
                print('   ERROR for collecting parallel cpus results \n')
                return
            
            ID_order      = []
            label_combine = []
            Pj_combine    = {}

            pop_method = 1

            for ii in range(bulk_Pj):

                if pop_method == 0:
                    res = res_list[ii]
                elif pop_method == 1:
                    res = res_list.pop(0)

                ID_Pj, label_Pj, Pj_list = res.get()

                print('    --------  ID = {} is popped from res_list  ---------- '.format(ID_Pj))

                ID_order.append(ID_Pj)
                label_combine += label_Pj
                Pj_combine.update(Pj_list)
            print('    ########    After popping -->  len(res_list) = {}   ######\n'.format(len(res_list)))


        if len(Pj_combine) != len(label_ALL):
            print('  ERROR:  len(Pj_combine) != len(label_ALL) \n') 
            return

        label_sorted = sorted(label_combine)
        if not label_sorted == label_ALL:
            print('  ERROR:  label_combine NOT equal label_ALL \n')
            return
        
        if method_combine == 1:
            print('      ID order of returning from parallel CPU  = {}\n'.format(ID_order))
        print('       ----------     Total # bulk Pj file = {}   ---------------'.format(bulk_Pj))
        print('       ----------     ALL Pj_list are combined   --------------- \n')

        return label_sorted, Pj_combine, bulk_Pj, num_cpus

    @classmethod
    def Load_Pj_part_wrap(cls, argv):
        """ wrapper of the function Load_Pj_part

        Args:
            argv (tuple): arguments of the function Load_Pj_part

        Returns:
            int: (ID_label) the ID number of this Pauli operator chunk
            list: (label_Pj) the stored labels
            dir: (Pj_list) the stored Pauli prjectors 
        """

        ID_label, label_Pj, Pj_list = cls.Load_Pj_part(*argv)

        return ID_label, label_Pj, Pj_list


    @classmethod
    def Load_Pj_part(cls, ID_label, Fname_lab, Fname_proj):
        """ to load labels and the corresponding Pauli operators for this chunk
            specified by the ID_label

        Args:
            ID_label (int): ID number of this Pauli operator chunk
            Fname_lab (str): file name for storing labels
            Fname_proj (str): file name for storing the Projectors 

        Returns:
            int: (ID_label) the ID number of this Pauli operator chunk
            list: (label_Pj) the stored labels
            dir: (Pj_list) the stored Pauli prjectors 
        """

        print('         Start to load Fname_lab = {}   -->  ID_label = {}'.format(Fname_lab, ID_label))
        with open(Fname_lab, 'rb') as f:
            label_Pj = pickle.load(f)

        with open(Fname_proj, 'rb') as f:
            Pj_list = pickle.load(f)
        print('         -->  ID = {},  Proj_part File = {}   is loaded\n'.format(ID_label, Fname_proj))

        return ID_label, label_Pj, Pj_list

    # Generate and save the projectors and do it in parallel, 
    # i.e. using all available cores in your system
    def mpPool_map(self, path, bulksize=6000):
        format = 'hdf5'
        if not path.endswith('.hdf5'):
            format = 'pickle'
            if not os.path.exists(path):
                os.mkdir(path)
                
        num_cpus   = multiprocessing.cpu_count()
        #num_cpus2  = num_cpus - 2
        if num_cpus > len(self.labels):
            num_cpus = 3
        #print(' num_cpus = {},  num_cpus2 = {}, self.size ={}'.format(num_cpus, num_cpus2, self.size))

        pool = multiprocessing.Pool(num_cpus)
        #pool = multiprocessing.Pool()

        # --------------------------------------- #
        #       determine & use label_part        #
        # --------------------------------------- #
        if self.size > bulksize:
            method = 0      #  = 0  for label_part

            numB = int(np.ceil(self.size/bulksize))

            label_part = [self.labels[ii*bulksize: (ii+1)*bulksize] \
                                for ii in range(numB)]

            Partition_Pj = 1
            print('  nB ={}, len(label_part) = {}'.format(numB, len(label_part)))            
            print('      -->  Partion_Pj = {}'.format(Partition_Pj))
        else:
            method = 1
            Partition_Pj = 0
            print('      Partion_Pj = {}  -->  No partion in Pj'.format(Partition_Pj))

        # ----------------------------------------- #
        #   use label_part sequentially             #
        #       each part is using parallel  CPUs   #
        # ----------------------------------------- #        

        saveP = 1                        #  the default size of saved bulk Pj_list
        if method == 0:
            mp_num_max = bulksize*5


            mp_method = 'map (bulksize sequentially)'
            
            res = []
            labels_collect = []
            NowSize = 0
            saveP = 1
            for ii, labels in enumerate(label_part):
                res_part = pool.map(_pickle_saveMap, labels)
                res            += res_part
                labels_collect += labels

                NowSize += len(labels)
                print('  ***  Proj {}-th bulk:  # {} projectors were created'.format(ii, NowSize))

                if int(NowSize/mp_num_max) >= saveP:
                    pool.close()
                    pool.join()

                    self.Save_label_Pj(res, labels_collect, path, saveP)

                    print('  ***  {}-th labels_part  ->  saved bulk size = {}, NowSize = {}'.format(ii, len(labels_collect), NowSize))
                    print('  #########       restart pool      #########\n')

                    saveP += 1
                    res            = []
                    labels_collect = []
                    pool = multiprocessing.Pool(num_cpus)
            
            print('   len(the Final labels_collect) = {}\n'.format(len(labels_collect)))
            if len(labels_collect) != 0:
                self.Save_label_Pj(res, labels_collect, path, saveP)
            else:
                saveP = saveP - 1
            print('\n     saved # bulk = {}, NowSize = {}\n'.format(saveP, NowSize))

            if NowSize != len(self.labels):
                print(' the pool map NOT completed yet!\n')
                return

            label_ALL_file = '{}/ALL_labels.pickle'.format(path)
            with open(label_ALL_file, 'wb') as f:
                pickle.dump(sorted(self.labels), f)


        # ----------------------------------------- #
        #    the whole  label_list = self.labells   #
        #    -->   the default size of saveP = 1    #
        # ----------------------------------------- #
        if method == 1:
            mp_method = 'map'
            res = pool.map(_pickle_saveMap, self.labels)

        elif method == 2:           
            mp_method = 'map_async'
            output = pool.map_async(_pickle_saveMap, self.labels)
            res = output.get()

        elif method == 3:
            mp_method = 'apply_async'

            res = []
            for ii in range(self.size):
                out = pool.apply_async(_pickle_saveMap, (self.labels[ii], ))
                res.append(out.get())
            
        pool.close()
        pool.join()

        # --------------------------------- #
        #   saving calculated projectors    #
        # --------------------------------- #
        if method > 0:
            if len(res) != len(self.labels):
                print(' the pool map NOT completed yet!\n')
                return
            self.Save_label_Pj(res, self.labels, path)

        print('     pool.{}  COMPLETED by #CPU = {} \n'.format(mp_method, num_cpus))
        print('     [projectors] num_cpus = {},  saved bulk Pj = {} \n'.format(num_cpus, saveP))

        return num_cpus, saveP, Partition_Pj

    @classmethod
    def Save_label_Pj_dict(cls, Pj_list, labels, path, Name=None):
        """ to save the given Pauli operators and labels in path

        Args:
            Pj_list (dict): dictionary referring to all Pauli operators
            labels (list): list of Pauli operator labels
            path (str): the path storing the Pauli operators 
            Name (str, optional): suffix for the file name. Defaults to None.
        """

        # ------------------------- #
        #   saving projectors       #
        # ------------------------- #

        if Name == None:
            Pj_file = '{}/Pj_list.pickle'.format(path)
        else:
            Pj_file = '{}/Pj_list_{}.pickle'.format(path, Name)

        with open(Pj_file, 'wb') as f:
            pickle.dump(Pj_list, f)
        print('\n  ***  Pj_file    = {}  is saved (i.e. dump)'.format(Pj_file))

        # ------------------------- #
        #   saving sorted labels    #
        # ------------------------- #
        s_label = sorted(labels)

        if Name == None:
            label_file = '{}/labels.pickle'.format(path)
        else:
            label_file = '{}/labels_{}.pickle'.format(path, Name)

        with open(label_file, 'wb') as f:
            pickle.dump(s_label, f)
        print('  ***  label_file = {}   is saved\n'.format(label_file))



    def Save_label_Pj(self, projectors, labels, path, Name=None):
        """ to save generated Pauli operators

        Args:
            projectors (class ProjectorStore): the object representing the projectors
            labels (list): list of sampled Pauli operator labels
            path (str): the path directory for storing the Pauli operators 
            Name (str, optional): File suffix. Defaults to None.
        """

        Pj_list = {}
        for Pj in projectors:
            Pj_list[Pj.label] = Pj

        self.Save_label_Pj_dict(Pj_list, labels, path, Name)



    # Load projectors previously saved under a disk folder
    @classmethod
    def load_PoolMap_bulk_list(cls, path):
        """ check if the projector directory exists

        Args:
            path (str): path to store the projectors
        """

        label_ALL_file = '{}/ALL_labels.pickle'.format(path)
        if os.path.exists(label_ALL_file):
            print('{} exists'.format(label_ALL_file))
        else:
            print(' Projectors ALL in one file \n')


    @classmethod
    def load_labels_from_file(cls, path):
        f_label = '{}/labels.pickle'.format(path)
        with open(f_label, 'rb') as f:
            labels = pickle.load(f)
        print('   --->  loading labels DONE from file = {}'.format(f_label))

        return labels
                
    # Load projectors previously saved under a disk folder
    @classmethod
    def load_PoolMap(cls, path, labels=None):
        """ to load the stored Pauli projectors

        Args:
            path (str): the path to the projectors
            labels (list, optional): list of Pauli operator labels. Defaults to None.

        Returns:
            dict: (projector_dict) dictionary with values referring to the projectors
        """

        if labels == None:
            labels = cls.load_labels_from_file(path)

        fname = '{}/Pj_list.pickle'.format(path)
        with open(fname, 'rb') as f:
            Pj_list = pickle.load(f)

        projector_dict = {}
        for label in labels:
            projector_dict[label] = Pj_list[label]

        return projector_dict


    # ------------------------------------------------- #
    #   below the original method saving each projector #
    # ------------------------------------------------- #
    # Generate and save the projectors and do it in parallel, .
    # i.e. using all available cores in your system
    def populate(self, path):
        format = 'hdf5'
        if not path.endswith('.hdf5'):
            format = 'pickle'
            if not os.path.exists(path):
                os.mkdir(path)
                
        num_cpus   = multiprocessing.cpu_count()
        num_cpus2  = num_cpus - 2
        num_rounds = 1
        if self.size > num_cpus2:
            num_rounds = self.size // num_cpus2 + 1 

        # XXX lock defies parallelization in generattion for hdf5
        # XXX Consider using MPI-based scheme
        if format == 'hdf5':
            lock = multiprocessing.Lock()

        print('  -- num_cpus = {}, num_cpus2= {}, size = {} -- '.format(num_cpus,num_cpus2, self.size))
        print('  *******  num_rounds = {}   *******'.format(num_rounds))

        for r in range(num_rounds):
            process_list = []
            for t in range(num_cpus2):
                idx = r * num_cpus2 + t
                if idx == self.size:
                    break
                label = self.labels[idx]

                print('r = {}, t = {}, idx = {}, label = {}'.format(r, t, idx, label))

                if format == 'pickle':
                    process = multiprocessing.Process(target=_pickle_saver,
                                                      args=(label, path))
                elif format == 'hdf5':
                    process = multiprocessing.Process(target=_hdf5_saver,
                                                     args = (label, path, lock))
                                
                process.start()
                process_list.append(process)

            # moving join() inside the outer loop to avoid "too many files open" error    
            for p in process_list:
                p.join()

    @classmethod
    def load_labels(cls, path, Nk=None):
        if path.endswith('.hdf5'):
            with h5py.File(path, 'r') as f:
                labels = f.keys()
        else:
            try:
                label_load1 = [fname.split('.')[0] for fname in os.listdir(path)]

                label_load2 = [label for label in label_load1 if not label.startswith('Pj_list')]    
                label_load3 = [label for label in label_load2 if not label.startswith('labels')]    
                labels      = [label for label in label_load3 if not label.startswith('ALL_')]    

                if Nk != None:
                    labels  = [label for label in labels if len(label)== Nk]    

            except:
                fragment_paths = [os.path.join(path, fragment) for fragment in os.listdir(path)]
                labels = []
                for fragment_path in fragment_paths:
                    fragment_labels = [fname.split('.')[0] for fname in os.listdir(fragment_path)]
                    labels.extend(fragment_labels)
        return labels
       
    # Load projectors previously saved under a disk folder
    @classmethod
    def load(cls, path, labels=None):
        """ to load all the stored Pauli projectors from a given path

        Args:
            path (str): the path to the projectors
            labels (list, optional): list of Pauli operator labels. Defaults to None.

        Returns:
            dict: (projector_dict) the dictionary with key of Pauli operator labels
                and value of Pauli operators themselves
        """

        if labels == None:
            labels = cls.load_labels(path)

        # checking if the store is fragmented and compute num_leading_symbols
        names = os.listdir(path)
        aname = names[0]
        apath = os.path.join(path, aname)
        if os.path.isdir(apath):           # is a directory or not (not checking file)
            num_leading_symbols = len(aname)
        else:
            num_leading_symbols = 0

        # load the store
        projectors = [Projector.load(path, label, num_leading_symbols) for label in labels]
        projector_dict = {}
        for label, projector in zip(*[labels, projectors]):
            projector_dict[label] = projector
        return projector_dict



