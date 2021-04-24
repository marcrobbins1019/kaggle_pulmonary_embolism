import pydicom
import numpy as np
import time
import pickle

arr = np.load('/algo/users/marc/kaggle_pe/scratch/00ac73cfc372.npy')
pickle.dump(arr,open('/algo/users/marc/kaggle_pe/scratch/00ac73cfc372.pkl','wb'))


def timing_decorator(n_repeats):
    def decorator(func):
        def print_time_elapsed(*args, **kwargs):
            start_time = time.time()
            output = ""
            for i in range(n_repeats):
                output = func(*args,**kwargs)
            end_time = time.time()
            elapsed_time = np.round((end_time - start_time),decimals=5)
            print(" {} time elapsed for {} runs: {}".format(func.__name__, n_repeats, elapsed_time))
            return output
        return print_time_elapsed
    return decorator

@timing_decorator(n_repeats=100)
def load_dcm(file_path):
    return pydicom.dcmread(file_path).pixel_array

@timing_decorator(n_repeats=100)
def load_npy(file_path):
    return np.load(file_path)

@timing_decorator(n_repeats=100)
def load_pkl(file_path):
    return pickle.load(open(file_path,'rb'))

load_dcm(file_path='/algo/users/marc/kaggle_pe/scratch/00ac73cfc372.dcm')
load_npy(file_path='/algo/users/marc/kaggle_pe/scratch/00ac73cfc372.npy')
load_pkl(file_path='/algo/users/marc/kaggle_pe/scratch/00ac73cfc372.pkl')



