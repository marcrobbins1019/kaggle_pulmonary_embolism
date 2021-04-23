import pydicom
import numpy as np
import time

def timing_decorator(func):
    def print_time_elapsed(*args,**kwargs):
        start_time = time.time()
        output = func(*args,**kwargs)
        end_time = time.time()
        elapsed_time = np.round((end_time - start_time),decimals=5)
        print(" {} time elapsed: {}".format(func.__name__, elapsed_time))
        return output

    return print_time_elapsed

@timing_decorator
def load_dcm(file_path):
    return pydicom.dcmread(file_path).pixel_array

@timing_decorator
def load_npy(file_path):
    return np.load(file_path)

load_dcm(file_path='/algo/users/marc/kaggle_pe/scratch/00ac73cfc372.dcm')
load_npy(file_path='/algo/users/marc/kaggle_pe/scratch/00ac73cfc372.npy')


