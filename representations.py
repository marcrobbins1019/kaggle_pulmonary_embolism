# class FullNormalizedSeriesMemoizer(input_cache_dir,output_cache_dir,normalize):
import os
import pickle

import numpy as np
import pandas as pd
import pydicom

SLICE_LEVEL_DICOMS = ["Pixel Data", "InstanceNumber", "SOPInstanceUID"]


def save_by_type(object, file_path):
    if isinstance(object, np.array):
        np.save(file_path, object)
    else:
        pickle.dump(object, open(file_path), 'rb')


def load_by_type(file_path):
    extension = os.path.splitext(file_path)[1]
    if extension == ".npy":
        obj = np.load(file_path)
    elif extension == ".pkl":
        obj = pickle.load(open(file_path), 'rb')
    else:
        raise Exception("Unrecognized File Extension")

    return obj


# def with_caching(func):
#     def process_cache(*args, **kwargs):
#         if len(kwargs['read_cache_dir']) > 0:
#             os.makedirs(kwargs['read_cache_dir'],exist_ok=True)
#             try:
#                 object = load_by_type(kwargs["read_cache_dir"])
#             except:
#                 object = func(*args, **kwargs)
#
#         if len(kwargs['write_cache_dir']) > 0:
#             os.makedirs(kwargs['read_cache_dir'],exist_ok=True)
#             save_by_type(object , kwargs["write_cache_dir"])
#
#     return process_cache

def with_caching(func, study_id, read_cache_dir, write_cache_dir):
    if len(read_cache_dir) > 0:
        os.makedirs(read_cache_dir, exist_ok=True)
        try:
            object = load_by_type(kwargs["read_cache_dir"])
        except:
            object = func(*args, **kwargs)

    if len(kwargs['write_cache_dir']) > 0:
        os.makedirs(kwargs['read_cache_dir'], exist_ok=True)
        save_by_type(object, kwargs["write_cache_dir"])


return process_cache


def combine_full_series(study_df, path_to_series):
    dcms = [pydicom.dcmread(os.path.join(path_to_series, dcm_id + '.dcm')) for dcm_id in
            study_df['SOPInstanceUID']]
    instance_number_array = [(int(dcm['InstanceNumber'].value), dcm) for dcm in dcms]
    study_df['InstanceNumber'] = [x[0] for x in instance_number_array]
    study_df.sort_values(by='InstanceNumber', inplace=True)

    output_dic = study_df.iloc[0].to_dict()
    output_dic['InstanceNumber'] = study_df['InstanceNumber'].values
    output_dic['SOPInstanceUID'] = study_df['SOPInstanceUID'].values
    output_dic['pe_present_on_image'] = study_df['pe_present_on_image'].values

    sorted_array = [x[1] for x in sorted(instance_number_array)]
    for dcm in sorted_array:
        dcm.decode()

    representative_dcm = sorted_array[0]
    dcm_dic = {elem.description(): elem.value for elem in representative_dcm.elements() if
               elem.description() not in SLICE_LEVEL_DICOMS}
    volume_3d = np.array([dcm.pixel_array for dcm in sorted_array])

    output_dic.update(dcm_dic)
    output_dic['volume_3d'] = volume_3d
    return output_dic


class FullSeriesPickleMemoizer():
    def __init__(self, train_csv_path,
                 name='standard',
                 read_cache_dir="",
                 write_cache_dir=""):
        self.name = name
        self.read_cache = len(read_cache_dir) > 0
        self.write_cache = len(write_cache_dir) > 0
        self.read_cache_dir = read_cache_dir
        self.write_cache_dir = write_cache_dir
        self.train_df = pd.read_csv(train_csv_path)

    # @with_caching
    def get_series(self, study_id):
        # move these top lines?
        study_df = self.train_df[self.train_df['StudyInstanceUID'] == study_id]
        series_id = study_df.iloc[0]['SeriesInstanceUID']
        path_to_series = os.path.join(self.read_cache_dir, study_id, series_id)
        series_3d_full = combine_full_series(study_df, path_to_series)
        return series_3d_full

    def get_series_with_caching(self, study_id):
        return with_caching(func=self.get_series,
                            study_id=study_id,
                            read_cache_dir=self.read_cache_dir,
                            write_cache_dir=self.write_cache_dir)


PATH_TO_TRAIN_CSV = '/algo/users/marc/kaggle_pe/train.csv'
PATH_TO_DCM_DIR = '/algo/users/marc/kaggle_pe/data/train'
PATH_TO_FULL_SERIES_CACHE = '/algo/users/marc/kaggle_pe/data/full_series_cache'

full_memoizer = FullSeriesPickleMemoizer(train_csv_path=PATH_TO_TRAIN_CSV,
                                         read_cache_dir=PATH_TO_DCM_DIR,
                                         write_cache_dir=PATH_TO_FULL_SERIES_CACHE)

full_memoizer.get_series(study_id='5a7be944da6d')

# maybe just put this in the second rep_fetcher (why combine these?)
# def get_representation(window=(400,40),
#                        dimension = (40,256,256),
#                        normalize_mm = False):
#     return "test"


# see if thing is cached first (special function for this?)
