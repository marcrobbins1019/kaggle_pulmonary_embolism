import os
import pickle

import numpy as np
import pandas as pd
import pydicom
from image_preprocessing import get_representation

SLICE_LEVEL_DICOMS = ["Pixel Data", "InstanceNumber", "SOPInstanceUID"]


def save_by_type(object, file_path):
    if isinstance(object, np.ndarray):
        np.save(file_path, object)
    else:
        file_path += '.pkl'
        pickle.dump(object, open(file_path,'wb'))


def load_by_type(file_path):
    extension = os.path.splitext(file_path)[1]
    if extension == ".npy":
        obj = np.load(file_path)
    elif extension == ".pkl":
        obj = pickle.load(open(file_path,'rb'))
    else:
        raise Exception("Unrecognized File Extension")

    return obj


def with_caching(func):
    def process_cache(*args, **kwargs):
        if args[0].read_cache:
            try:
                object = load_by_type(os.path.join(args[0].named_cache_dir,kwargs['study_id']))
            except:
                object = func(*args, **kwargs)
                if args[0].write_cache:
                    save_by_type(object, os.path.join(args[0].named_cache_dir, kwargs['study_id']))

        return object

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
                 dcm_dir=None,
                 cache_dir=None,
                 read_cache = True,
                 write_cache = True,
                 name='standard',):
        self.name = name
        self.dcm_dir = dcm_dir
        self.read_cache = read_cache
        self.write_cache = write_cache
        self.train_df = pd.read_csv(train_csv_path)

        self.cache_dir = cache_dir
        if cache_dir:
            self.named_cache_dir = os.path.join(cache_dir, name)
            os.makedirs(self.named_cache_dir, exist_ok=True)


    @with_caching
    def get_study(self, study_id):
        study_df = self.train_df[self.train_df['StudyInstanceUID'] == study_id]
        series_id = study_df.iloc[0]['SeriesInstanceUID']
        path_to_series = os.path.join(self.dcm_dir, study_id, series_id)
        series_3d_full = combine_full_series(study_df, path_to_series)
        return series_3d_full

class FinalRepPickleMemoizer():
    def __init__(self, rep_name = 'final_rep',
                 window = (400,40),
                 dimension = (40,128,128),
                 normalize_mm=False,
                 *args,
                 **kwargs):

        self.full_memoizer = FinalRepPickleMemoizer(*args, **kwargs)
        self.name = rep_name
        self.window = window
        self.dimension = dimension
        self.normalize_mm = normalize_mm

        self.kwargs = kwargs

        if kwargs['cache_dir']:
            self.named_cache_dir = os.path.join(kwargs['cache_dir'], rep_name)
            os.makedirs(self.named_cache_dir, exist_ok=True)

    @with_caching
    def get_study(self, study_id):
        full_series = self.full_memoizer.get_study(study_id=study_id)

        final_series = get_representation(full_series,
                                          self.window,
                                          self.dimension,
                                          self.normalize_mm)
        return final_series




PATH_TO_TRAIN_CSV = '/algo/users/marc/kaggle_pe/train.csv'
PATH_TO_DCM_DIR = '/algo/users/marc/kaggle_pe/data/train'
PATH_TO_FULL_SERIES_CACHE = '/algo/users/marc/kaggle_pe/data/full_series_cache'

full_memoizer = FullSeriesPickleMemoizer(train_csv_path=PATH_TO_TRAIN_CSV,
                                         dcm_dir=PATH_TO_DCM_DIR,
                                         cache_dir=PATH_TO_FULL_SERIES_CACHE,
                                         read_cache=True,
                                         write_cache=True
                                         )

full_memoizer.get_study(study_id='5a7be944da6d')

final_memoizer = FinalRepPickleMemoizer(train_csv_path=PATH_TO_TRAIN_CSV,
                                         dcm_dir=PATH_TO_DCM_DIR,
                                         cache_dir=PATH_TO_FULL_SERIES_CACHE,
                                         read_cache=True,
                                         write_cache=True
                                         )

out_rep = final_memoizer.get_study(study_id='5a7be944da6d')
pickle.dump(out_rep,open('/tmp/outtest.pkl','wb'))



# maybe just put this in the second rep_fetcher (why combine these?)
# def get_representation(window=(400,40),
#                        dimension = (40,256,256),
#                        normalize_mm = False):
#     return "test"


# see if thing is cached first (special function for this?)
