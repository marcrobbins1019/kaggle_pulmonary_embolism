import pandas as pd
from pathlib import Path
import os
import pydicom
from tqdm import tqdm

PATH_TO_TRAIN_CSV = '/algo/users/marc/kaggle_pe/train.csv'
PATH_TO_DCM_DIR = '/algo/users/marc/kaggle_pe/data/train'
PATH_TO_MERGED_CSV = '/algo/users/marc/kaggle_pe/merged_train.csv'

def get_dcm_metadata_slice_level(path_to_dcm_file):
    rep_dcm = pydicom.dcmread(path_to_dcm_file)
    rep_dcm.decode()
    dcm_dic = {elem.description():elem.value for elem in rep_dcm.elements() if elem.description()!='Pixel Data'}
    return dcm_dic

# def build_merged_df_slice_level(path_to_train_csv,path_to_dcm_dir):
#     df = pd.read_csv(path_to_train_csv)
#     for index, row in tqdm(df.iterrows()):
#         path_to_dcm_dir =
#         dcm_dic = get_dcm_metadata(study_id=index,path_to_dcm_dir=path_to_dcm_dir)
#         for key, value in dcm_dic.items():
#             try:
#                 study_level_df.loc[index,key] = value
#             except:
#                 study_level_df.loc[index, key] = str(value)
#
#     return study_level_df




def get_dcm_metadata(study_id, path_to_dcm_dir):
    dcm_dir = os.path.join(path_to_dcm_dir, study_id)
    dcm_paths = []
    for dcm_path in Path(dcm_dir).rglob('*.dcm'):
        dcm_paths.append(dcm_path)
    rep_dcm = pydicom.dcmread(str(dcm_paths[0]))
    rep_dcm.decode()
    dcm_dic = {elem.description():elem.value for elem in rep_dcm.elements() if elem.description()!='Pixel Data'}
    dcm_dic['number_slices'] = len(dcm_paths)
    return dcm_dic

def build_merged_df(path_to_train_csv,path_to_dcm_dir):
    df = pd.read_csv(path_to_train_csv)
    study_level_df = df.sort_values(by='pe_present_on_image', ascending=False)
    study_level_df = study_level_df.drop_duplicates(subset=['StudyInstanceUID'])
    study_level_df.set_index('StudyInstanceUID',inplace=True)
    for index, row in tqdm(study_level_df.iterrows()):
        dcm_dic = get_dcm_metadata(study_id=index,path_to_dcm_dir=path_to_dcm_dir)
        for key, value in dcm_dic.items():
            try:
                study_level_df.loc[index,key] = value
            except:
                study_level_df.loc[index, key] = str(value)

    return study_level_df

def get_ordered_dicoms(df,study_id, path_to_dcm_dir):
    study_df = df[df['StudyInstanceUID']==study_id]
    series_id = study_df.iloc[0]['SeriesInstanceUID']
    dcm_base_path = os.path.join(path_to_dcm_dir, study_id, series_id)
    dcms = [pydicom.dcmread(os.path.join(dcm_base_path, dcm_id + '.dcm')) for dcm_id in
            study_df['SOPInstanceUID']]
    instance_number_array = [(int(dcm['InstanceNumber'].value), dcm) for dcm in dcms]
    sorted_array = [x[1] for x in sorted(instance_number_array)]
    return sorted_array

# merged_df = build_merged_df(path_to_train_csv=PATH_TO_TRAIN_CSV,
#                             path_to_dcm_dir=PATH_TO_DCM_DIR)
#
# merged_df.to_csv(PATH_TO_MERGED_CSV)









