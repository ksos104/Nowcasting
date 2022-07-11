import os
import h5py # needs conda/pip install h5py
import matplotlib.pyplot as plt
import pdb
import utils
from tqdm import tqdm
import numpy as np
import pandas as pd

DATA_PATH = "/mnt/server14_hard0/seungju/dataset/SEVIR_ORIGIN"
CATALOG_PATH = os.path.join(DATA_PATH,"CATALOG.csv")
total_len = 0


catalog = pd.read_csv(CATALOG_PATH,parse_dates=['time_utc'],low_memory=False)
img_types = set(['vil'])

# Group by event id, and filter to only events that have all desired img_types
events = catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')



# file_list = [DATA_PATH + "/vil/2017/SEVIR_VIL_RANDOMEVENTS_2017_0501_0831.h5", DATA_PATH + "/vil/2017/SEVIR_VIL_RANDOMEVENTS_2017_0901_1231.h5", \
#              DATA_PATH + "/vil/2018/SEVIR_VIL_RANDOMEVENTS_2018_0101_0430.h5", DATA_PATH + "/vil/2018/SEVIR_VIL_RANDOMEVENTS_2018_0501_0831.h5", \
#              DATA_PATH + "/vil/2019/SEVIR_VIL_RANDOMEVENTS_2018_0901_1231.h5", DATA_PATH + "/vil/2019/SEVIR_VIL_RANDOMEVENTS_2019_0101_0430.h5", \
#               DATA_PATH + "/vil/2019/SEVIR_VIL_RANDOMEVENTS_2019_0501_0831.h5", DATA_PATH + "/vil/2019/SEVIR_VIL_RANDOMEVENTS_2019_0901_1231.h5" ]


file_list = [
                DATA_PATH + "/vil/2017/SEVIR_VIL_STORMEVENTS_2017_0101_0630.h5",  DATA_PATH + "/vil/2017/SEVIR_VIL_STORMEVENTS_2017_0701_1231.h5", \
                 DATA_PATH + "/vil/2018/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5",  DATA_PATH + "/vil/2018/SEVIR_VIL_STORMEVENTS_2018_0701_1231.h5", \
                  DATA_PATH + "/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5",  DATA_PATH + "/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5"
                ]

# test_file_list = [DATA_PATH + "/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5"]


train_path = "/mnt/server14_hard0/seungju/dataset/SEVIR/vil/STORM/train"
test_path =  "/mnt/server14_hard0/seungju/dataset/SEVIR/vil/STORM/test"

if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)

'''
Event Frames:  [-----------------------------------------------]
                [----13-----][---12----]
                            [----13----][----12----]
                                        [-----13----][----12----]
'''
train_idx = 0
test_idx = 0
for test_data in file_list:
    with h5py.File(test_data, mode='r') as hf:
        for idx,x in tqdm(enumerate(hf['vil'])):
         
            name = hf['id'][idx].decode('UTF-8')
            time  = str(catalog[catalog['id'] ==name]['time_utc'].values[0])[:16].replace('-','')
            date = time[:8]
            x1 = x[:,:,:25]
            x2 = x[:,:,12:37]
            x3 = x[:,:,-25:]
            if date < "20190601" :
                path  = train_path
                
                np.save(os.path.join(path, name + "_"  + time + "_1.npy"),x1)
                np.save(os.path.join(path, name + "_"  + time + "_2.npy"),x2)
                np.save(os.path.join(path, name + "_"  + time + "_3.npy"),x3)
                train_idx += 1
            else:
                path = test_path
                np.save(os.path.join(path, name + "_" + time + "_1.npy"),x1)
                np.save(os.path.join(path, name + "_" + time + "_2.npy"),x2)
                np.save(os.path.join(path, name + "_" + time + "_3.npy"),x3)
                test_idx += 1
