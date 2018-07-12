"""
Collects all h5 files from MillionSongSubset folder and places them into h5_files.pkl
"""

import os
import pandas as pd
import time


def get_all_files(rootdir, maxamount=float('inf')):
    rootdir = os.path.normcase(rootdir)
    file_paths = []
    for dirpath, dirs, files in os.walk(rootdir):
        for filename in files:
            if len(file_paths) >= maxamount:
                print('short')
                return file_paths
            elif filename[0] != '.':
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths

start_time = time.time()

h5_files = get_all_files('./MillionSongSubset/data')
print('Number of files:', len(h5_files))

pd.to_pickle(h5_files, './pkl/h5_files.pkl')

print("--- %s seconds ---" % (time.time() - start_time))
