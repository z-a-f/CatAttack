import os
from functools import partial
from pathlib import Path

import torchdata
from torchtext.data.datasets_utils import _create_dataset_directory

URL = 'https://download.pytorch.org/tutorial/data.zip'
MD5 = 'fbb3849632b35bc5ecf9e3b033074f6e'
DATASET_NAME = 'EngFra'

def _path_fn(root, url):
    return os.path.join(root, os.path.basename(url))

def _filename_filter_fn(includes, excludes, filename):
    filename = filename[0].rsplit(os.sep, 1)[1]
    if includes is not None and filename not in includes:
        return False
    if excludes is not None and filename in excludes:
        return False
    return True

def _split_line(t):
    return t.split('\t')

@_create_dataset_directory(dataset_name=DATASET_NAME)
def EngFra(root):
    pipeline = torchdata.datapipes.iter.IterableWrapper([URL]) \
                .on_disk_cache(
                    filepath_fn=partial(_path_fn, root),
                    hash_dict={_path_fn(root, URL): MD5},
                    hash_type='md5') \
                .read_from_http() \
                .end_caching(mode="wb", same_filepath_fn=True) \
                .open_files(mode='rb') \
                .load_from_zip() \
                .filter(partial(_filename_filter_fn,
                                ['eng-fra.txt'],
                                None)) \
                .readlines(decode=True,
                           encoding='utf-8',
                           return_path=False,
                           strip_newline=True) \
                .map(_split_line)
    return pipeline
