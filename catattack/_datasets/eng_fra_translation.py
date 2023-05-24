import torchtext

print(torchtext.__file__)

import torchtext._internal

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import _create_dataset_directory
from torchtext.data.datasets_utils import _wrap_split_argument

URL = 'https://download.pytorch.org/tutorial/data.zip'
MD5 = 'fbb3849632b35bc5ecf9e3b033074f6e'
DATASET_NAME = 'EngFraTranslation'


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def EngFraTranslation(root, split):
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )


if __name__ == '__main__':
    datapipe = EngFraTranslation('.cache')
    print(datapipe)
