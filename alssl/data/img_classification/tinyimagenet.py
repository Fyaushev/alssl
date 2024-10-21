import os
import sys
from pathlib import Path

import torchvision
from torchvision import transforms

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

import shutil

from torch.utils.data import DataLoader


# Code from
# https://colab.research.google.com/github/yandexdataschool/mlhep2019/blob/master/notebooks/day-3/seminar_convnets.ipynb#scrollTo=tvz-gycUrYD1
# downloaded as !wget https://raw.githubusercontent.com/yandexdataschool/Practical_DL/spring2019/week03_convnets/tiny_img.py -O tiny_img.py
def download_tinyImg200(path,
                     url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                     tarname='tiny-imagenet-200.zip'):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        return
    urlretrieve(url, os.path.join(path,tarname))
    print(os.path.join(path,tarname))
    import zipfile
    zip_ref = zipfile.ZipFile(os.path.join(path,tarname), 'r')
    zip_ref.extractall()
    zip_ref.close()

def sort_val_images(data_path: Path):
    if (data_path / "val_sorted").exists():
        return
    (data_path / "val_sorted").mkdir(exist_ok=True)
    with open(data_path / "val" / "val_annotations.txt") as f:
        for line in f:
            img_name, cls, _, _, _, _ = line.split('\t')
            (data_path / "val_sorted" / cls / "images").mkdir(exist_ok=True, parents=True)
            shutil.copyfile(data_path / "val" / "images" / img_name, data_path / "val_sorted" / cls / "images" / img_name)
    

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomRotation(10),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

def get_num_classes():
    return 200

def get_dataset(subset="train", data_path=Path("/shared/projects/active_learning/tinyimagenet")):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    download_tinyImg200(data_path)
    sort_val_images(data_path)
    if subset == "train":
        data_path = data_path / "train"
    else:
        data_path = data_path / "val_sorted"
    return torchvision.datasets.ImageFolder(
        root=data_path, transform=transform
    )