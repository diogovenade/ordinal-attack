# The same datasets as https://arxiv.org/abs/1911.10720
# Breast cancer grading; Photographs dating;  Age estimation
# A list of ordinal datasets: https://github.com/tomealbuquerque/ordinal-datasets

from skimage.io import imread
import torch
import torchvision
import numpy as np
import pandas as pd
import json
import os

def split(ds, split, seed):
    # FIXME: we could use stratified sampling
    # sklearn.model_selection.train_test_split(range(len(ds)), train_size=0.2, random_state=42, statify=labels)

    # random holdout: train-val-test 60-20-20
    rng = np.random.default_rng(123+seed)
    ix = rng.choice(len(ds), len(ds), False)
    s = ['train', 'val', 'test'].index(split)
    i, j = [(0, 0.6), (0.6, 0.8), (0.8, 1)][s]
    ix = ix[int(i*len(ds)):int(j*len(ds))]
    return torch.utils.data.Subset(ds, ix)

class GroupLabels(torch.utils.data.Dataset):
    def __init__(self, ds, nclasses):
        self.ds = ds
        self.bins = [int(round(i*ds.K/nclasses)) for i in range(1, nclasses)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i, only_labels=False):
        if only_labels:
            y = self.ds.__getitem__(i, True)
            y = np.digitize(y, self.bins)
            return y
        x, y = self.ds[i]
        y = np.digitize(y, self.bins)
        return x, y

class ImageDataset(torch.utils.data.Dataset):
    modality = 'image'
    HFLIP = VFLIP = False
    CROPCENTER = False

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i, only_labels=False):
        fname = self.files[i]
        label = self.labels[i]
        if only_labels:
            return label
        if fname.lower().endswith('.tif') or fname.lower().endswith('.bmp'):
            img = imread(fname).astype(np.uint8)
            img = np.moveaxis(img, 2, 0)
            img = torch.tensor(img)
        else:
            img = torchvision.io.read_image(fname, torchvision.io.ImageReadMode.RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

class BACH(ImageDataset):
    # https://iciar2018-challenge.grand-challenge.org/Dataset/
    K = 4
    def __init__(self, root, transform):
        root = os.path.join(root, 'ICIAR2018_BACH_Challenge', 'Photos')
        classes = ['Normal', 'Benign', 'InSitu', 'Invasive']
        self.files = [os.path.join(root, klass, f) for klass in classes for f in sorted(os.listdir(os.path.join(root, klass))) if f.endswith('.tif')]
        self.labels = [i for i, klass in enumerate(classes) for f in os.listdir(os.path.join(root, klass)) if f.endswith('.tif')]
        self.transform = transform

class DHCI(ImageDataset):
    # http://graphics.cs.cmu.edu/projects/historicalColor/
    K = 5
    def __init__(self, root, transform):
        root = os.path.join(root, 'HistoricalColor-ECCV2012', 'data', 'imgs', 'decade_database')
        decades = sorted(os.listdir(root))
        self.files = [os.path.join(root, d, f) for d in decades for f in sorted(os.listdir(os.path.join(root, d))) if not f.startswith('.')]
        self.labels = [i for i, d in enumerate(decades) for f in sorted(os.listdir(os.path.join(root, d))) if not f.startswith('.')]
        self.transform = transform

class FGNET(ImageDataset):
    # https://yanweifu.github.io/FG_NET_data/
    K = 70
    HFLIP = True
    def __init__(self, root, transform):
        root = os.path.join(root, 'FGNET', 'images')
        files = sorted(os.listdir(root))
        self.files = [os.path.join(root, f) for f in files]
        self.labels = [int(f[4:6]) for f in files]
        self.transform = transform

class AFAD(ImageDataset):
    # http://afad-dataset.github.io/
    def __init__(self, root, folder, transform):
        root = os.path.join(root, folder)
        ages = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        self.files = [os.path.join(froot, f) for age in ages for froot, _, files in os.walk(os.path.join(root, age)) for f in sorted(files) if f.endswith('.jpg')]
        self.labels = [i for i, age in enumerate(ages) for froot, _, files in os.walk(os.path.join(root, age)) for f in sorted(files) if f.endswith('.jpg')]
        self.transform = transform

class AFAD_Lite(AFAD):
    K = 39-18+1
    def __init__(self, root, transform):
        super().__init__(root, 'AFAD-Lite', transform)

class AFAD_Full(AFAD):
    K = 75-15+1
    def __init__(self, root, transform):
        super().__init__(root, 'AFAD-Full', transform)

class SMEAR2005(ImageDataset):
    # http://mde-lab.aegean.gr/index.php/downloads
    K = 5
    def __init__(self, root, transform):
        class_folders = {1: 'normal_superficiel', 2: 'normal_intermediate',
            3: 'normal_columnar', 4: 'light_dysplastic', 5: 'moderate_dysplastic'}
        root = os.path.join(root, 'smear2005', 'New database pictures')
        self.files = [os.path.join(root, d, f) for _, d in class_folders.items() for f in sorted(os.listdir(os.path.join(root, d))) if f.endswith('BMP')]
        self.labels = [k-1 for k, d in class_folders.items() for f in sorted(os.listdir(os.path.join(root, d))) if f.endswith('BMP')]
        self.transform = transform

class FOCUSPATH(ImageDataset):
    # https://zenodo.org/record/3926181
    K = 12
    HFLIP = VFLIP = True
    CROPCENTER = True
    def __init__(self, root, transform):
        df = pd.read_excel(os.path.join(root, 'focuspath', 'DatabaseInfo.xlsx'))
        root = os.path.join(root, 'focuspath', 'FocusPath_full')
        self.files = [os.path.join(root, row['Name'][:-3] + 'png') for _, row in df.iterrows()]
        self.labels = [min(abs(row['Subjective Score']), 11) for _, row in df.iterrows()]
        self.transform = transform

class UTKFACE(ImageDataset):
    # https://susanqq.github.io/UTKFace/
    K = 116
    HFLIP = True
    def __init__(self, root, transform):
        root = os.path.join(root, 'UTKFace')
        files = sorted(os.listdir(root))
        self.files = [os.path.join(root, f) for f in files]
        self.labels = [int(f.split('_')[0])-1 for f in files]
        self.transform = transform

class CARSDB(ImageDataset):
    # https://pages.cs.wisc.edu/~yongjaelee/iccv2013.html
    K = 1999-1920+1
    HFLIP = True
    def __init__(self, root, transform):
        root = os.path.join(root, 'carsdb')
        tr = sorted(os.listdir(os.path.join(root, 'train')))
        ts = sorted(os.listdir(os.path.join(root, 'test')))
        train_json = json.load(open(os.path.join(root, 'train.json')))
        test_json = json.load(open(os.path.join(root, 'test.json')))
        self.files = [os.path.join(root, 'train', f) for f in tr] + [os.path.join(root, 'test', f) for f in ts]
        self.labels = [train_json[f]-1920 for f in tr] + [test_json[f]-1920 for f in ts]
        self.transform = transform

class CSAW_M(ImageDataset):
    # https://figshare.scilifelab.se/articles/dataset/CSAW-M_An_Ordinal_Classification_Dataset_for_Benchmarking_Mammographic_Masking_of_Cancer/14687271
    K = 8
    def __init__(self, root, transform):
        root = os.path.join(root, 'CSAW-M')
        tr = pd.read_csv(os.path.join(root, 'labels', 'CSAW-M_train.csv'), sep=';', usecols=[0, 1])
        ts = pd.read_csv(os.path.join(root, 'labels', 'CSAW-M_test.csv'), sep=';', usecols=[0, 1])
        self.files = [os.path.join(root, 'images', 'preprocessed', 'train', f) for f in tr['Filename']] + \
            [os.path.join(root, 'images', 'preprocessed', 'test', f) for f in ts['Filename']]
        self.labels = list(tr['Label']-1) + list(ts['Label']-1)
        self.transform = transform

class FFB(ImageDataset):
    # https://data.mendeley.com/datasets/424y96m6sw/1
    K = 5
    HFLIP = VFLIP = True
    def __init__(self, root, transform):
        root = os.path.join(root, 'FFB')
        dirs = sorted(os.listdir(os.path.join(root, 'Images')))
        self.files = [os.path.join(os.path.join(root, 'Images', d, f)) for d in dirs for f in sorted(os.listdir(os.path.join(root, 'Images', d)))]
        self.labels = [k for k, d in enumerate(dirs) for _ in range(len(os.listdir(os.path.join(root, 'Images', d))))]
        self.transform = transform

class TabularDataset(torch.utils.data.Dataset):
    modality = 'tabular'

    def __init__(self, root, fname, sep, cols_ignore, cols_category, col_label, labels, discretize_nbins):
        fname = os.path.join(root, 'UCI', fname)
        df = pd.read_csv(fname, header=None, sep=sep)
        X = df.drop(columns=df.columns[col_label])
        Y = df.iloc[:, col_label]
        if cols_ignore:
            X.drop(columns=X.columns[cols_ignore], inplace=True)
        X = pd.get_dummies(X, columns=cols_category).to_numpy(np.float32)
        X = (X-X.mean(0)) / X.std(0)  # z-normalization
        if discretize_nbins:
            Y = Y.to_numpy(np.int64)-1
            bins = np.linspace(0, Y.max(), discretize_nbins, False)
            Y = np.digitize(Y, bins)-1
        elif labels:
            Y = np.array([labels.index(y) for y in Y], np.int64)
        else:
            Y = Y.to_numpy(np.int64)-1
        self.K = Y.max()+1
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i, only_labels=False):
        if only_labels:
            return self.Y[i]
        return self.X[i], self.Y[i]

'''
wget https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data
'''

def ABALONE5(root, transform):
    return TabularDataset(root, 'abalone.data', ',', None, None, -1, None, 5)
def ABALONE10(root, transform):
    return TabularDataset(root, 'abalone.data', ',', None, None, -1, None, 10)
def BALANCE_SCALE(root, transform):
    return TabularDataset(root, 'balance-scale.data', ',', None, None, -1, None, None)
def CAR(root, transform):
    return TabularDataset(root, 'car.data', ',', None, None, -1, ['unacc', 'acc', 'good', 'vgood'], None)
def LENSES(root, transform):
    return TabularDataset(root, 'lenses.data', r'\s+', [0], [1, 2, 3, 4], -1, None, None)
def NEW_THYROID(root, transform):
    return TabularDataset(root, 'new-thyroid.data', ',', None, None, 0, None, None)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--datadir', default='/data/ordinal')
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    ds = globals()[args.dataset]
    print('total size:', len(ds(args.datadir, None)))
    ds = ds(args.datadir, None)
    i = np.random.choice(len(ds))
    print('K:', ds.K)
    print('training N:', len(ds))
    x, y = ds[i]
    plt.imshow(x.permute(1, 2, 0))
    plt.title(y)
    plt.show()
