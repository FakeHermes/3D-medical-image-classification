from collections.abc import Sequence
import random
import os
import keras
import numpy as np
from mylib.dataloader.path_manager import PATH
from mylib.utils.misc import rotation, reflection, crop, random_center, _triple,mixup
import pandas as pd

INFO = PATH.info
TEST = PATH.test
test_path=PATH.test_nodule_path
LABEL = [0,1]

class ClfDataset(Sequence):
    def __init__(self, crop_size=32, move=None, subset=[0, 1, 2, 3],
                 define_label=lambda l: [l[0],l[1]]):
        '''The classification-only dataset.

        :param crop_size: the input size
        :param move: the random move
        :param subset: choose which subset to use
        :param define_label: how to define the label. default: for 2-output classification one hot encoding.
        '''
        index = []
        for sset in subset:
            index += list(INFO[INFO['subset'] == sset].index)
        self.index = tuple(sorted(index))  # the index in the info
        self.label = np.array([[label == s for label in LABEL] for s in INFO.loc[self.index, 'lable']])
        self.transform = Transform(crop_size,move)
        self.define_label = define_label

    def __getitem__(self, item):
        name = INFO.loc[self.index[item], 'name']
        name2 = INFO.loc[self.index[item//2], 'name']
        label = self.label[item]
        label2 = self.label[item//2]
		#旋转
        with np.load(os.path.join(PATH.nodule_path, '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel'])
        #mixup
        with np.load(os.path.join(PATH.nodule_path, '%s.npz' % name2)) as npz2:
            voxel2 = self.transform(npz2['voxel'])
            voxel2,y2 = mixup(x1=voxel,x2=voxel2,y1=label,y2=label2)
        
        
        return voxel2, self.define_label(y2)

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        
        return np.array(xs), np.array(ys)


def get_x_test():
    list_path=TEST
    name=list_path['ID']
    num=len(list_path)
    xs = np.empty((num,*(32,32,32), 1))
    print(np.shape(xs))
    for i in range(num):
        with np.load(os.path.join(test_path, '%s.npz' % name[i])) as npz:
            voxel=npz['voxel'][34:66,34:66,34:66]
            voxel=np.expand_dims(voxel,axis=-1)
        xs[i,]=voxel
        
    return xs

def get_loader(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)


def get_balanced_loader(dataset, batch_sizes):
    assert len(batch_sizes) == len(LABEL)
    total_size = len(dataset)
    print('Size', total_size)
    index_generators = []
    for l_idx in range(len(batch_sizes)):
        # this must be list, or `l_idx` will not be eval
        iterator = [i for i in range(total_size) if dataset.label[i, l_idx]]
        index_generators.append(shuffle_iterator(iterator))
    while True:
        data = []
        for i, batch_size in enumerate(batch_sizes):
            generator = index_generators[i]
            for _ in range(batch_size):
                idx = next(generator)
                data.append(dataset[idx])
        yield dataset._collate_fn(data)


class Transform:
    '''The online data augmentation, including:
    1) random move the center by `move`
    2) rotation 90 degrees increments
    3) reflection in any axis
    '''

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            #center = random_center(shape, self.move)
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                print('rotation')
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        
            

def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)
