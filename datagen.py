import numpy as np
import keras
import cv2
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from retinex import automatedMSRCR

from albumentations.augmentations.functional import brightness_contrast_adjust
import albumentations as A
class IndependentRandomBrightnessContrast(A.ImageOnlyTransform):
    """ Change brightness & contrast independently per channels """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5):
        super(IndependentRandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = A.to_tuple(brightness_limit)
        self.contrast_limit = A.to_tuple(contrast_limit)

    def apply(self, img, **params):
        img = img.copy()
        for ch in range(img.shape[2]):
            alpha = 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1])
            beta = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])
            img[..., ch] = brightness_contrast_adjust(img[..., ch], alpha, beta)

        return img

albu_tfms =  A.Compose([
    A.OneOf([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, value=0),
        A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0),
        A.NoOp()
    ]),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.5,
                                   contrast_limit=0.4),
        IndependentRandomBrightnessContrast(brightness_limit=0.25,
                                                        contrast_limit=0.24),
        A.RandomGamma(gamma_limit=(50, 150)),
        A.NoOp()
    ]),
    A.OneOf([
        A.FancyPCA(),
        A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
        A.HueSaturationValue(hue_shift_limit=5,
                             sat_shift_limit=5),
        A.NoOp()
    ]),
    A.OneOf([
        A.CLAHE(),
        A.NoOp()
    ]),
    A.HorizontalFlip(p=0.5),
])

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32),
                 shuffle=True, type_gen='train'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.type_gen = type_gen
        self.aug_gen = ImageDataGenerator()
        print("all:", len(self.list_IDs), " batch per epoch", int(np.floor(len(self.list_IDs) / self.batch_size)))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        if self.type_gen == 'predict':
            return X
        else:
            return X, y

    def sequence_augment(self, img):
        name_list = ['rotate', 'width_shift', 'height_shift',
                     'brightness', 'flip_horizontal', 'width_zoom',
                     'height_zoom']
        dictkey_list = ['theta', 'ty', 'tx',
                        'brightness', 'flip_horizontal', 'zy',
                        'zx']
        random_aug = np.random.randint(2, 5)  # random 2-4 augmentation method
        pick_idx = np.random.choice(len(dictkey_list), random_aug, replace=False)  #

        dict_input = {}
        for i in pick_idx:
            if dictkey_list[i] == 'theta':
                dict_input['theta'] = np.random.randint(-10, 10)

            elif dictkey_list[i] == 'ty':  # width_shift
                dict_input['ty'] = np.random.randint(-20, 20)

            elif dictkey_list[i] == 'tx':  # height_shift
                dict_input['tx'] = np.random.randint(-20, 20)

            elif dictkey_list[i] == 'brightness':
                dict_input['brightness'] = np.random.uniform(0.75, 1.25)

            elif dictkey_list[i] == 'flip_horizontal':
                dict_input['flip_horizontal'] = bool(random.getrandbits(1))

            elif dictkey_list[i] == 'zy':  # width_zoom
                dict_input['zy'] = np.random.uniform(0.75, 1.25)

            elif dictkey_list[i] == 'zx':  # height_zoom
                dict_input['zx'] = np.random.uniform(0.75, 1.25)
        img = self.aug_gen.apply_transform(img, dict_input)
        return img

    def albu_aug(self, image, tfms = albu_tfms):
        seed = random.randint(0, 99999)

        random.seed(seed)
        image = tfms(image=image.astype('uint8'))['image']
        return image

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, self.dim[0], self.dim[1], 3)), np.empty((self.batch_size, self.dim[0], self.dim[1], 3))]
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):  # ID is name of file
            img = cv2.imread(ID)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.dim[1], self.dim[0]))

            if self.type_gen =='train':
                img = self.sequence_augment(img)

                new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                new_img = np.expand_dims(new_img, -1)
                new_img = automatedMSRCR(new_img, [10, 20, 30])
                new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)

                X[0][i] = img/255.0
                X[1][i] = new_img/255.0
            else:
                new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                new_img = np.expand_dims(new_img, -1)
                new_img = automatedMSRCR(new_img, [10, 20, 30])
                new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)

                X[0][i] = img/255.0
                X[1][i] = new_img/255.0

            Y[i] = self.labels[ID]

        return X, Y

