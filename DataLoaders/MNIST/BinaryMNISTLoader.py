from __future__ import print_function
from IPython.core.debugger import Pdb
pdb = Pdb()
trace = pdb.set_trace

import math
import scipy.misc
import os
import time
import glob
from pathlib import Path
import h5py

import numpy as np
np.set_printoptions(suppress=True)

class DataLoader:
    def __init__(self, batch_size):
        self.dataset_name = 'Binary MNIST'
        self.batch_size = batch_size
        self.iter = 0
        self.image_size = [-1, 28, 28, 1]
        self.dataset_dir = str(Path.home())+'/Datasets/MNIST/BinaryMNIST/'
        try: self.load_h5()
        except: self.create_h5(); self.load_h5()
        self.setup('Training')
    
    def load_h5(self):
        processed_file_path = self.dataset_dir + 'binary_mnist.h5'
        print('Loading from h5 file: '+ processed_file_path)
        start_indiv = time.time()
        hf = h5py.File(processed_file_path, 'r')

        self.training_images = hf['training_images'][()]
        self.training_labels = hf['training_labels'][()]
        self.validation_images = hf['validation_images'][()]
        self.validation_labels = hf['validation_labels'][()]
        self.test_images = hf['test_images'][()]
        self.test_labels = hf['test_labels'][()]

        hf.flush(); hf.close()
        end_indiv = time.time()
        print('Success loading data from h5 file. Time: '+ str(end_indiv-start_indiv))

        self.training_images = self.training_images.astype(np.float32)
        self.training_labels = self.training_labels.astype(np.float32)
        self.validation_images = self.validation_images.astype(np.float32)
        self.validation_labels = self.validation_labels.astype(np.float32)
        self.test_images = self.test_images.astype(np.float32)
        self.test_labels = self.test_labels.astype(np.float32)

        self.label_size = [-1] + list(self.training_labels.shape[1:])

    def create_h5(self):
        print('Loading from h5 file failed. Creating h5 file from data sources.')
        if not os.path.exists(self.dataset_dir): os.makedirs(self.dataset_dir)
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets(self.dataset_dir, one_hot=True)
        mnist_image_size = [-1, 28, 28, 1]
        for f in glob.glob(self.dataset_dir + '*.gz'): os.remove(f)

        training_images = (mnist.train.images > 0.5).reshape(self.image_size)
        validation_images = (mnist.validation.images > 0.5).reshape(self.image_size)
        test_images = (mnist.test.images > 0.5).reshape(self.image_size)
        
        training_labels = (mnist.train.labels).astype(np.bool)
        validation_labels = (mnist.validation.labels).astype(np.bool)
        test_labels = (mnist.test.labels).astype(np.bool)
        
        processed_file_path = self.dataset_dir + 'binary_mnist.h5'
        print('Creating h5 file: ' + processed_file_path)
        hf = h5py.File(processed_file_path, 'w')
        hf.create_dataset('training_images', data=training_images, chunks=tuple(training_images.shape))
        hf.create_dataset('training_labels', data=training_labels, chunks=tuple(training_labels.shape))
        hf.create_dataset('validation_images', data=validation_images, chunks=tuple(validation_images.shape))
        hf.create_dataset('validation_labels', data=validation_labels, chunks=tuple(validation_labels.shape))
        hf.create_dataset('test_images', data=test_images, chunks=tuple(test_images.shape))
        hf.create_dataset('test_labels', data=test_labels, chunks=tuple(test_labels.shape))
        hf.flush(); hf.close()

    def report_status(self):
        print('\n')
        print('################  DataLoader  ####################')
        print('Dataset name: '+str(self.dataset_name))
        print('Overall batch size: '+str(self.batch_size))
        print('Image size: '+str(self.image_size[1:]))
        print('Label size: '+str(self.image_size[1:]))
        print('Stage: '+str(self.stage))
        print('Data order randomized: '+str(self.randomized))
        print('# Batches: ' + str(self.curr_max_iter))
        print('# Samples: ' + str(self.curr_n_samples))
        print('##################################################')
        print('\n')

    def setup(self, stage, randomized=False, verbose=True):
        assert (stage == 'Training' or stage == 'Validation' or stage == 'Test')
        self.stage = stage

        if self.stage == 'Training':
            self.curr_data_order = np.arange(len(self.training_images))
            if randomized: self.curr_data_order=np.random.permutation(self.curr_data_order)
            self.curr_images = self.training_images[self.curr_data_order, ...]
            self.curr_labels = self.training_labels[self.curr_data_order, ...]
        elif self.stage == 'Validation':
            self.curr_data_order = np.arange(len(self.validation_images))
            if randomized: self.curr_data_order=np.random.permutation(self.curr_data_order)
            self.curr_images = self.validation_images[self.curr_data_order, ...]
            self.curr_labels = self.validation_labels[self.curr_data_order, ...]
        elif self.stage == 'Test':
            self.curr_data_order = np.arange(len(self.test_images))
            if randomized: self.curr_data_order=np.random.permutation(self.curr_data_order)
            self.curr_images = self.test_images[self.curr_data_order, ...]
            self.curr_labels = self.test_labels[self.curr_data_order, ...]

        self.randomized = randomized
        self.curr_n_samples = len(self.curr_images)
        self.curr_max_iter = np.int(np.ceil(float(self.curr_n_samples)/float(self.batch_size)))
        if verbose: self.report_status()
        self.reset()

    def reset(self):
        self.iter = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iter == self.curr_max_iter: raise StopIteration

        curr_start_ind = self.iter*self.batch_size
        curr_end_ind = min(curr_start_ind+self.batch_size, self.curr_n_samples)
        curr_batch_size = curr_end_ind-curr_start_ind

        batch_dict = {'Image': self.curr_images[curr_start_ind:curr_end_ind:, ...], 
                      'Label': self.curr_labels[curr_start_ind:curr_end_ind:, ...]}
        self.iter += 1
        return self.iter-1, curr_batch_size, batch_dict


