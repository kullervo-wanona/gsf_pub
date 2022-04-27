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
import pickle
import wget
import tarfile


import numpy as np
np.set_printoptions(suppress=True)

class DataLoader:
    def __init__(self, batch_size):
        self.dataset_name = 'Cifar100'
        self.batch_size = batch_size
        self.iter = 0
        self.image_size = [-1, 32, 32, 3]
        self.dataset_dir = str(Path.home())+'/Datasets/CIFAR/Cifar100/'
        try: self.load_h5()
        except: self.create_h5(); self.load_h5()        
        self.setup('Training')

    def load_h5(self):
        processed_file_path = self.dataset_dir + 'cifar100.h5'
        print('Loading from h5 file: '+ processed_file_path)
        start_indiv = time.time()

        hf = h5py.File(processed_file_path, 'r')
        self.coarse_class_names = hf['coarse_class_names'][()]
        self.fine_class_names = hf['fine_class_names'][()]
        self.training_images = hf['training_images'][()]
        self.training_coarse_labels = hf['training_coarse_labels'][()]
        self.training_fine_labels = hf['training_fine_labels'][()]
        self.training_filenames = hf['training_filenames'][()]
        self.test_images = hf['test_images'][()]
        self.test_coarse_labels = hf['test_coarse_labels'][()]
        self.test_fine_labels = hf['test_fine_labels'][()]
        self.test_filenames = hf['test_filenames'][()]
        hf.flush(); hf.close()
        end_indiv = time.time()
        print('Success loading data from h5 file. Time: '+ str(end_indiv-start_indiv))
        
        self.training_images = self.training_images.astype(np.float32)/255.
        self.training_coarse_labels = self.training_coarse_labels.astype(np.float32)
        self.training_fine_labels = self.training_fine_labels.astype(np.float32)
        self.test_images = self.test_images.astype(np.float32)/255.
        self.test_coarse_labels = self.test_coarse_labels.astype(np.float32)
        self.test_fine_labels = self.test_fine_labels.astype(np.float32)

        self.label_size = [-1] + list(self.training_fine_labels.shape[1:])

    def create_h5(self):
        print('Loading from h5 file failed. Creating h5 file from data sources.')
        if not os.path.exists(self.dataset_dir): os.makedirs(self.dataset_dir)
        cifar100_image_size = [-1, 3, 32, 32]

        wget.download('http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', self.dataset_dir)
        tarfile_object = tarfile.open(self.dataset_dir + 'cifar-100-python.tar.gz')
        tarfile_object.extractall(self.dataset_dir)
        tarfile_object.close()
        os.remove(self.dataset_dir + 'cifar-100-python.tar.gz')
        f = open(self.dataset_dir + 'cifar-100-python/meta', 'rb')
        metadata_dict = pickle.load(f, encoding='bytes')
        f.close()

        f = open(self.dataset_dir + 'cifar-100-python/train', 'rb')
        training_chunk = (pickle.load(f, encoding='bytes'))
        f.close()

        f = open(self.dataset_dir + 'cifar-100-python/test', 'rb')
        test_chunk = (pickle.load(f, encoding='bytes'))
        f.close()

        for f in glob.glob(self.dataset_dir + 'cifar-100-python/*'): os.remove(f)
        os.rmdir(self.dataset_dir + 'cifar-100-python/')

        coarse_class_names = metadata_dict[b'coarse_label_names']
        fine_class_names = metadata_dict[b'fine_label_names']

        training_images = training_chunk[b'data'].reshape(cifar100_image_size)
        training_filenames = np.asarray([e.decode("utf-8") for e in training_chunk[b'filenames']])   

        training_coarse_labels = np.zeros([training_images.shape[0], len(coarse_class_names)], np.bool)
        for j in range(len(training_chunk[b'coarse_labels'])):
            training_coarse_labels[j, training_chunk[b'coarse_labels'][j]] = True
        
        training_fine_labels = np.zeros([training_images.shape[0], len(fine_class_names)], np.bool)
        for j in range(len(training_chunk[b'fine_labels'])):
            training_fine_labels[j, training_chunk[b'fine_labels'][j]] = True
        
        test_images = test_chunk[b'data'].reshape(cifar100_image_size)
        test_filenames = np.asarray([e.decode("utf-8") for e in test_chunk[b'filenames']])   

        test_coarse_labels = np.zeros([test_images.shape[0], len(coarse_class_names)], np.bool)
        for j in range(len(test_chunk[b'coarse_labels'])):
            test_coarse_labels[j, test_chunk[b'coarse_labels'][j]] = True
        
        test_fine_labels = np.zeros([test_images.shape[0], len(fine_class_names)], np.bool)
        for j in range(len(test_chunk[b'fine_labels'])):
            test_fine_labels[j, test_chunk[b'fine_labels'][j]] = True

        training_images = np.transpose(training_images, [0, 2, 3, 1])
        test_images = np.transpose(test_images, [0, 2, 3, 1])

        coarse_class_names = np.array(coarse_class_names, dtype='S')
        fine_class_names = np.array(fine_class_names, dtype='S')
        training_filenames = np.array(training_filenames, dtype='S')
        test_filenames = np.array(test_filenames, dtype='S')

        processed_file_path = self.dataset_dir + 'cifar100.h5'
        print('Creating h5 file: ' + processed_file_path)
        hf = h5py.File(processed_file_path, 'w')
        hf.create_dataset('coarse_class_names', data=coarse_class_names, chunks=tuple(coarse_class_names.shape))
        hf.create_dataset('fine_class_names', data=fine_class_names, chunks=tuple(fine_class_names.shape))
        hf.create_dataset('training_images', data=training_images, chunks=tuple(training_images.shape))
        hf.create_dataset('training_coarse_labels', data=training_coarse_labels, chunks=tuple(training_coarse_labels.shape))
        hf.create_dataset('training_fine_labels', data=training_fine_labels, chunks=tuple(training_fine_labels.shape))
        hf.create_dataset('training_filenames', data=training_filenames, chunks=tuple(training_filenames.shape))
        hf.create_dataset('test_images', data=test_images, chunks=tuple(test_images.shape))
        hf.create_dataset('test_coarse_labels', data=test_coarse_labels, chunks=tuple(test_coarse_labels.shape))
        hf.create_dataset('test_fine_labels', data=test_fine_labels, chunks=tuple(test_fine_labels.shape))
        hf.create_dataset('test_filenames', data=test_filenames, chunks=tuple(test_filenames.shape))
        hf.flush(); hf.close()

    def report_status(self):
        print('\n')
        print('################  DataLoader  ####################')
        print('Dataset name: '+str(self.dataset_name))
        print('Overall batch size: '+str(self.batch_size))
        print('Image size: '+str(self.image_size[1:]))
        print('Label size: '+str(self.label_size[1:]))
        print('Stage: '+str(self.stage))
        print('Data order randomized: '+str(self.randomized))
        print('# Batches: ' + str(self.curr_max_iter))
        print('# Samples: ' + str(self.curr_n_samples))
        print('##################################################')
        print('\n')

    def setup(self, stage, randomized=False, verbose=True):
        assert (stage == 'Training' or stage == 'Test')
        self.stage = stage

        if self.stage == 'Training':
            self.curr_data_order = np.arange(len(self.training_images))
            if randomized: self.curr_data_order=np.random.permutation(self.curr_data_order)
            self.curr_images = self.training_images[self.curr_data_order, ...]
            self.curr_labels = self.training_fine_labels[self.curr_data_order, ...]
        elif self.stage == 'Test':
            self.curr_data_order = np.arange(len(self.test_images))
            if randomized: self.curr_data_order=np.random.permutation(self.curr_data_order)
            self.curr_images = self.test_images[self.curr_data_order, ...]
            self.curr_labels = self.test_fine_labels[self.curr_data_order, ...]

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

        batch_dict = {'Image': (self.curr_images[curr_start_ind:curr_end_ind:, ...]), 
                      'Label': (self.curr_labels[curr_start_ind:curr_end_ind:, ...])}
        self.iter += 1
        return self.iter-1, curr_batch_size, batch_dict

