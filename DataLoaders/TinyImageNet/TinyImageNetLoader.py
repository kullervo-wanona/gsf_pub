from __future__ import print_function
from IPython.core.debugger import Pdb
pdb = Pdb()
trace = pdb.set_trace

import math
import scipy.misc
import os
import time
import glob
import gc
import csv
from pathlib import Path
import h5py
import pickle
import wget
import zipfile
from PIL import Image
import shutil

import numpy as np
np.set_printoptions(suppress=True)

class DataLoader:
    def __init__(self, batch_size):
        self.dataset_name = 'TinyImageNet'
        self.batch_size = batch_size
        self.iter = 0
        self.image_size = [-1, 64, 64, 3]
        self.dataset_dir = str(Path.home())+'/Datasets/TinyImageNet/'
        try: self.load_h5()
        except: self.create_h5(); self.load_h5()
        self.setup('Training')

    def load_h5(self):
        processed_file_path = self.dataset_dir + 'tiny_image_net.h5'
        print('Loading from h5 file: '+ processed_file_path)
        start_indiv = time.time()

        hf = h5py.File(processed_file_path, 'r')
        self.class_names = hf['class_names'][()]
        self.training_filenames = hf['training_filenames'][()]
        self.training_labels = hf['training_labels'][()]
        self.training_bounding_boxes = hf['training_bounding_boxes'][()]
        self.training_images = hf['training_images'][()]
        self.validation_filenames = hf['validation_filenames'][()]
        self.validation_labels = hf['validation_labels'][()]
        self.validation_bounding_boxes = hf['validation_bounding_boxes'][()]
        self.validation_images = hf['validation_images'][()]
        # self.test_images = hf['test_images'][()]
        # self.test_filenames = hf['test_filenames'][()]
        hf.close()
        end_indiv = time.time()
        print('Success loading data from h5 file. Time: '+ str(end_indiv-start_indiv))
        
        self.training_images = self.training_images.astype(np.float32)/255.
        self.training_labels = self.training_labels.astype(np.float32)
        self.validation_images = self.validation_images.astype(np.float32)/255.
        self.validation_labels = self.validation_labels.astype(np.float32)

        self.label_size = [-1] + list(self.training_labels.shape[1:])

    def create_h5(self):
        print('Loading from h5 file failed. Creating h5 file from data sources.')
        if not os.path.exists(self.dataset_dir): os.makedirs(self.dataset_dir)
        
        if not os.path.exists(self.dataset_dir + 'tiny-imagenet-200.zip'): 
            wget.download('http://cs231n.stanford.edu/tiny-imagenet-200.zip', self.dataset_dir)
        print('\n\nUnzipping the dataset. May take a few minutes.')
        zipfile_object = zipfile.ZipFile(self.dataset_dir + 'tiny-imagenet-200.zip', 'r')
        zipfile_object.extractall(self.dataset_dir)
        zipfile_object.close()
        os.remove(self.dataset_dir + 'tiny-imagenet-200.zip')

        training_class_paths = sorted(glob.glob(self.dataset_dir + 'tiny-imagenet-200/train/*/'))
        training_class_names = [os.path.basename(e[:-1]) for e in training_class_paths]
        n_training_classes = len(training_class_names)

        training_filenames = []
        training_labels_list = []
        training_bounding_boxes = []
        training_images = []

        class_index_dict = {}
        print('Processing the training data.')
        for class_id, training_class_path in enumerate(training_class_paths):
            class_name = training_class_names[class_id]
            class_index_dict[class_name] = class_id
            print('Processing training class: ' + class_name + '; ' + str(class_id) + '/' + str(n_training_classes))

            with open(training_class_path + class_name + '_boxes.txt', newline='') as csvfile:
                row_counter = 0
                for row in csv.reader(csvfile, delimiter='\t', quotechar='|'):
                    training_filenames.append(row[0])
                    training_labels_list.append(class_id)
                    training_bounding_boxes.append(np.asarray(row[1:])[np.newaxis, :].astype(np.uint8))

                    image_path = training_class_path + 'images/' + row[0]
                    rgb_image = np.asarray(Image.open(image_path).convert('RGB'))
                    training_images.append(rgb_image[np.newaxis, ...])

        training_filenames = np.asarray(training_filenames)
        training_labels = np.zeros((len(training_filenames), n_training_classes), np.bool)
        for i, e in enumerate(training_labels_list): training_labels[i, e] = True
        assert ((training_labels.sum(1) == 1).all())
        training_bounding_boxes = np.concatenate(training_bounding_boxes, axis=0)
        training_images = np.concatenate(training_images, axis=0)
        gc.collect(); gc.collect()

        training_indices = np.random.permutation(np.arange(len(training_filenames)))
        training_filenames = training_filenames[training_indices, ...]
        training_labels = training_labels[training_indices, ...]
        training_images = training_images[training_indices, ...]
        training_bounding_boxes = training_bounding_boxes[training_indices, ...]
        gc.collect(); gc.collect()

        print('Processing the validation data.')
        validation_filenames = []
        validation_labels_list = []
        validation_bounding_boxes = []
        validation_images = []

        with open(self.dataset_dir + 'tiny-imagenet-200/val/val_annotations.txt', newline='') as csvfile:
            row_counter = 0
            for row in csv.reader(csvfile, delimiter='\t', quotechar='|'):
                validation_filenames.append(row[0])
                validation_labels_list.append(class_index_dict[row[1]])
                validation_bounding_boxes.append(np.asarray(row[2:])[np.newaxis, :].astype(np.uint8))

                image_path = self.dataset_dir + 'tiny-imagenet-200/val/images/' + row[0]
                rgb_image = np.asarray(Image.open(image_path).convert('RGB'))
                validation_images.append(rgb_image[np.newaxis, ...])

        validation_filenames = np.asarray(validation_filenames)
        assert (n_training_classes == (max(validation_labels_list)+1))
        validation_labels = np.zeros((len(validation_filenames), n_training_classes), np.bool)
        for i, e in enumerate(validation_labels_list): validation_labels[i, e] = True
        validation_bounding_boxes = np.concatenate(validation_bounding_boxes, axis=0)
        validation_images = np.concatenate(validation_images, axis=0)
        gc.collect(); gc.collect()

        validation_indices = np.random.permutation(np.arange(len(validation_filenames)))
        validation_filenames = validation_filenames[validation_indices, ...]
        validation_labels = validation_labels[validation_indices, ...]
        validation_bounding_boxes = validation_bounding_boxes[validation_indices, ...]
        validation_images = validation_images[validation_indices, ...]
        gc.collect(); gc.collect()

        print('Processing the test data.')
        test_image_paths = sorted(glob.glob(self.dataset_dir + 'tiny-imagenet-200/test/images/*.JPEG'))
        test_filenames = [os.path.basename(e) for e in test_image_paths]
        test_images = []
        for image_path in test_image_paths:
            rgb_image = np.asarray(Image.open(image_path).convert('RGB'))
            test_images.append(rgb_image[np.newaxis, ...])

        test_filenames = np.asarray(test_filenames)
        test_images = np.concatenate(test_images, axis=0)
        gc.collect(); gc.collect()

        test_indices = np.random.permutation(np.arange(len(test_filenames)))
        test_filenames = test_filenames[test_indices, ...]
        test_images = test_images[test_indices, ...]
        gc.collect(); gc.collect()

        class_names = [None]*validation_labels.shape[1]
        for class_name in class_index_dict:
            class_names[class_index_dict[class_name]] = class_name
        class_names = np.array(class_names, dtype='S')

        training_filenames = np.array(training_filenames, dtype='S')
        validation_filenames = np.array(validation_filenames, dtype='S')
        test_filenames = np.array(test_filenames, dtype='S')

        shutil.rmtree(self.dataset_dir + 'tiny-imagenet-200/')
        gc.collect(); gc.collect()

        processed_file_path = self.dataset_dir + 'tiny_image_net.h5'
        print('Creating h5 file: ' + processed_file_path)
        hf = h5py.File(processed_file_path, 'w')
        hf.create_dataset('class_names', data=class_names, chunks=tuple(class_names.shape))
        hf.create_dataset('training_filenames', data=training_filenames, chunks=tuple(training_filenames.shape))
        hf.create_dataset('training_labels', data=training_labels, chunks=tuple(training_labels.shape))
        hf.create_dataset('training_bounding_boxes', data=training_bounding_boxes, chunks=tuple(training_bounding_boxes.shape))
        hf.create_dataset('training_images', data=training_images, chunks=tuple(training_images.shape))
        hf.create_dataset('validation_filenames', data=validation_filenames, chunks=tuple(validation_filenames.shape))
        hf.create_dataset('validation_labels', data=validation_labels, chunks=tuple(validation_labels.shape))
        hf.create_dataset('validation_bounding_boxes', data=validation_bounding_boxes, chunks=tuple(validation_bounding_boxes.shape))
        hf.create_dataset('validation_images', data=validation_images, chunks=tuple(validation_images.shape))
        hf.create_dataset('test_images', data=test_images, chunks=tuple(test_images.shape))
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
            self.curr_labels = self.training_labels[self.curr_data_order, ...]
        elif self.stage == 'Test':
            self.curr_data_order = np.arange(len(self.validation_images))
            if randomized: self.curr_data_order=np.random.permutation(self.curr_data_order)
            self.curr_images = self.validation_images[self.curr_data_order, ...]
            self.curr_labels = self.validation_labels[self.curr_data_order, ...]

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







