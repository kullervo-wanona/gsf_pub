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
import zipfile
from shutil import copyfile
import csv
import gc
from PIL import Image

import numpy as np
np.set_printoptions(suppress=True)

class DataLoader:
    def __init__(self, batch_size):
        self.dataset_name = 'CelebA 128'
        self.batch_size = batch_size
        self.buffer_batch_ratio = 200
        self.iter = 0
        self.image_size = [-1, 128, 128, 3]
        self.dataset_dir = str(Path.home())+'/Datasets/CelebA/CelebA128/'
        try: self.load_h5()
        except: self.create_h5(); self.load_h5()
        self.setup('Training')
        self.setup_buffers()

    def setup_buffers(self):
        self.buffer_images = np.zeros([self.batch_size*self.buffer_batch_ratio,] + self.image_size[1:], np.float32)
        self.buffer_attributes = np.zeros([self.batch_size*self.buffer_batch_ratio,] + self.attribute_size[1:], np.float32)

    def load_h5(self):
        processed_file_path = self.dataset_dir + 'celebA128.h5'
        print('Loading from h5 file: '+ processed_file_path)
        start_indiv = time.time()

        hf = h5py.File(processed_file_path, 'r')
        self.attribute_names = hf['attribute_names']
        self.landmark_names = hf['landmark_names']
        self.training_images = hf['training_images']
        self.training_attributes = hf['training_attributes']
        self.training_landmarks = hf['training_landmarks']
        self.validation_images = hf['validation_images']
        self.validation_attributes = hf['validation_attributes']
        self.validation_landmarks = hf['validation_landmarks']
        self.test_images = hf['test_images']
        self.test_attributes = hf['test_attributes']
        self.test_landmarks = hf['test_landmarks']
        end_indiv = time.time()
        print('Success loading data from h5 file. Time: '+ str(end_indiv-start_indiv))

        self.attribute_size = [-1] + list(self.training_attributes.shape[1:])

    def crop_raw_image(self, raw_image):
        crop_shape = min(raw_image.shape[0], raw_image.shape[1])
        crop_image_size = [crop_shape, crop_shape]        
        start_x, start_y = raw_image.shape[0]//2-crop_image_size[0]//2, raw_image.shape[1]//2-crop_image_size[1]//2
        return raw_image[start_x:start_x+crop_image_size[0], start_y:start_y+crop_image_size[1], ...]

    def downsample_image(self, raw_image, downsample_image_size, downsample_method='bicubic'):
        assert (downsample_method == 'nearest' or downsample_method == 'bilinear' or \
            downsample_method == 'bicubic' or downsample_method == 'lanczos')
        if downsample_method == 'nearest': 
            downsampled = np.array(Image.fromarray(raw_image).resize(downsample_image_size, Image.NEAREST))
        elif downsample_method == 'bilinear': 
            downsampled = np.array(Image.fromarray(raw_image).resize(downsample_image_size, Image.BILINEAR))
        elif downsample_method == 'bicubic': 
            downsampled = np.array(Image.fromarray(raw_image).resize(downsample_image_size, Image.BICUBIC))
        elif downsample_method == 'lanczos': 
            downsampled = np.array(Image.fromarray(raw_image).resize(downsample_image_size, Image.LANCZOS))
        return downsampled

    def create_h5(self, chunk_size=5000):
        print('Loading from h5 file failed. Creating h5 file from data sources.')
        if not os.path.exists(self.dataset_dir): os.makedirs(self.dataset_dir)
        source_dir = str(Path.home())+'/DataSources/CelebA/'
        
        filenames = []
        with open(source_dir + 'list_eval_partition.txt', newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i, row in enumerate(rows): 
                filenames.append(row[0])

        training_mask = np.zeros(len(filenames), np.bool)
        validation_mask = np.zeros(len(filenames), np.bool)
        test_mask = np.zeros(len(filenames), np.bool)

        with open(source_dir + 'list_eval_partition.txt', newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i, row in enumerate(rows): 
                try:
                    assert (filenames[i] == row[0])
                except:
                    trace()
                if row[1] == '0': 
                    training_mask[i] = True
                elif row[1] == '1': 
                    validation_mask[i] = True
                elif row[1] == '2': 
                    test_mask[i] = True
        assert (np.all((training_mask.astype(np.float)+validation_mask.astype(np.float)+test_mask.astype(np.float)) == 1))

        processed_file_path = self.dataset_dir + 'celebA128.h5'
        print('Creating h5 file: ' + processed_file_path)
        hf = h5py.File(processed_file_path, 'w')

        attribute_names = None
        attributes = None
        with open(source_dir + 'list_attr_celeba.txt', newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i, row in enumerate(rows): 
                if i == 0: 
                    assert (len(filenames) == int(row[0]))
                elif i == 1:
                    attribute_names = [e for e in row if e != '']
                    attributes = np.zeros((len(filenames), len(attribute_names)), np.int)
                else:
                    row_stripped = [e for e in row if e != '']
                    assert (filenames[i-2] == row_stripped[0])                    
                    attributes[i-2] = np.asarray(row_stripped[1:]).astype(np.int)
            assert (len(filenames)+2 == (i+1))
        assert (np.all(((attributes == 1).astype(np.float) + (attributes == -1).astype(np.float)) == 1))
        attributes = attributes == 1

        training_attributes = attributes[training_mask, ...]
        validation_attributes = attributes[validation_mask, ...]
        test_attributes = attributes[test_mask, ...]

        n_training_images = training_attributes.shape[0]
        n_validation_images = validation_attributes.shape[0]
        n_test_images = test_attributes.shape[0]

        attributes = None
        gc.collect(); gc.collect()

        attribute_names = np.array(attribute_names, dtype='S')
        hf.create_dataset('attribute_names', data=attribute_names, chunks=tuple(attribute_names.shape))
        hf.create_dataset('training_attributes', data=training_attributes, chunks=tuple(training_attributes.shape))
        hf.create_dataset('validation_attributes', data=validation_attributes, chunks=tuple(validation_attributes.shape))
        hf.create_dataset('test_attributes', data=test_attributes, chunks=tuple(test_attributes.shape))
        hf.flush();
        attribute_names, training_attributes, validation_attributes, test_attributes = None, None, None, None
        gc.collect(); gc.collect()

        landmark_names = None
        landmarks = None
        with open(source_dir + 'list_landmarks_align_celeba.txt', newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i, row in enumerate(rows): 
                if i == 0: 
                    assert (len(filenames) == int(row[0]))
                elif i == 1:
                    landmark_names = row
                    landmarks = np.zeros((len(filenames), len(landmark_names)), np.int)
                else:
                    row_stripped = [e for e in row if e != '']
                    assert (filenames[i-2] == row_stripped[0])
                    landmarks[i-2] = np.asarray(row_stripped[1:]).astype(np.int)
            assert (len(filenames)+2 == (i+1))
        assert (landmarks.max() < 256)
        landmarks = landmarks.astype(np.uint8)

        training_landmarks = landmarks[training_mask, ...]
        validation_landmarks = landmarks[validation_mask, ...]
        test_landmarks = landmarks[test_mask, ...]
        landmarks = None
        gc.collect(); gc.collect()

        landmark_names = np.array(landmark_names, dtype='S')
        hf.create_dataset('landmark_names', data=landmark_names, chunks=tuple(landmark_names.shape))
        hf.create_dataset('training_landmarks', data=training_landmarks, chunks=tuple(training_landmarks.shape))
        hf.create_dataset('validation_landmarks', data=validation_landmarks, chunks=tuple(validation_landmarks.shape))
        hf.create_dataset('test_landmarks', data=test_landmarks, chunks=tuple(test_landmarks.shape))
        hf.flush();
        landmark_names, training_landmarks, validation_landmarks, test_landmarks = None, None, None, None
        gc.collect(); gc.collect()

        training_images_data = hf.create_dataset('training_images', tuple([n_training_images]+self.image_size[1:]), chunks=tuple([min(chunk_size, n_training_images),]+self.image_size[1:]), dtype='uint8')        
        validation_images_data = hf.create_dataset('validation_images', tuple([n_validation_images]+self.image_size[1:]), chunks=tuple([min(chunk_size, n_validation_images),]+self.image_size[1:]), dtype='uint8')        
        test_images_data = hf.create_dataset('test_images', tuple([n_test_images]+self.image_size[1:]), chunks=tuple([min(chunk_size, n_test_images),]+self.image_size[1:]), dtype='uint8')        

        zipfile_object = zipfile.ZipFile(source_dir+ 'img_align_celeba.zip', 'r')

        training_index = 0
        validation_index = 0
        test_index = 0
        print('Filling the h5 file with images.')
        start_indiv = time.time()
        for i, file in enumerate(zipfile_object.infolist()):
            if i > 0:
                if (i-1) % (len(filenames)//20) == 0: 
                    end_indiv = time.time()
                    print(i-1, len(filenames), end_indiv-start_indiv)
                    hf.flush()
                    gc.collect(); gc.collect()
                    start_indiv = time.time()

                assert (file.filename[file.filename.find('/')+1:] == filenames[i-1])
                zipfile_object.extract(file, source_dir)
                image_file_path = source_dir + 'img_align_celeba/' + filenames[i-1]
                raw_image = np.asarray(Image.open(image_file_path))
                cropped_image = self.crop_raw_image(raw_image)
                downsampled_image = self.downsample_image(cropped_image, self.image_size[1:3])
                                  
                if training_mask[i-1]:
                    training_images_data[training_index, ...] = downsampled_image
                    training_index += 1
                elif validation_mask[i-1]:
                    validation_images_data[validation_index, ...] = downsampled_image
                    validation_index += 1
                elif test_mask[i-1]:
                    test_images_data[test_index, ...] = downsampled_image
                    test_index += 1
            else:
                zipfile_object.extract(file, source_dir)

            if i % 100 == 0:
                for path in glob.glob(source_dir + 'img_align_celeba/*'): os.remove(path)
        for path in glob.glob(source_dir + 'img_align_celeba/*'): os.remove(path)
        os.rmdir(source_dir + 'img_align_celeba/')

        hf.flush()
        hf.close()
        gc.collect(); gc.collect()

    def report_status(self):
        print('\n')
        print('################  DataLoader  ####################')
        print('Dataset name: ' + str(self.dataset_name))
        print('Overall batch size: ' + str(self.batch_size))
        print('Image size: ' + str(self.image_size[1:]))
        print('Attribute size: '+str(self.attribute_size[1:]))
        print('Stage: ' + str(self.stage))
        print('Buffer order randomized: ' + str(self.buffer_order_randomized))
        print('In-buffer order randomized: ' + str(self.in_buffer_order_randomized))
        print('Buffer size: ' + str(self.batch_size*self.buffer_batch_ratio))
        print('# Buffers: ' + str(len(self.curr_buffer_order)))
        print('# Batches: ' + str(self.curr_max_iter))
        print('# Samples: ' + str(self.curr_n_samples))
        print('##################################################')
        print('\n')

    def setup(self, stage, randomized=False, verbose=True):
        assert (stage == 'Training' or stage == 'Validation' or stage == 'Test')
        self.stage = stage

        if self.stage == 'Training':
            self.curr_images = self.training_images
            self.curr_attributes = self.training_attributes
        elif self.stage == 'Validation':
            self.curr_images = self.validation_images
            self.curr_attributes = self.validation_attributes
        elif self.stage == 'Test':
            self.curr_images = self.test_images
            self.curr_attributes = self.test_attributes

        self.curr_n_samples = len(self.curr_images)
        self.curr_max_iter = np.int(np.ceil(float(self.curr_n_samples)/float(self.batch_size)))
        self.buffer_order_randomized = randomized
        self.in_buffer_order_randomized = randomized

        n_buffers_float = float(self.curr_n_samples)/float(self.batch_size*self.buffer_batch_ratio)
        n_full_buffers = int(np.floor(n_buffers_float))
        n_all_buffers = int(np.ceil(n_buffers_float))

        self.curr_buffer_order = np.arange(n_full_buffers)
        if self.buffer_order_randomized: self.curr_buffer_order = np.random.permutation(self.curr_buffer_order)
        if n_all_buffers != n_full_buffers: self.curr_buffer_order = np.concatenate([self.curr_buffer_order, [n_all_buffers-1,]], axis=0)

        if verbose: self.report_status()
        self.reset()

    def fill_buffer(self):
        ith_buffer = self.iter//self.buffer_batch_ratio
        self.curr_buffer_index = self.curr_buffer_order[ith_buffer]
        start_data_index = self.curr_buffer_index*self.buffer_batch_ratio*self.batch_size
        end_data_index = min(start_data_index+self.batch_size*self.buffer_batch_ratio, self.curr_n_samples)
        assert (end_data_index > start_data_index)

        self.curr_valid_buffer_size = end_data_index-start_data_index        
        self.buffer_images[:self.curr_valid_buffer_size, ...] = self.curr_images[start_data_index:end_data_index, ...]
        self.buffer_images[:self.curr_valid_buffer_size, ...] = self.buffer_images[:self.curr_valid_buffer_size, ...]/255.
        self.buffer_attributes[:self.curr_valid_buffer_size, ...] = self.curr_attributes[start_data_index:end_data_index, ...]

        self.curr_in_buffer_order = np.arange(self.buffer_images.shape[0])
        if self.in_buffer_order_randomized:            
            self.curr_in_buffer_order[:self.curr_valid_buffer_size] = np.random.permutation(self.curr_in_buffer_order[:self.curr_valid_buffer_size])
            self.buffer_images = self.buffer_images[self.curr_in_buffer_order, ...]
            self.buffer_attributes = self.buffer_attributes[self.curr_in_buffer_order, ...]

    def reset(self):
        self.iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter == self.curr_max_iter: raise StopIteration
        if self.iter % self.buffer_batch_ratio == 0: 
            self.fill_buffer()

        in_buffer_ith_batch = self.iter % self.buffer_batch_ratio
        curr_start_ind = in_buffer_ith_batch*self.batch_size
        curr_end_ind = min(curr_start_ind+self.batch_size, self.curr_valid_buffer_size)
        curr_batch_size = curr_end_ind-curr_start_ind

        batch_dict = {'Image': self.buffer_images[curr_start_ind:curr_end_ind, ...],
                      'Attributes': self.buffer_attributes[curr_start_ind:curr_end_ind, ...]}
        self.iter += 1
        return self.iter-1, curr_batch_size, batch_dict



