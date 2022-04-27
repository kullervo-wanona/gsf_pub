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
import zipfile
from shutil import copyfile
import csv
import gc
import subprocess
# from urllib.request import Request, urlopen
import lmdb
from PIL import Image

import numpy as np
np.set_printoptions(suppress=True)

class DataLoader:
    def __init__(self, batch_size):
        self.dataset_name = 'LSUN Bedroom 64'
        self.batch_size = batch_size
        self.buffer_batch_ratio = 500
        self.iter = 0
        self.image_size = [-1, 64, 64, 3]
        self.dataset_dir = str(Path.home())+'/Datasets/LSUN/Bedroom/LSUNBedroom64/'
        try: self.load_h5()
        except: self.create_h5(); self.load_h5()
        self.setup('Training')
        self.setup_buffers()

    def setup_buffers(self):
        self.buffer_images = np.zeros([self.batch_size*self.buffer_batch_ratio,] + self.image_size[1:], np.float32)

    def load_h5(self):
        processed_file_path = self.dataset_dir + 'lsun_bedroom_64.h5'
        print('Loading from h5 file: '+ processed_file_path)
        start_indiv = time.time()

        hf = h5py.File(processed_file_path, 'r')
        self.training_images = hf['training_images']
        self.test_images = hf['test_images']
        end_indiv = time.time()
        print('Success loading data from h5 file. Time: '+ str(end_indiv-start_indiv))

    def crop_raw_image(self, raw_image, crop_image_size=[256, 256]):
        # https://github.com/igul222/improved_wgan_training/issues/57
        start_x, start_y = raw_image.shape[0]//2-crop_image_size[0]//2, raw_image.shape[1]//2-crop_image_size[1]//2
        return raw_image[start_x:start_x+crop_image_size[0], start_y:start_y+crop_image_size[1], ...]

    def downsample_image(self, raw_image, downsample_image_size, downsample_method='cubic'):
        assert (downsample_method == 'nearest' or downsample_method == 'bilinear' or \
            downsample_method == 'bicubic' or downsample_method == 'cubic' or downsample_method == 'lanczos')
        return scipy.misc.imresize(raw_image, (downsample_image_size[1], downsample_image_size[2]), interp=downsample_method)

    def create_h5(self, chunk_size=20000, test_percentage = 0.2, remove_extracted_files=False):
        if not os.path.exists(self.dataset_dir): os.makedirs(self.dataset_dir)
        
        source_dir = str(Path.home())+'/DataSources/LSUN/Bedroom/'
        if not os.path.exists(source_dir): os.makedirs(source_dir)
        category = 'bedroom'
        for set_name in ['train', 'val']:
            url = 'http://dl.yf.io/lsun/scenes/{category}_{set_name}_lmdb.zip'.format(**locals())
            zip_name = '{category}_{set_name}_lmdb'.format(**locals())
            zip_path = source_dir + zip_name + '.zip'
            if (not os.path.exists(source_dir + zip_name + '/')) and (not os.path.exists(zip_path)):
                print('Downloading', category, set_name, 'set')
                print(url)
                subprocess.call(['curl', url, '-o', zip_path])

        processed_file_path = self.dataset_dir + 'lsun_bedroom_64.h5'
        print('Creating h5 file: ' + processed_file_path)
        hf = h5py.File(processed_file_path, 'w')

        if not os.path.exists(source_dir + 'bedroom_val_lmdb/'):
            print('Extracting the bedroom_val_lmdb database.')
            zipfile_object = zipfile.ZipFile(source_dir + 'bedroom_val_lmdb.zip', 'r')
            zipfile_object.extractall(source_dir)
            zipfile_object.close()

        n_validation_images = None
        with lmdb.open(source_dir + 'bedroom_val_lmdb/', map_size=1099511627776, max_readers=100, readonly=True).begin(write=False) as txn:
            n_validation_images = txn.stat()['entries']
        validation_images_data = hf.create_dataset('validation_images', tuple([n_validation_images]+self.image_size[1:]), chunks=tuple([min(chunk_size, n_validation_images),]+self.image_size[1:]), dtype='uint8')
        
        print('Filling the h5 file with validation images.')
        start_indiv = time.time()
        with lmdb.open(source_dir + 'bedroom_val_lmdb/', map_size=1099511627776, max_readers=100, readonly=True).begin(write=False) as txn:
            i = 0
            for key, val in txn.cursor():
                if i % (n_validation_images//20) == 0: 
                    end_indiv = time.time()
                    print(i, n_validation_images, end_indiv-start_indiv)
                    hf.flush()
                    gc.collect(); gc.collect()
                    start_indiv = time.time()

                image_file_path = source_dir + 'temp.webp'
                with open(image_file_path, 'wb') as fp: fp.write(val)
                raw_image = np.asarray(Image.open(image_file_path))
                cropped_image = self.crop_raw_image(raw_image)
                downsampled_image = self.downsample_image(cropped_image, downsample_image_size=self.image_size)
                validation_images_data[i, ...] = downsampled_image
                i += 1
        hf.flush()
        gc.collect(); gc.collect()
        os.remove(source_dir + 'temp.webp')
        if remove_extracted_files:
            os.remove(source_dir + 'bedroom_val_lmdb/lock.mdb')
            os.remove(source_dir + 'bedroom_val_lmdb/data.mdb')
            os.rmdir(source_dir + 'bedroom_val_lmdb/')

        if not os.path.exists(source_dir + 'bedroom_train_lmdb/'):
            print('Extracting the bedroom_train_lmdb database. May take ~10 mins.')
            zipfile_object = zipfile.ZipFile(source_dir+ 'bedroom_train_lmdb.zip', 'r')
            zipfile_object.extractall(source_dir)
            zipfile_object.close()
        
        n_training_images_in_database = None
        n_training_images = 0
        n_test_images = 0
        with lmdb.open(source_dir + 'bedroom_train_lmdb/', map_size=1099511627776, max_readers=100, readonly=True).begin(write=False) as txn:
            n_training_images_in_database = txn.stat()['entries']

            if not os.path.exists(source_dir + 'test_split.txt'): 
                print('Test split file does not exist. Creating test split file at: ' + source_dir + 'test_split.txt')
                test_split_file_object = open(source_dir + 'test_split.txt', 'w') 
                start_indiv = time.time()
                i = 0
                all_str = ''
                for key, val in txn.cursor():
                    if i % (n_training_images_in_database//20) == 0: 
                        end_indiv = time.time()
                        print(i, n_training_images_in_database, end_indiv-start_indiv)
                        start_indiv = time.time()
                        gc.collect(); gc.collect()

                    test_sample_indicator = int(np.random.random(1)[0] < test_percentage)
                    if test_sample_indicator: n_test_images += 1
                    else: n_training_images += 1
                    write_str = key.decode("utf-8") + ', ' + str(test_sample_indicator) + '\n'
                    all_str += write_str
                    i += 1
                print('Writing to the test split file.')
                test_split_file_object.write(all_str)
                test_split_file_object.close()
            
        print('Loading test split file.')
        test_split_file_object = open(source_dir + 'test_split.txt', 'r')
        test_split_lines = test_split_file_object.readlines()
        test_split_file_object.close()

        test_split_lines = [e.strip().split(', ') for e in test_split_lines]
        test_split_keys = [e[0] for e in test_split_lines]
        test_mask = np.asarray([(e[1] == '1') for e in test_split_lines])
        test_split_lines = None
        assert (n_training_images_in_database == len(test_split_keys))
        assert (n_training_images_in_database == len(test_mask))
        gc.collect(); gc.collect()
        n_test_images = test_mask.sum()
        n_training_images = (test_mask == False).sum()

        training_images_data = hf.create_dataset('training_images', tuple([n_training_images]+self.image_size[1:]), chunks=tuple([min(chunk_size, n_training_images),]+self.image_size[1:]), dtype='uint8')
        test_images_data = hf.create_dataset('test_images', tuple([n_test_images]+self.image_size[1:]), chunks=tuple([min(chunk_size, n_test_images),]+self.image_size[1:]), dtype='uint8')

        print('Filling the h5 file with training and test images.')
        start_indiv = time.time()
        training_index = 0
        test_index = 0
        with lmdb.open(source_dir + 'bedroom_train_lmdb/', map_size=1099511627776, max_readers=100, readonly=True).begin(write=False) as txn:
            i = 0
            for key, val in txn.cursor():                
                if i % (n_training_images_in_database//300) == 0: 
                    end_indiv = time.time()
                    print(i, n_training_images_in_database, end_indiv-start_indiv)
                    hf.flush()
                    gc.collect(); gc.collect()
                    start_indiv = time.time()
                assert (key.decode("utf-8") == test_split_keys[i])

                image_file_path = source_dir + 'temp.webp'
                with open(image_file_path, 'wb') as fp: fp.write(val)
                raw_image = np.asarray(Image.open(image_file_path))
                cropped_image = self.crop_raw_image(raw_image)
                downsampled_image = self.downsample_image(cropped_image, downsample_image_size=self.image_size)

                if test_mask[i]:
                    test_images_data[test_index, ...] = downsampled_image
                    test_index += 1
                else:
                    training_images_data[training_index, ...] = downsampled_image
                    training_index += 1
                i += 1
        hf.flush()
        hf.close()
        gc.collect(); gc.collect()
        os.remove(source_dir + 'temp.webp')
        if remove_extracted_files:
            os.remove(source_dir + 'bedroom_train_lmdb/lock.mdb')
            os.remove(source_dir + 'bedroom_train_lmdb/data.mdb')
            os.rmdir(source_dir + 'bedroom_train_lmdb/')
        
    def report_status(self):
        print('\n')
        print('################  DataLoader  ####################')
        print('Dataset name: ' + str(self.dataset_name))
        print('Overall batch size: ' + str(self.batch_size))
        print('Image size: ' + str(self.image_size[1:]))
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
        assert (stage == 'Training' or stage == 'Test')
        self.stage = stage

        if self.stage == 'Training':
            self.curr_images = self.training_images
        elif self.stage == 'Test':
            self.curr_images = self.test_images

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

        self.curr_in_buffer_order = np.arange(self.buffer_images.shape[0])
        if self.in_buffer_order_randomized:            
            self.curr_in_buffer_order[:self.curr_valid_buffer_size] = np.random.permutation(self.curr_in_buffer_order[:self.curr_valid_buffer_size])
            self.buffer_images = self.buffer_images[self.curr_in_buffer_order, ...]

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

        batch_dict = {'Image': self.buffer_images[curr_start_ind:curr_end_ind, ...]}
        self.iter += 1
        return self.iter-1, curr_batch_size, batch_dict
