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
from urllib.request import Request, urlopen
import lmdb
import requests

import numpy as np
np.set_printoptions(suppress=True)

class DataLoader:
    def __init__(self, batch_size):
        self.dataset_name = 'ImageNet'
        self.batch_size = batch_size
        self.buffer_batch_ratio = 500
        self.iter = 0
        self.image_size = [-1, 64, 64, 3]
        self.dataset_dir = str(Path.home())+'/Datasets/ImageNet64/'
        # try: self.load_h5()
        # except: self.create_h5(); self.load_h5()
        self.create_h5();
        self.setup('Training')
        self.setup_buffers()
    
    def setup_buffers(self):
        self.buffer_images = np.zeros([self.batch_size*self.buffer_batch_ratio,] + self.image_size[1:], np.float32)

    def load_h5(self):
        processed_file_path = self.dataset_dir + 'all_imagenet_64.h5'
        print('Loading from h5 file: '+ processed_file_path)
        start_indiv = time.time()

        hf = h5py.File(processed_file_path, 'r')
        self.training_images = hf['training_images']
        end_indiv = time.time()
        print('Success loading data from h5 file. Time: '+ str(end_indiv-start_indiv))

    def crop_raw_image(self, raw_image):
        crop_shape = min(raw_image.shape[0], raw_image.shape[1])
        crop_image_size = [crop_shape, crop_shape]        
        start_x, start_y = raw_image.shape[0]//2-crop_image_size[0]//2, raw_image.shape[1]//2-crop_image_size[1]//2
        # print('start_x, start_y', start_x, start_y)
        return raw_image[start_x:start_x+crop_image_size[0], start_y:start_y+crop_image_size[1], ...]

    def downsample_image(self, raw_image, downsample_image_size, downsample_method='cubic'):
        assert (downsample_method == 'nearest' or downsample_method == 'bilinear' or \
            downsample_method == 'bicubic' or downsample_method == 'cubic' or downsample_method == 'lanczos')
        return scipy.misc.imresize(raw_image, (downsample_image_size[1], downsample_image_size[2]), interp=downsample_method)

    def create_h5(self, chunk_size=50000):
        if not os.path.exists(self.dataset_dir): os.makedirs(self.dataset_dir)
        
        source_dir = str(Path.home())+'/DataSources/ImageNet/'
        class_names_dict = {}
        with open(source_dir + 'imagenet_urls.csv', newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter='\\', quotechar='|')
            n_images = 0
            n_unique_classes = 0
            for i, row in enumerate(rows): 
                assert (len(row) == 3)
                url, class_name, word_net_id = row
                url = url.strip()
                class_name = class_name.strip()
                word_net_id = word_net_id.strip()

                if class_name not in class_names_dict:
                    class_names_dict[class_name] = {'count': 1, 'wnid': word_net_id}
                    n_unique_classes += 1
                else:
                    assert (class_names_dict[class_name]['wnid'] == word_net_id)
                    class_names_dict[class_name]['count'] += 1
                n_images += 1
        
        wnid_list = []
        for class_name in class_names_dict:
            wnid_list.append(class_names_dict[class_name]['wnid'])
        wnid_list_sorted = sorted(wnid_list, key=lambda v: (v.upper(), v[0].islower()))
        assert (len(wnid_list_sorted) == n_unique_classes)

        for class_name in class_names_dict:
            class_names_dict[class_name]['class_ID'] = wnid_list_sorted.index(class_names_dict[class_name]['wnid'])

        processed_file_path = self.dataset_dir + 'all_imagenet_64.h5'
        print('Creating h5 file: ' + processed_file_path)
        hf = h5py.File(processed_file_path, 'w')

        training_images_data = hf.create_dataset('training_images', tuple([n_images] + self.image_size[1:]), chunks=tuple([min(chunk_size, n_images),] + self.image_size[1:]), dtype='uint8')
        training_labels_data = hf.create_dataset('training_labels', tuple([n_images, n_unique_classes]), chunks=tuple([min(chunk_size, n_images), 1]), dtype='bool')
        label_dummy = np.zeros((n_unique_classes,), np.bool)
        
        dummy_folder = self.dataset_dir + 'dummy_folder/'
        if not os.path.exists(dummy_folder): os.makedirs(dummy_folder)

        with open(source_dir + 'imagenet_urls.csv', newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter='\\', quotechar='|')
            for i, row in enumerate(rows): 
                if i % 10000 == 0:
                    print('Training set: Finished', i, 'images')
                    hf.flush()
                    gc.collect(); gc.collect()

                url, class_name, word_net_id = row
                url = url.strip()
                class_name = class_name.strip()

                label_dummy[class_names_dict[class_name]['class_ID']] = True
                training_labels_data[i, ...] = label_dummy
                label_dummy[class_names_dict[class_name]['class_ID']] = False

                try:
                    wget.download(url, dummy_folder)
                except:
                    requests.get(url)
                    trace()

                file_path = glob.glob(dummy_folder+'*')[0]

                # if url.find('.jpg') != -1:
                #     file_path = self.dataset_dir+url[-url[::-1].find('/'):url.find('.jpg')+4]                
                # if url.find('.png') != -1:
                #     file_path = self.dataset_dir+url[-url[::-1].find('/'):url.find('.png')+4]                
                raw_image = scipy.misc.imread(file_path)[..., :3]

                cropped_image = self.crop_raw_image(raw_image)
                downsampled_image = self.downsample_image(cropped_image, downsample_image_size=self.image_size)

                training_images_data[i, ...] = downsampled_image
                os.remove(file_path)

        os.rmdir(dummy_folder)

        trace()

        with open(source_dir + 'imagenet_urls.csv', newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i, row in enumerate(rows): 
                url, class_name, word_net_id = row
                wget.download(url, self.dataset_dir)
                file_path = self.dataset_dir+url[-url[::-1].find('/'):]
                os.remove(file_path)
                trace()

        if not os.path.exists(source_dir): 
            os.makedirs(source_dir)
            category = 'bedroom'
            for set_name in ['train', 'val', 'test']:
                if set_name == 'test':
                    url = 'http://dl.yf.io/lsun/scenes/test_lmdb.zip'
                    zip_name = 'test_lmdb.zip'
                else:
                    url = 'http://dl.yf.io/lsun/scenes/{category}_{set_name}_lmdb.zip'.format(**locals())
                    zip_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
                zip_path = source_dir + zip_name
                print('Downloading', category, set_name, 'set')
                print(url)
                subprocess.call(['curl', url, '-o', zip_path])


        if not os.path.exists(self.dataset_dir + 'bedroom_val_lmdb/'):
            print('Extracting the bedroom_val_lmdb database.')
            zipfile_object = zipfile.ZipFile(source_dir+ 'bedroom_val_lmdb.zip', 'r')
            zipfile_object.extractall(self.dataset_dir)
            zipfile_object.close()

        n_validation_images = None
        with lmdb.open(self.dataset_dir + 'bedroom_val_lmdb/', map_size=1099511627776, max_readers=100, readonly=True).begin(write=False) as txn:
            n_validation_images = txn.stat()['entries']

        validation_images_data = hf.create_dataset('validation_images', tuple([n_validation_images]+self.image_size[1:]), chunks=tuple([min(chunk_size, n_validation_images),]+self.image_size[1:]), dtype='uint8')
        
        with lmdb.open(self.dataset_dir + 'bedroom_val_lmdb/', map_size=1099511627776, max_readers=100, readonly=True).begin(write=False) as txn:
            i = 0
            for key, val in txn.cursor():
                if i % 10000 == 0:
                    print('Validation set: Finished', i, 'images')
                    hf.flush()
                    gc.collect(); gc.collect()

                with open(self.dataset_dir + 'temp.webp', 'wb') as fp: fp.write(val)
                raw_image = scipy.misc.imread(self.dataset_dir + 'temp.webp')
                cropped_image = self.crop_raw_image(raw_image, crop_image_size=[-1, 256, 256, 3])
                downsampled_image = self.downsample_image(cropped_image, downsample_image_size=self.image_size)
                validation_images_data[i, ...] = downsampled_image
                i += 1
        hf.flush()
        os.remove(self.dataset_dir + 'temp.webp')
        os.remove(self.dataset_dir + 'bedroom_val_lmdb/lock.mdb')
        os.remove(self.dataset_dir + 'bedroom_val_lmdb/data.mdb')
        os.rmdir(self.dataset_dir + 'bedroom_val_lmdb/')

        if not os.path.exists(self.dataset_dir + 'bedroom_train_lmdb/'):
            print('Extracting the bedroom_train_lmdb database.')
            zipfile_object = zipfile.ZipFile(source_dir+ 'bedroom_train_lmdb.zip', 'r')
            zipfile_object.extractall(self.dataset_dir)
            zipfile_object.close()
        
        n_training_images = None
        with lmdb.open(self.dataset_dir + 'bedroom_train_lmdb/', map_size=1099511627776, max_readers=100, readonly=True).begin(write=False) as txn:
            n_training_images = txn.stat()['entries']

        training_images_data = hf.create_dataset('training_images', tuple([n_training_images]+self.image_size[1:]), chunks=tuple([min(chunk_size, n_training_images),]+self.image_size[1:]), dtype='uint8')

        with lmdb.open(self.dataset_dir + 'bedroom_train_lmdb/', map_size=1099511627776, max_readers=100, readonly=True).begin(write=False) as txn:
            i = 0
            for key, val in txn.cursor():
                if i % 10000 == 0:
                    print('Training set: Finished saving', i, 'images')
                    hf.flush()
                    gc.collect(); gc.collect()

                with open(self.dataset_dir + 'temp.webp', 'wb') as fp: fp.write(val)
                raw_image = scipy.misc.imread(self.dataset_dir + 'temp.webp')
                cropped_image = self.crop_raw_image(raw_image, crop_image_size=[-1, 256, 256, 3])
                downsampled_image = self.downsample_image(cropped_image, downsample_image_size=self.image_size)
                training_images_data[i, ...] = downsampled_image
                i += 1
        hf.flush()
        os.remove(self.dataset_dir + 'temp.webp')
        os.remove(self.dataset_dir + 'bedroom_train_lmdb/lock.mdb')
        os.remove(self.dataset_dir + 'bedroom_train_lmdb/data.mdb')
        os.rmdir(self.dataset_dir + 'bedroom_train_lmdb/')

        hf.close()
        gc.collect(); gc.collect()

    def report_status(self):
        print('\n')
        print('################  DataLoader  ####################')
        print('Dataset name: ' + str(self.dataset_name))
        print('Overall batch size: ' + str(self.batch_size))
        print('Image size: ' + str(self.image_size[1:]))
        print('Stage: ' + str(self.stage))
        print('Buffer order randomized: ' + str(self.buffer_order_randomize))
        print('In-buffer order randomized: ' + str(self.in_buffer_order_randomize))
        print('Buffer size: ' + str(self.batch_size*self.buffer_batch_ratio))
        print('# Buffers: ' + str(len(self.curr_buffer_order)))
        print('# Batches: ' + str(self.curr_max_iter))
        print('# Samples: ' + str(self.curr_n_samples))
        print('##################################################')
        print('\n')


    def reset(self):
        self.iter = 0

    def __iter__(self):
        return self

    def setup(self, stage, randomize=False):
        assert (stage == 'Training' or stage == 'Test')
        self.stage = stage

        if self.stage == 'Training':
            self.curr_images = self.training_images
        elif self.stage == 'Test':
            self.curr_images = self.training_images

        self.curr_n_samples = len(self.curr_images)
        self.curr_max_iter = np.int(np.ceil(float(self.curr_n_samples)/float(self.batch_size)))

        self.buffer_order_randomize = randomize
        self.in_buffer_order_randomize = randomize

        n_buffers_float = float(self.curr_n_samples)/float(self.batch_size*self.buffer_batch_ratio)
        n_full_buffers = int(np.floor(n_buffers_float))
        n_all_buffers = int(np.ceil(n_buffers_float))

        self.curr_buffer_order = np.arange(n_full_buffers)
        if self.buffer_order_randomize: self.curr_buffer_order = np.random.permutation(self.curr_buffer_order)
        if n_all_buffers != n_full_buffers: self.curr_buffer_order = np.concatenate([self.curr_buffer_order, [n_all_buffers-1,]], axis=0)

        self.report_status()
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
        if self.in_buffer_order_randomize:            
            self.curr_in_buffer_order[:self.curr_valid_buffer_size] = np.random.permutation(self.curr_in_buffer_order[:self.curr_valid_buffer_size])
            self.buffer_images = self.buffer_images[self.curr_in_buffer_order, ...]

    def __next__(self):
        if self.iter == self.curr_max_iter: raise StopIteration
        if self.iter % self.buffer_batch_ratio == 0: self.fill_buffer()

        in_buffer_ith_batch = self.iter % self.buffer_batch_ratio
        curr_start_ind = in_buffer_ith_batch*self.batch_size
        curr_end_ind = min(curr_start_ind+self.batch_size, self.curr_valid_buffer_size)
        curr_batch_size = curr_end_ind-curr_start_ind

        batch_dict = {'Image': self.buffer_images[curr_start_ind:curr_end_ind, ...]}
        self.iter += 1
        return self.iter-1, curr_batch_size, batch_dict

data_loader = DataLoader(100)
# print(data_loader.curr_buffer_order)
# data_loader.setup('Test', randomize=False)
# print(data_loader.curr_buffer_order)

# t_start = time.time()
# i = 0
# for _, _, batch_dict in data_loader:   
#     if i % 1000 == 0: print(i)
#     i += 1

# t_end = time.time()
# print('Overall time:', t_end-t_start)
# trace()











# examples_dir = str(Path.home())+'/LSUN_bedroom_examples/cubic/'
# if not os.path.exists(examples_dir): os.makedirs(examples_dir)
# for i in range(20):
#     scipy.misc.toimage(batch_dict['Image'][10+i]).save(examples_dir+'/'+str(i)+'.png')
# trace()
