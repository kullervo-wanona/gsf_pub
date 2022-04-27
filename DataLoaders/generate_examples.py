from __future__ import print_function
from IPython.core.debugger import Pdb
pdb = Pdb()
trace = pdb.set_trace

import os
import time
from pathlib import Path
import scipy.misc
from PIL import Image

import numpy as np
np.set_printoptions(suppress=True)

batch_size = 100
loader_names = [\
				# 'MNISTLoader', 
				# 'BinaryMNISTLoader', 
				# 'ColorMNISTLoader', 
				# 'Cifar10Loader', 
				# 'Cifar100Loader',
				# 'TinyImageNetLoader', 
				# 'CelebA32Loader', 
				# 'CelebA64Loader', 
				'CelebA128Loader', 
				# 'CelebA178Loader', 
				# 'LSUNBedroom64Loader', 
				# 'LSUNBedroom128Loader', 
				# 'LSUNBedroom256Loader', 
			   ]

for loader_name in loader_names:
	if loader_name == 'MNISTLoader':
		#################### REGULAR MNIST ####################
		print('\n\n\n\n\n######################    MNISTLoader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/MNIST/MNIST/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from MNIST.MNISTLoader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		
		for i in range(batch['Image'].shape[0]): 
		    scipy.misc.toimage(np.tile(batch['Image'][i], [1, 1, 3])).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    scipy.misc.toimage(np.tile(batch['Image'][i], [1, 1, 3])).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    scipy.misc.toimage(np.tile(batch['Image'][i], [1, 1, 3])).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    scipy.misc.toimage(np.tile(batch['Image'][i], [1, 1, 3])).save(examples_dir+'Test/First/'+str(i)+'.png')


	if loader_name == 'BinaryMNISTLoader':
		#################### BINARY MNIST ####################
		print('\n\n\n\n\n######################    BinaryMNISTLoader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/MNIST/BinaryMNIST/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from MNIST.BinaryMNISTLoader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    scipy.misc.toimage(np.tile(batch['Image'][i], [1, 1, 3])).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    scipy.misc.toimage(np.tile(batch['Image'][i], [1, 1, 3])).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    scipy.misc.toimage(np.tile(batch['Image'][i], [1, 1, 3])).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
			scipy.misc.toimage(np.tile(batch['Image'][i], [1, 1, 3])).save(examples_dir+'Test/First/'+str(i)+'.png')


	if loader_name == 'ColorMNISTLoader':
		#################### COLOR MNIST ####################
		print('\n\n\n\n\n######################    ColorMNISTLoader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/MNIST/ColorMNIST/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from MNIST.ColorMNISTLoader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		   	Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')


	if loader_name == 'Cifar10Loader':
		#################### CIFAR 10 ####################
		print('\n\n\n\n\n######################    Cifar10Loader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/CIFAR/Cifar10/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from CIFAR.Cifar10Loader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')


	if loader_name == 'Cifar100Loader':
		#################### CIFAR 100 ####################
		print('\n\n\n\n\n######################    Cifar100Loader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/CIFAR/Cifar100/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from CIFAR.Cifar100Loader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')


	if loader_name == 'TinyImageNetLoader':
		#################### Tiny ImageNet ####################
		print('\n\n\n\n\n######################    TinyImageNetLoader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/TinyImagenet/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from TinyImageNet.TinyImageNetLoader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')


	if loader_name == 'CelebA32Loader':
		#################### CelebA 32 ####################
		print('\n\n\n\n\n######################    CelebA32Loader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/CelebA/CelebA32/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from CelebA.CelebA32Loader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')


	if loader_name == 'CelebA64Loader':
		#################### CelebA 64 ####################
		print('\n\n\n\n\n######################    CelebA64Loader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/CelebA/CelebA64/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from CelebA.CelebA64Loader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
			Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
		    i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
			Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')


	if loader_name == 'CelebA128Loader':
		#################### CelebA 128 ####################
		print('\n\n\n\n\n######################    CelebA128Loader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/CelebA/CelebA128/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from CelebA.CelebA128Loader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
			if i % (data_loader.curr_max_iter//5) == 0: 
				t_end = time.time()
				print(i, data_loader.curr_max_iter, t_end-t_start)
			i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
			if i % (data_loader.curr_max_iter//5) == 0: 
				t_end = time.time()
				print(i, data_loader.curr_max_iter, t_end-t_start)
			i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')


	if loader_name == 'CelebA178Loader':
		################### CelebA 178 ####################
		print('\n\n\n\n\n######################    CelebA178Loader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/CelebA/CelebA178/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from CelebA.CelebA178Loader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
			if i % (data_loader.curr_max_iter//5) == 0: 
				t_end = time.time()
				print(i, data_loader.curr_max_iter, t_end-t_start)
			i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
			if i % (data_loader.curr_max_iter//5) == 0: 
				t_end = time.time()
				print(i, data_loader.curr_max_iter, t_end-t_start)
			i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')

	if loader_name == 'LSUNBedroom64Loader':
		################### LSUN Bedrooms 64 ####################
		print('\n\n\n\n\n######################    LSUNBedroom64Loader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/LSUN/Bedroom/LSUNBedroom64/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from LSUN.Bedroom.LSUNBedroom64Loader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
			if i % (data_loader.curr_max_iter//5) == 0: 
				t_end = time.time()
				print(i, data_loader.curr_max_iter, t_end-t_start)
			i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
			if i % (data_loader.curr_max_iter//5) == 0: 
				t_end = time.time()
				print(i, data_loader.curr_max_iter, t_end-t_start)
			i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')

	if loader_name == 'LSUNBedroom128Loader':
		################### LSUN Bedrooms 128 ####################
		print('\n\n\n\n\n######################    LSUNBedroom128Loader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/LSUN/Bedroom/LSUNBedroom128/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from LSUN.Bedroom.LSUNBedroom128Loader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
			if i % (data_loader.curr_max_iter//5) == 0: 
				t_end = time.time()
				print(i, data_loader.curr_max_iter, t_end-t_start)
			i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
			if i % (data_loader.curr_max_iter//5) == 0: 
				t_end = time.time()
				print(i, data_loader.curr_max_iter, t_end-t_start)
			i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')

		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')

	if loader_name == 'LSUNBedroom256Loader':
		################### LSUN Bedrooms 256 ####################
		print('\n\n\n\n\n######################    LSUNBedroom256Loader Test    ######################')
		examples_dir = str(Path.home())+'/Datasets/Examples/LSUN/Bedroom/LSUNBedroom256/'
		if not os.path.exists(examples_dir+'Training/First/'): os.makedirs(examples_dir+'Training/First/')
		if not os.path.exists(examples_dir+'Training/Last/'): os.makedirs(examples_dir+'Training/Last/')
		if not os.path.exists(examples_dir+'Test/First/'): os.makedirs(examples_dir+'Test/First/')
		if not os.path.exists(examples_dir+'Test/Last/'): os.makedirs(examples_dir+'Test/Last/')

		from LSUN.Bedroom.LSUNBedroom256Loader import DataLoader
		data_loader = DataLoader(batch_size)

		data_loader.setup('Training', randomized=True)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
			if i % (data_loader.curr_max_iter//5) == 0: 
				t_end = time.time()
				print(i, data_loader.curr_max_iter, t_end-t_start)
			i += 1
		t_end = time.time()
		print('Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/Last/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		t_start = time.time()
		i = 0
		for _, _, batch in data_loader:
			if i % (data_loader.curr_max_iter//5) == 0: 
				t_end = time.time()
				print(i, data_loader.curr_max_iter, t_end-t_start)
			i += 1
		t_end = time.time()
		print('Non-Randomized overall fetch time:', t_end-t_start, '# batches:', i)
		print('\n\n\n\n\n')
		
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/Last/'+str(i)+'.png')

		data_loader.setup('Training', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Training/First/'+str(i)+'.png')

		data_loader.setup('Test', randomized=False)
		_, _, batch = next(data_loader)
		for i in range(batch['Image'].shape[0]): 
		    Image.fromarray((np.clip(batch['Image'][i], 0, 1)*255.).astype(np.uint8)).save(examples_dir+'Test/First/'+str(i)+'.png')
# trace()
