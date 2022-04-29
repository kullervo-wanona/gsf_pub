from __future__ import print_function
from IPython.core.debugger import Pdb
pdb = Pdb()
trace = pdb.set_trace

import math
import scipy as scipy
import copy

import numpy as np 
np.set_printoptions(suppress=True)

import helper

class NormalDistribution():
	def __init__(self, name='NormalDistribution'):
		self.name = name

	@staticmethod
	def param_info(mode='names'):
		assert (mode == 'names' or mode == 'rates')
		if mode == 'names': return ['mean', 'std']
		if mode == 'rates': return [[1, 0], [1, 0]]

	@staticmethod
	def num_pre_params(num_dim):
		return sum([rate[0]*num_dim+rate[1] for rate in GaussianDistribution.param_info('rates')])

	def set_params_dict(self):
			self.params_dict = {}

			assert ('mean_mode' in self.params_config)
			assert (self.params_config['mean_mode'] == 'Identity' or self.params_config['mean_mode'] == 'SimpleBounded')
			if self.params_config['mean_mode'] == 'Identity':
				self.params_dict['mean'] = self.pre_params_dict['pre_mean']
			elif self.params_config['mean_mode'] == 'SimpleBounded':
				if type(self.params_config['min_std']) == float and type(self.params_config['max_std']) == float:
					assert (self.params_config['max_std'] > self.params_config['min_std'])
				self.params_dict['mean'] = self.params_config['min_mean']+ \
					tf.nn.sigmoid(self.pre_params_dict['pre_mean'])*(self.params_config['max_mean']-self.params_config['min_mean'])

			assert ('std_mode' in self.params_config)
			assert (self.params_config['std_mode'] == 'Identity' or self.params_config['std_mode'] == 'Exponential' or self.params_config['std_mode'] == 'Softplus' or self.params_config['std_mode'] == 'SimpleBounded')
			if self.params_config['std_mode'] == 'Identity':
				self.params_dict['std'] = self.pre_params_dict['pre_std']
			if self.params_config['std_mode'] == 'Exponential':
				self.params_dict['std'] = 1e-7+tf.exp(self.pre_params_dict['pre_std'])
			elif self.params_config['std_mode'] == 'Softplus':
				self.params_dict['std'] = 1e-7+tf.nn.softplus(self.pre_params_dict['pre_std'])/np.log(2)
			elif self.params_config['std_mode'] == 'SimpleBounded':
				if type(self.params_config['min_std']) == float and type(self.params_config['max_std']) == float:
					assert (self.params_config['min_std'] > 1e-7 and self.params_config['max_std'] > self.params_config['min_std'])
				self.params_dict['std'] = self.params_config['min_std']+ \
					tf.nn.sigmoid(self.pre_params_dict['pre_std'])*(self.params_config['max_std']-self.params_config['min_std'])

			self.sample_dim = self.params_dict['mean'].get_shape()[1].value

	def get_statistic(self, stat_name):
		stat_name = stat_name.lower()
		assert (stat_name == 'mean' or stat_name == 'std' or stat_name == 'variance' or stat_name == 'mode' or stat_name == 'entropy')
		
		if stat_name == 'mean' or stat_name == 'mode': # [None, dim]
			return self.params_dict['mean']
		elif stat_name == 'std': # [None, dim]
			return self.params_dict['std']
		elif stat_name == 'variance': # [None, dim]
			return self.params_dict['std']**2
		elif stat_name == 'entropy': # [None, dim], since independent, overall entropy is sum(entropy_per_dim)
			return 0.5*(1+np.log(2*np.pi)+2*helper.safe_log_tf(self.params_dict['std']))

	def sample(self):
		epsilon = tf_compat_v1.random.normal(shape=tf.shape(self.params_dict['mean']), dtype=tf.float32)
		sample = (self.params_dict['std']*epsilon)+self.params_dict['mean']
		return sample

	def log_pdf(self, sample):
		assert (len(sample.get_shape()) == 2)
		n_dim = self.params_dict['mean'].get_shape()[-1].value
		residual = (self.params_dict['mean']-sample)/(1e-7+self.params_dict['std'])
		unnormalized_log_prob = -0.5*tf.reduce_sum(residual**2, axis=1, keepdims=True)
		log_partition = -tf.reduce_sum(helper.safe_log_tf(self.params_dict['std']), axis=1, keepdims=True)-(n_dim/2.0)*np.log(2*np.pi)
		log_prob = unnormalized_log_prob+log_partition
		return log_prob


