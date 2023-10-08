#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom Tensorflow generators.
"""
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.set_soft_device_placement(True)
from tensorflow.keras.utils import Sequence

LOCAL_SEED = None

def set_generators_seed(seed):
	globals()['LOCAL_SEED'] = seed


rankdata = lambda x: 1.+np.argsort(np.argsort(x, axis=0), axis=0)
class CopulaBatchGenerator(Sequence):
	''' 
	Random batch generator of maximum-entropy copula learning.
	'''
	def __init__(self, z, batch_size=1000, steps_per_epoch=100):
		self.batch_size = batch_size
		self.d = z.shape[1]
		self.n = z.shape[0]
		self.z = z
		self.steps_per_epoch = steps_per_epoch
		self.emp_u = rankdata(self.z)/(self.n + 1.)
		self.emp_u[np.isnan(self.z)] = 0.5
		self.rnd_gen = np.random.default_rng(LOCAL_SEED)

		if self.n < 200*self.d:
			dn = 200*self.d - self.n
			selected_rows = self.rnd_gen.choice(self.n, dn, replace=True)
			emp_u = self.emp_u[selected_rows, :].copy()
			scale = 1./(100.*self.n)
			emp_u += (scale*self.rnd_gen.uniform(size=emp_u.shape) - 0.5*scale)
			self.emp_u = np.concatenate([self.emp_u, emp_u], axis=0)
			self.n = self.emp_u.shape[0]

		self.batch_selector = self.rnd_gen.choice(self.n, self.batch_size*self.steps_per_epoch, replace=True)
		self.batch_selector = self.batch_selector.reshape((self.steps_per_epoch, self.batch_size))


	def getitem_ndarray(self, idx):
		''' '''
		i = idx % self.steps_per_epoch
		selected_rows = self.batch_selector[i]
		emp_u_ = self.emp_u[selected_rows, :]
		z_p = emp_u_.copy()
		z_q = self.rnd_gen.uniform(size=emp_u_.shape)

		z = np.empty((self.batch_size, self.d, 2))
		z[:, :, 0] = z_p
		z[:, :, 1] = z_q
		batch_x = z
		batch_y = np.ones((self.batch_size, 2))  # Not used  
		return batch_x, batch_y


	def __getitem__(self, idx):
		''' '''
		batch_x, batch_y = self.getitem_ndarray(idx)
		return tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)


	def __len__(self):
		return self.steps_per_epoch
