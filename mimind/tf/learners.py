#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tensorflow learners.
"""
import numpy as np
import logging

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam

from .generators import CopulaBatchGenerator, set_generators_seed
from .initializers import set_initializers_seed
from .models import CopulaModel
from .losses import MINDLoss, ApproximateMINDLoss, RectifiedMINDLoss
from .config import get_default_parameter

def set_seed(seed):
	set_generators_seed(seed)
	set_initializers_seed(seed)


class CopulaLearner(object):
	'''
	Maximum-entropy learner.
	'''
	def __init__(self, d, beta_1=None, beta_2=None, epsilon=None, amsgrad=None, \
			name='Adam', learning_rate=None, subsets=[], epochs=None, steps_per_epoch=None, verbose=0):
		self.d = d
		self.model = CopulaModel(self.d, subsets=subsets)
		beta_1 = get_default_parameter('beta_1') if beta_1 is None else beta_1
		beta_2 = get_default_parameter('beta_2') if beta_2 is None else beta_2
		learning_rate = get_default_parameter('learning_rate') if learning_rate is None else learning_rate
		amsgrad = get_default_parameter('amsgrad') if amsgrad is None else amsgrad
		epsilon = get_default_parameter('epsilon') if epsilon is None else epsilon
		# logging.info('Using the Adam optimizer with learning parameters: ' \
		# 	'learning_rate: %.4f, beta_1: %.4f, beta_2: %.4f, epsilon: %.8f, amsgrad: %s' % \
		# 	(learning_rate, beta_1, beta_2, epsilon, amsgrad))
		self.opt = Adam(beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad, \
			name=name, learning_rate=learning_rate)
		self.loss = MINDLoss()
		self.model.compile(optimizer=self.opt, loss=self.loss)
		self.copula_entropy = None
		self.epochs = epochs
		self.steps_per_epoch = steps_per_epoch
		self.verbose = verbose


	def fit(self, z, batch_size=10000): #, steps_per_epoch=1000, epochs=None
		''' '''
		
		# epochs = get_default_parameter('epochs') if epochs is None else self.epochs #epochs
		epochs = self.epochs #epochs
		steps_per_epoch = self.steps_per_epoch
		
		z_gen = CopulaBatchGenerator(z, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
		
		self.model.fit(z_gen, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, \
			callbacks=[EarlyStopping(patience=3, monitor='loss', verbose=self.verbose), TerminateOnNaN()],
			verbose = self.verbose)
		self.copula_entropy = self.model.evaluate(z_gen)
