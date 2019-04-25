### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
#from atari_parameters import par
from parameters import par

class Network:

	def __init__(self, multiple_networks = False):

		self.multiple_networks = multiple_networks
		self.alpha_LTD = 1.
		self.weight_decay = 0.995
		self.names = ['pos']

		self.actions = []
		for k in range(4):
			a = np.zeros((par['batch_size'], 4), dtype = np.float32)
			a[:, k] = 1.
			self.actions.append(tf.constant(a))

		self.W_fixed = {}
		self.W_trace = {}
		self.W = {}

		for name in self.names:
			self.W_fixed[name] =  tf.constant(np.float32(np.random.choice([0., 1.], \
				size = [par['n_latent'], par['n_striatum']], p=[0.8, 0.2])))
			self.W_trace[name] = tf.Variable(np.zeros([par['batch_size'], par['n_latent'], \
				par['n_striatum']], dtype=np.float32), trainable = False)

			W_init = np.zeros([par['batch_size'],par['n_latent'], par['n_striatum']], dtype=np.float32) \
				if multiple_networks else np.zeros([par['n_latent'], par['n_striatum']], dtype=np.float32)
			self.W[name] = tf.Variable(W_init, trainable = False)


	def trace_layer(self, latent, action, name):

		x = tf.concat([latent, par['action_weight']*action], axis = 1)

		y_trace = tf.cast(tf.einsum('ij,jk->ik', x, self.W_fixed[name]) > par['trace_th'], tf.float32)

		x_contrib = tf.einsum('ij,jk->ijk', x, self.W_fixed[name]) - \
			self.alpha_LTD * tf.einsum('ij,jk->ijk', (1. - x), self.W_fixed[name])

		W_trace_grad = tf.einsum('ijk,ik->ijk', x_contrib, y_trace)

		print(y_trace, self.W_trace[name], W_trace_grad)

		return tf.assign(self.W_trace[name], par['trace_decay'] * tf.einsum('ik,ijk->ijk', (1. - y_trace), \
			self.W_trace[name]) + W_trace_grad)


	def dense_layer(self, latent, action, name):

		x = tf.concat([latent, par['action_weight']*action], axis = 1)
		if self.multiple_networks:
			return tf.nn.relu(tf.einsum('ij,ijk->ik', x, self.W[name]) - par['striatum_th'])
		else:
			return tf.nn.relu(x @ self.W[name] - par['striatum_th'])

	def read_striatum(self, latent):

		outputs = [tf.reduce_sum(self.dense_layer(latent, self.actions[k], 'pos'),axis=1,keepdims=True)\
		 	for k in range(4)]

		return tf.concat([*outputs], axis = 1)

	def write_striatum(self, latent, action, reward):

		update_trace_ops = [self.trace_layer(latent, action, name) for name in self.names]

		update_weight_ops = []
		print(self.names)
		for name in self.names:
			rew = tf.nn.relu(reward) if name == 'pos' else tf.nn.relu(-reward)
			W_grad = tf.einsum('ijk,i->jk',self.W_trace[name], tf.squeeze(rew))
			update_weight_ops.append(tf.assign(self.W[name], \
				tf.nn.relu(self.weight_decay * self.W[name] + W_grad)))

		normalize_ops = []
		norm_axis = 1 if self.multiple_networks else 0
		for name in self.names:
			d = tf.nn.relu(tf.reduce_sum(self.W[name], axis=1, keepdims=True) - 0.)
			#normalize_ops.append(tf.assign(self.W[name], tf.nn.relu(self.W[name] - d)))
			#normalize_ops.append(tf.assign(self.W[name], 0.*self.W[name]))
			normalize_ops.append(tf.assign(self.W[name], \
				self.W[name]/(1e-3+tf.reduce_sum(self.W[name], axis=1, keepdims=True))))

		return tf.group(*update_trace_ops), tf.group(*update_weight_ops), tf.group(*normalize_ops)


	def return_weights(self):

		return self.W['pos'], self.W['neg'], self.W_trace['pos'], self.W_trace['neg']


	def clear_traces_weights(self):

		clear_ops = []
		for name in self.names:
			clear_ops.append(tf.assign(self.W[name], 0.*self.W[name]))
			clear_ops.append(tf.assign(self.W_trace[name], 0.*self.W_trace[name]))

		return tf.group(*clear_ops)
