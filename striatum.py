### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
#from atari_parameters import par
from parameters import par

class Network:

	def __init__(self):

		self.alpha_LTD = 0.1
		self.weight_decay = 0.995

		self.W_fixed = {}
		self.W_trace = {}
		self.W = {}

		for name in ['pos', 'neg']:
			self.W_fixed[name] =  tf.constant(np.float32(np.random.choice([0., 1.], \
				size = [par['n_latent'], par['n_striatum']], p=[0.9, 0.1])))

			self.W_trace[name] = tf.Variable(np.zeros([par['batch_size'], par['n_latent'], \
				par['n_striatum']], dtype=np.float32), trainable = False)

			self.W[name] = tf.Variable(np.zeros([par['n_latent'], par['n_striatum']], \
				dtype=np.float32), trainable = False)

	def trace_layer(self, x, name):

		y_trace = tf.nn.relu(tf.einsum('ij,jk->ik', x, self.W_fixed[name]) - par['trace_th'])
		y_trace = tf.minimum(1., y_trace)

		x_contrib = tf.einsum('ij,jk->ijk', x, self.W_fixed[name]) - \
			self.alpha_LTD * tf.einsum('ij,jk->ijk', (1. - x), self.W_fixed[name])

		W_trace_grad = tf.einsum('ijk,ik->ijk', x_contrib, y_trace)

		return tf.assign(self.W_trace[name], \
			par['trace_decay'] * self.W_trace[name] + W_trace_grad)


	def dense_layer(self, x, name):

		print(x.shape, self.W['pos'].shape)
		return tf.nn.relu(x @ self.W[name] - par['striatum_th'])


	def run(self, latent, reward):

		names = ['pos', 'neg']
		update_trace_ops = [self.trace_layer(latent, name) for name in names]

		outputs = [self.dense_layer(latent, name) for name in names]

		update_weight_ops = []
		for name in names:
			rew = tf.nn.relu(reward) if name == 'pos' else tf.nn.relu(-reward)
			W_grad = tf.einsum('ijk,i->jk',self.W_trace[name], tf.squeeze(rew))
			update_weight_ops.append(tf.assign(self.W[name], self.weight_decay * self.W[name] + W_grad))

		normalize_ops = []
		for name in names:
			normalize_ops.append(tf.assign(self.W[name], tf.nn.relu(self.W[name]) \
				/(1e-6 + tf.reduce_sum(tf.nn.relu(self.W[name]), axis=0, keepdims=True))))
			#normalize_ops.append(tf.assign(self.W_trace[name], \
			#	self.W_trace[name]/tf.reduce_sum(self.W_trace[name], axis=1, keepdims=True)))

		return tf.concat([*outputs], axis = 1), tf.group(*update_trace_ops), \
			tf.group(*update_weight_ops), tf.group(*normalize_ops)

	def return_weights(self):

		return self.W['pos'], self.W['neg'], self.W_trace['pos'], self.W_trace['neg']
