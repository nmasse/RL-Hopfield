### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time
from itertools import product

# Plotting suite
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model modules
from parameters_v10 import *
import stimulus_sequence
import AdamOpt_sequence as AdamOpt
import time

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Hopfield:

	def __init__(self, stim_size, action_size, reward_size):

		with tf.variable_scope('hopfield'):
			tf.get_variable(name="b1", shape=[128], initializer=tf.zeros_initializer())
			self.H_stim = tf.get_variable('H_stim', shape = [par['batch_size'], par['n_hopf_stim'], \
				par['n_hopf_stim']], initializer=tf.zeros_initializer())
			self.H_act_f = tf.get_variable('H_act_f', shape = [par['batch_size'], par['n_hopf_stim'], \
				par['n_hopf_act']], initializer=tf.zeros_initializer())
			self.H_act_r = tf.get_variable('H_act_r', shape = [par['batch_size'], par['n_hopf_act'], \
				par['n_hopf_stim']], initializer=tf.zeros_initializer())

		self.H_stim_mask = tf.constant(par['H_stim_mask'])


	def read_fast_weights(self, h):

		h = tf.nn.relu(h)
		h_hat = tf.zeros_like(h)
		alpha = 0.5
		cycles = 6
		for n in range(cycles):

			h_hat = alpha*h_hat + (1 - alpha) * tf.einsum('ij, ijk->ik', h, self.H_stim) + h
			h_hat = tf.nn.relu(h_hat)
			h_hat = tf.minimum(100., h_hat)

		pred_action = tf.einsum('ij, ijk->ik', h_hat, self.H_act_f)

		return pred_action

	def write_fast_weights(self, h, a):

		x = tf.placeholder(tf.float32, [par['batch_size'], par['n_latent']], 'x_hopfield')
		a = tf.placeholder(tf.float32, [par['batch_size'], par['n_pol']], 'a_hopfield')

		h = tf.nn.relu(h)
		hh = tf.einsum('ij,ik->ijk',h,h)
		h_old_new = tf.einsum('ij,ik->ijk', h_old, h)
		#h_old_new *= par['H_old_new_mask']

		if par['covariance_method']:
			H_stim += (h_old_new + hh)/par['n_hopf_stim']
		else:
			h1 = tf.einsum('ij, jk->ijk', h, self.H_stim_mask)
			c = tf.einsum('ijk,ikm->ijm', h1, self.H_stim)
			H_stim_grad = (h_old_new + hh - c - tf.transpose(c,[0,2,1]))/par['n_hopf_stim']
			H_stim_grad = tf.einsum('ijk,jk->ijk', H_stim_grad, self.H_stim_mask)

		H_act_grad = tf.einsum('ij, ik->ijk', h, a)
		# DO I INCLUDE THIS NEXT PART?
		#H_act += tf.einsum('ij, ik->ijk', a_old, h)

		update_H_stim = tf.assign_add(self.H_stim, H_stim_grad)
		update_H_act = tf.assign_add(self.H_act, H_act_grad)

		self.update_hopfield = tf.group(*[update_H_stim, update_H_act])


class RL:

	def __init__(self):

		self.declare_variables()
		self.forward_pass()

	def declare_variables(self):

		self.var_dict = {}
		RL_prefixes 		= ['W0', 'W1', 'b0', 'b1' ,'W_pol', 'W_val', 'b_pol', 'b_val']

		with tf.variable_scope('RL'):
			for p in RL_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer = par[p + '_init'])

		self.W_stim_write = tf.constant(par['W_stim_write'])
		self.W_act_write = tf.constant(par['W_act_write'])


	def forward_pass(self):

		"""
		Compute the policy/value functions and choose an action from the encoded
		stimulus and read-out from the Hopfield network
		"""
		x0 = tf.placeholder(tf.float32, [par['batch_size'], par['n_latent']], 'x0')
		x1 = tf.nn.relu(x0 @ self.var_dict['W0'] + self.var_dict['b0'])
		#x1 = tf.layers.dropout(x1, rate = par['drop_rate'], training = True)
		x2 = tf.nn.relu(x1 @ self.var_dict['W1'] + self.var_dict['b2'])

		self.pol_out = x2 @ self.var_dict['W_pol'] + self.var_dict['b_pol']
		self.val_out = x2 @ self.var_dict['W_val'] + self.var_dict['b_val']

		"""
		Perform gradient descent
		"""
		action   = tf.placeholder(tf.float32, [par['batch_size'], par['n_pol']], 'action')
		reward 	 = tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'reward')
		prev_val = tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'prev_val')

		RL_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='RL')
		adam_optimizer = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])

		terminal_state = tf.cast(tf.logical_not(tf.equal(reward, tf.constant(0.))), tf.float32)
		advantage = prev_val - reward - par['discount_rate']*self.val_out*terminal_state
		self.val_loss = 0.5*tf.reduce_mean(mask_static*tf.square(advantage))

		pol_out_softmax   = tf.nn.softmax(self.pol_out, axis = -1)
		self.pol_loss     = -tf.reduce_mean(tf.stop_gradient(advantage)*action \
			*tf.log(1e-6 + pol_out_softmax)
		self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(pol_out_softmax \
			*tf.log(1e-6 + pol_out_softmax), axis = -1))

		self.loss = self.pol_loss + par['val_cost']*self.val_loss \
			- par['entropy_cost']*self.ent_loss
		self.train_RL = adam_optimizer.minimize(self.loss, var_list = RL_vars))



class Encoder:

	def __init__(self):

		self.declare_variables()
		self.forward_pass()

	def declare_variables(self):

		self.var_dict = {}
		encoding_prefixes   = ['W_enc', 'W_dec', 'b_enc']

		with tf.variable_scope('encoding'):
			for p in encoding_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer = par[p + '_init'])

		self.W_stim_read = tf.constant(par['W_stim_read'])
		self.hopfield = Hopfield()


	def forward_pass(self):


		"""
		Encode the stimulus into a sparse representation
		"""
		stim = tf.placeholder(tf.float32, [par['batch_size'], par['n_input']], 'stim')
		x = tf.nn.relu(stim @ self.var_dict['W_enc'] + self.var_dict['b_enc'])
		stim_hat = x @ self.var_dec['W_enc']

		self.reconstruction_loss = tf.reduce_mean(tf.square(stim - stim_hat))
		self.weight_loss = tf.reduce_mean(tf.abs(self.var_dict['W0'])) + tf.reduce_mean(tf.abs(self.var_dict['W1']))
		x_mask = np.ones((par['n_latent'], par['n_latent']),dtype = np.float32) - np.eye((par['n_latent']),dtype = np.float32)
		self.sparsity_loss = tf.reduce_mean(x_mask*(tf.transpose(y) @ y))

		self.loss = self.reconstruction_loss + par['sparsity_cost']*self.sparsity_loss \
			+ par['weight_cost']*self.weight_loss

		encoding_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'encoding')
		adam_optimizer = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
		self.train_encoder = adam_optimizer.minimize(self.loss, var_lis = encoding_vars)


		"""
		2. Pass the encoded stimulus into the Hopfield network, retrieve predicted action/values
		"""
		x_mapped = x @ self.W_stim_read
		x_read = self.hopefield.read_fast_weights(x_mapped)
		x_read = tf.reshape(x_read, [par['batch_size'], par['num_reward_types']*par['n_pol'], \
			par['num_time_steps']//par['temporal_div']])
		x_read = tf.reduce_sum(x_read, axis = 2)

		# main output
		self.x_concat = tf.concat([x_read, x], axis = 1)


def main(gpu_id=None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	stim = stimulus_sequence.Stimulus()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) \
		if gpu_id == '0' else tf.GPUOptions()

	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			rl = RL()
			encoder = Encoder()
			hopfield = Hopfield()

		sess.run(tf.global_variables_initializer())

		for i in range(par['n_batches']):

			rooms # initialize rooms

			for t in range(par['n_time_steps']):

				x = sess.run([self.predictions], feed_dict = {self.stim_pl: s})





def print_important_params():

	notes = ''

	keys = ['learning_method', 'n_hidden', 'n_latent', 'noise_in','noise_rnn','top_down',\
		'A_alpha_init', 'A_beta_init', 'inner_steps', 'batch_norm_inner', 'learning_rate', \
		'task_list', 'trials_per_seq', 'fix_break_penalty', 'wrong_choice_penalty', \
		'correct_choice_reward', 'discount_rate', 'num_motion_dirs', 'sparsity_cost', 'n_filters', \
		'rec_cost', 'weight_cost', 'entropy_cost', 'val_cost', 'drop_rate', 'batch_size', \
		'n_batches', 'share_hippocampus', 'save_fn','temporal_div']

	print('-'*60)
	[print('{:<24} : {}'.format(k, par[k])) for k in keys]
	print('{:<24} : {}'.format('notes', notes))
	print('-'*60 + '\n')



if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
