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
import time

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

	def __init__(self, stim, reward, action, prev_val):

		# Placeholders
		self.stim_pl		= stim
		self.reward_pl		= reward
		self.action_pl		= action
		self.prev_val_pl	= prev_val

		self.declare_variables()
		self.stimulus_encoding()
		self.policy()
		self.update_encoder_weights()
		self.update_policy_weights()


	def declare_variables(self):

		self.var_dict = {}
		encoding_prefixes   = ['W_enc', 'W_dec', 'b_enc']
		RL_prefixes 		= ['W0', 'W1', 'b0', 'b1' ,'W_pol', 'W_val', 'b_pol', 'b_val']

		with tf.variable_scope('encoding'):
			for p in encoding_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer = par[p + '_init'])
		with tf.variable_scope('RL'):
			for p in RL_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer = par[p + '_init'])
		with tf.variable_scope('hopfield'):
			self.H_stim = tf.get_variable('H_stim', shape = [par['batch_size'], par['n_hopf_stim'], \
				par['n_hopf_stim']], initializer=tf.zeros_initializer(), trainable = False)
			self.H_act_f = tf.get_variable('H_act_f', shape = [par['batch_size'], par['n_hopf_stim'], \
				par['n_hopf_act']], initializer=tf.zeros_initializer(), trainable = False)
			self.H_act_r = tf.get_variable('H_act_r', shape = [par['batch_size'], par['n_hopf_act'], \
				par['n_hopf_stim']], initializer=tf.zeros_initializer(), trainable = False)

		self.W_stim_write = tf.constant(par['W_stim_write'])
		self.W_act_write = tf.constant(par['W_act_write'])
		self.W_stim_read = tf.constant(par['W_stim_read'])
		self.H_stim_mask = tf.constant(par['H_stim_mask'])

	def stimulus_encoding():

		"""
		Encode the stimulus into a sparse representation
		"""
		self.latent = tf.nn.relu(self.stim_pl @ self.var_dict['W_enc'] + self.var_dict['b_enc'])
		self.stim_hat = x @ self.var_dec['W_enc']

		"""
		Project the encoded stimulus into the Hopfield network, retrieve predicted action/values
		"""
		x_mapped = x @ self.W_stim_read
		x_read = self.read_hopfield(x_mapped)
		x_read = tf.reshape(x_read, [par['batch_size'], par['num_reward_types']*par['n_pol'], \
			par['num_time_steps']//par['temporal_div']])
		x_read = tf.reduce_sum(x_read, axis = 2)
		self.encoding_out = tf.concat([x_read, x], axis = 1) # main output


	def policy(self):


		x1 = tf.nn.relu(self.encoding_out @ self.var_dict['W0'] + self.var_dict['b0'])
		#x1 = tf.layers.dropout(x1, rate = par['drop_rate'], training = True)
		x2 = tf.nn.relu(x1 @ self.var_dict['W1'] + self.var_dict['b2'])

		self.pol_out = x2 @ self.var_dict['W_pol'] + self.var_dict['b_pol']
		self.val_out = x2 @ self.var_dict['W_val'] + self.var_dict['b_val']


	def update_encoder_weights(self):

		"""
		Update encoding weights
		"""
		self.adam_optimizer = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
		encoding_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'encoding')

		self.reconstruction_loss = tf.reduce_mean(tf.square(self.stim_pl - self.stim_hat))
		self.weight_loss = tf.reduce_mean(tf.abs(self.var_dict['W_enc'])) + tf.reduce_mean(tf.abs(self.var_dict['W_dec']))
		latent_mask = np.ones((par['n_latent'], par['n_latent']),dtype = np.float32) - np.eye((par['n_latent']),dtype = np.float32)
		self.sparsity_loss = tf.reduce_mean(latent_mask*(tf.transpose(self.latent) @ self.latent))
		self.loss = self.reconstruction_loss + par['sparsity_cost']*self.sparsity_loss \
			+ par['weight_cost']*self.weight_loss
		self.train_encoder = self.adam_optimizer.minimize(self.loss, var_lis = encoding_vars)


	def update_policy_weights(self):

		"""
		Perform gradient descent
		"""
		RL_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='RL')

		terminal_state = tf.cast(tf.logical_not(tf.equal(self.reward_pl, tf.constant(0.))), tf.float32)
		advantage = self.prev_val_pl - self.reward_pl - par['discount_rate']*self.val_out*terminal_state
		self.val_loss = 0.5*tf.reduce_mean(tf.square(advantage))

		pol_out_softmax   = tf.nn.softmax(self.pol_out, axis = -1)
		self.pol_loss     = -tf.reduce_mean(tf.stop_gradient(advantage)*action \
			*tf.log(1e-6 + pol_out_softmax)
		self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(pol_out_softmax \
			*tf.log(1e-6 + pol_out_softmax), axis = -1))

		self.loss = self.pol_loss + par['val_cost']*self.val_loss \
			- par['entropy_cost']*self.ent_loss
		self.train_RL = adam_optimizer.minimize(self.loss, var_list = RL_vars)


	def read_hopfield(self):

		h_hat = tf.zeros_like(self.latent)
		alpha = 0.5
		cycles = 6
		for n in range(cycles):

			h_hat = alpha*h_hat + (1-alpha)*tf.einsum('ij, ijk->ik', self.latent, self.H_stim) + self.latent
			h_hat = tf.nn.relu(h_hat)
			h_hat = tf.minimum(100., h_hat)

		pred_action = tf.einsum('ij, ijk->ik', h_hat, self.H_act_f)

		return pred_action


	def write_hopfield(self):

		action_reward = tf.concat([self.action_pl*self.reward_pl, \
			self.action_pl*(1 - self.reward_pl)], axis = -1)
		hh = tf.einsum('ij,ik->ijk', self.latent, self.latent)
		h_old_new = tf.einsum('ij,ik->ijk', h_old, self.latent)
		#h_old_new *= par['H_old_new_mask']

		if par['covariance_method']:
			H_stim += (h_old_new + hh)/par['n_hopf_stim']
		else:
			h1 = tf.einsum('ij, jk->ijk', self.latent, self.H_stim_mask)
			c = tf.einsum('ijk,ikm->ijm', h1, self.H_stim)
			H_stim_grad = (h_old_new + hh - c - tf.transpose(c,[0,2,1]))/par['n_hopf_stim']
			H_stim_grad = tf.einsum('ijk,jk->ijk', H_stim_grad, self.H_stim_mask)

		H_act_grad = tf.einsum('ij, ik->ijk', self.latent, action_reward)
		# DO I INCLUDE THIS NEXT PART?
		#H_act += tf.einsum('ij, ik->ijk', a_old, h)

		update_H_stim = tf.assign_add(self.H_stim, H_stim_grad)
		update_H_act = tf.assign_add(self.H_act, H_act_grad)

		self.update_hopfield = tf.group(*[update_H_stim, update_H_act])


def main(gpu_id = None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	stim = stimulus_sequence.Stimulus()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) \
		if gpu_id == '0' else tf.GPUOptions()

	tf.reset_default_graph()
	stim     = tf.placeholder(tf.float32, [par['batch_size'], par['n_input']], 'stim')
	action   = tf.placeholder(tf.float32, [par['batch_size'], par['n_pol']], 'action')
	reward 	 = tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'reward')
	prev_val = tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'prev_val')

	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model()


		sess.run(tf.global_variables_initializer())

		for i in range(par['n_batches']):

			stimulus = rooms # initialize rooms
			prev_val = np.zeros((par['batch_size'], par['n_val']), dtype = np.float32)
			total_reward = np.zeros((par['batch_size'], 1), dtype = np.float32)

			for t in range(par['n_time_steps']):

				# Train encoder weights, output the policy and value functions
				_, pol, val = sess.run([model.train_encoder, model.pol_out, model.val_out], \
					feed_dict = {stim: stimulus})

				# Choose action, calculate reward and determine next state
				reward 			= 0. # CALCULATE REWARD
				action_index	= np.random.multinomial(pol, 1)
				action 			= np.one_hot(tf.squeeze(action_index), par['n_pol'])

				sess.run([model.update_hopfield], \
					feed_dict = {stim: stimulus, action: action, reward: reward})

				stimulus, reward = rooms
				total_reward += reward
				prev_val = val

				# append to buffer

			sess.run(rl.train_RL, feed_dict = {rl.x0: latent, rl.action: action, \
				rl.reward: reward, rl.prev_val: prev_val})





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
