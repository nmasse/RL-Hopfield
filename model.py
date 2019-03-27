### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import AdamOpt
import pickle
import os, sys, time
from itertools import product

# Plotting suite
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model modules
from parameters import *
import stimulus
import time

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

	def __init__(self, stim, reward, action, future_val, time_step):

		# Placeholders
		self.stim_pl		= stim
		self.reward_pl		= reward
		self.action_pl		= action
		self.future_val_pl	= future_val
		self.time_step_pl	= time_step

		self.declare_variables()
		self.stimulus_encoding()
		self.policy()
		self.write_hopfield()
		self.calculate_encoder_grads()
		self.calculate_policy_grads()
		self.update_weights()


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
			self.H_neuron = tf.get_variable('H_act_n', shape = [par['batch_size'], par['n_hopf_stim']], \
				initializer=tf.zeros_initializer(), trainable = False)

		self.W_stim_write = tf.constant(par['W_stim_write'])
		self.W_act_write = tf.constant(par['W_act_write'])
		self.W_stim_read = tf.constant(par['W_stim_read'])
		self.H_stim_mask = tf.constant(par['H_stim_mask'])


	def stimulus_encoding(self):

		"""
		Encode the stimulus into a sparse representation
		"""
		self.latent = tf.nn.relu(self.stim_pl @ self.var_dict['W_enc'] + self.var_dict['b_enc'])
		self.stim_hat = self.latent @ self.var_dict['W_dec']

		"""
		Project the encoded stimulus into the Hopfield network, retrieve predicted action/values
		"""
		x_mapped = self.latent @ self.W_stim_read
		x_read = self.read_hopfield(x_mapped)
		x_read = tf.reshape(x_read, [par['batch_size'], (len(par['rewards']) + 1)*par['n_pol'], \
			par['hopf_multiplier']])
		self.x_read = tf.reduce_sum(x_read, axis = 2)
		#print(x_read)
		#print(self.latent)
		self.encoding_out = tf.concat([self.x_read, self.latent], axis = 1) # main output


	def policy(self):

		"""
		Calculate the policy and value functions based on the latent
		representation and the read-out from the Hopfield network
		"""
		x1 = tf.nn.relu(self.encoding_out @ self.var_dict['W0'] + self.var_dict['b0'])
		#x1 = tf.layers.dropout(x1, rate = par['drop_rate'], training = True)
		x2 = tf.nn.relu(x1 @ self.var_dict['W1'] + self.var_dict['b1'])

		self.pol_out = tf.nn.softmax(x2 @ self.var_dict['W_pol'] + self.var_dict['b_pol'])
		self.val_out = x2 @ self.var_dict['W_val'] + self.var_dict['b_val']


	def calculate_encoder_grads(self):

		"""
		Calculate the gradient on the latent weights
		"""
		encoding_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'encoding')
		self.encoding_optimizer = AdamOpt.AdamOpt(encoding_vars, 0.001)

		self.reconstruction_loss = tf.reduce_mean(tf.square(self.stim_pl - self.stim_hat))
		self.weight_loss = tf.reduce_mean(tf.abs(self.var_dict['W_enc'])) + tf.reduce_mean(tf.abs(self.var_dict['W_dec']))
		latent_mask = np.ones((par['n_latent'], par['n_latent']),dtype = np.float32) - np.eye((par['n_latent']),dtype = np.float32)
		self.sparsity_loss = tf.reduce_mean(latent_mask*(tf.transpose(self.latent) @ self.latent))/par['batch_size']
		self.loss = self.reconstruction_loss + par['sparsity_cost']*self.sparsity_loss \
			+ par['weight_cost']*self.weight_loss
		self.train_encoder = self.encoding_optimizer.compute_gradients(self.loss)


	def calculate_policy_grads(self):

		"""
		Calculate the gradient on the policy/value weights
		"""
		RL_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='RL')
		self.RL_optimizer = AdamOpt.AdamOpt(RL_vars, par['learning_rate'])

		not_terminal_state = tf.cast(tf.equal(self.reward_pl, tf.constant(0.)), tf.float32)
		advantage = self.reward_pl + par['discount_rate']*self.future_val_pl*not_terminal_state - self.val_out
		self.val_loss = 0.5*tf.reduce_mean(tf.square(advantage))
		self.pol_loss     = -tf.reduce_mean(tf.stop_gradient(advantage*self.action_pl) \
			*tf.log(1e-9 + self.pol_out))
		self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.pol_out \
			*tf.log(1e-9 + self.pol_out), axis = -1))

		self.loss = self.pol_loss + par['val_cost']*self.val_loss \
			- par['entropy_cost']*self.entropy_loss
		self.train_RL = self.RL_optimizer.compute_gradients(self.loss)

	def update_weights(self):

		"""
		Apply the weight changes
		"""
		self.update_weights = tf.group(*[self.encoding_optimizer.update_weights(), \
			self.RL_optimizer.update_weights()])


	def read_hopfield(self, x):

		h_hat = tf.zeros_like(x)
		alpha = 0.5
		cycles = 6
		for n in range(cycles):

			h_hat = alpha*h_hat + (1-alpha)*tf.einsum('ij, ijk->ik', x, self.H_stim) + x
			h_hat = tf.nn.relu(h_hat)
			h_hat = tf.minimum(100., h_hat)

		pred_action = tf.einsum('ij, ijk->ik', h_hat, self.H_act_f)

		return pred_action/tf.reduce_sum(pred_action, axis = 1, keepdims = True)


	def write_hopfield(self):

		h = self.latent @ self.W_stim_write[self.time_step_pl % par['hopf_multiplier'], :, :]
		action_reward = tf.concat([self.action_pl*self.reward_pl, \
			self.action_pl*(1 - self.reward_pl)], axis = -1)
		action_reward = action_reward @ self.W_act_write[self.time_step_pl % par['hopf_multiplier'], :, :]
		hh = tf.einsum('ij,ik->ijk', h, h)
		h_old_new = tf.einsum('ij,ik->ijk', 0.*self.H_neuron, h)
		#h_old_new *= par['H_old_new_mask']

		if par['covariance_method']:
			H_stim += (h_old_new + hh)/par['n_hopf_stim']
		else:
			h1 = tf.einsum('ij, jk->ijk', h, self.H_stim_mask)
			c = tf.einsum('ijk,ikm->ijm', h1, self.H_stim)
			H_stim_grad = (h_old_new + hh - c - tf.transpose(c,[0,2,1]))/par['n_hopf_stim']
			H_stim_grad = tf.einsum('ijk,jk->ijk', H_stim_grad, self.H_stim_mask)

		H_act_grad = tf.einsum('ij, ik->ijk', h, action_reward)
		# DO I INCLUDE THIS NEXT PART?
		#H_act += tf.einsum('ij, ik->ijk', a_old, h)

		update_H_stim = tf.assign_add(self.H_stim, H_stim_grad)
		update_H_act = tf.assign_add(self.H_act_f, H_act_grad)
		update_H_neuron = tf.assign(self.H_neuron, par['hopf_neuron_alpha']*self.H_neuron + \
			(1-par['hopf_neuron_alpha'])*h)

		self.update_hopfield = tf.group(*[update_H_stim, update_H_act])

		update_H_ops = []
		update_H_ops.append(tf.assign(self.H_neuron, 0.*self.H_neuron))
		update_H_ops.append(tf.assign(self.H_act_f, 0.*self.H_act_f))
		update_H_ops.append(tf.assign(self.H_stim, 0.*self.H_stim))
		self.reset_hopfield = tf.group(*update_H_ops)


def main(gpu_id = None):

	# Print out context
	print_important_params()

	# Select GPU
	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	# Reduce memory consumption for GPU 0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) \
		if gpu_id == '0' else tf.GPUOptions()

	# Initialize stimulus environment
	environment = stimulus.Stimulus()

	# Reset graph and designate placeholders
	tf.reset_default_graph()
	stim_pl      = tf.placeholder(tf.float32, [par['batch_size'], par['n_input']], 'stim')
	action_pl    = tf.placeholder(tf.float32, [par['batch_size'], par['n_pol']], 'action')
	reward_pl 	 = tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'reward')
	future_val_pl = tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'prev_val')
	time_step_pl = tf.placeholder(tf.int32, [], 'time_step')

	# Start TensorFlow session
	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		# Set up and initialize model on desired device
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(stim_pl, reward_pl, action_pl, future_val_pl, time_step_pl)
		sess.run(tf.global_variables_initializer())

		# Start training loop
		for i in range(par['num_batches']):

			# Reset environment at the start of each iteration
			environment.reset_agents()
			environment.reset_rewards()

			# Pre-allocate prev_val and total_reward
			prev_stim = np.zeros((par['batch_size'], par['n_input']), dtype = np.float32)
			future_val = np.zeros((par['batch_size'], par['n_val']), dtype = np.float32)
			prev_action = np.zeros((par['batch_size'], par['n_pol']), dtype = np.float32)
			prev_reward = np.zeros((par['batch_size'], par['n_val']), dtype = np.float32)
			prev_t = 0
			total_reward = np.zeros((par['batch_size'],1), dtype = np.float32)

			action_record = []

			# Iterate through time
			for t in range(par['num_time_steps']):

				# Make inputs
				stim_in = environment.make_inputs()

				# Train encoder weights, output the policy and value functions
				_, pol, val, rec_loss, sparsity_loss, x_read = sess.run([model.train_encoder, model.pol_out, model.val_out, \
					model.reconstruction_loss, model.sparsity_loss, model.x_read], feed_dict = {stim_pl: stim_in})

				W = sess.run(model.var_dict)

				# Choose action, calculate reward and determine next state
				action = np.array([np.random.multinomial(1, pol[t,:]-1e-6) for t in range(par['batch_size'])])
				reward = environment.agent_action(action)
				action_record.append(action)

				# Update total reward and prev_val
				total_reward += reward


				if i > 100:
					# Update the Hopfield network
					sess.run([model.update_hopfield, model.train_RL], \
						feed_dict = {stim_pl: prev_stim, action_pl: prev_action, reward_pl: prev_reward, \
						future_val_pl: val, time_step_pl:prev_t})

					prev_stim = stim_in
					prev_reward = reward
					prev_action = action
					prev_t = t

				# Reset agents that have obtained a reward
				environment.reset_agents(reward != 0.)

			# Update model weights
			sess.run(model.update_weights)
			sess.run(model.reset_hopfield)

			# Analyze actions
			action_record = np.concatenate(action_record, axis=0)
			action_record = np.round(np.mean(action_record, axis=0), 2).tolist()

			# Output network performance
			print('Iter {:>4} | Mean Reward: {:6.3f} | Recon Loss: {:8.6f} | Sparsity Loss: {:8.6f} | Action Dist: {}'.format(\
				i, np.mean(total_reward), rec_loss, sparsity_loss, action_record))
			# print('x_read ', np.mean(x_read))

def print_important_params():

	keys = ['save_fn', 'learning_rate', 'n_hidden', 'n_latent', 'hopf_multiplier', \
		'hopf_alpha', 'hopf_neuron_alpha', 'hopf_beta', 'hopf_cycles', 'covariance_method', \
		'drop_rate', 'discount_rate', 'num_time_steps', 'num_batches', 'batch_size', 'rewards', \
		'room_width', 'room_height', 'sparsity_cost', 'rec_cost', 'weight_cost', 'entropy_cost', \
		'val_cost']

	print('-'*60)
	[print('{:<24} : {}'.format(k, par[k])) for k in keys]
	print('-'*60 + '\n')

if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
