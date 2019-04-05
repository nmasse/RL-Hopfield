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

	def __init__(self, stim, prev_latent, latent, hp_latent, prev_reward, reward, \
		prev_action, action, prev_val, future_val):

		# Placeholders
		self.stim_pl		= stim
		self.prev_latent_pl	= prev_latent
		self.latent_pl		= latent
		self.hop_latent_pl  = hp_latent
		self.reward_pl		= reward
		self.prev_reward_pl	= prev_reward
		self.action_pl		= action
		self.prev_action_pl	= prev_action
		self.prev_val_pl	= prev_val
		self.future_val_pl	= future_val

		self.declare_variables()
		self.stimulus_encoding()
		self.policy()
		self.calculate_encoder_grads()
		self.calculate_policy_grads()
		self.write_hopfield()
		self.update_weights()


	def declare_variables(self):

		self.var_dict = {}
		encoding_prefixes = ['W_enc0', 'W_dec0', 'W_dec1', 'b_enc0', 'b_dec0', 'b_dec1']
		VAE_prefixes      = ['W_mu', 'W_sigma', 'b_mu', 'b_sigma']
		RL_prefixes       = ['W0', 'W1', 'b0', 'b1' ,'W_pol', 'W_val', 'b_pol', 'b_val']

		with tf.variable_scope('encoding'):
			for p in encoding_prefixes + VAE_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer = par[p + '_init'])

		with tf.variable_scope('RL'):
			for p in RL_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer = par[p + '_init'])

		with tf.variable_scope('hopfield'):
			self.H_sas = tf.get_variable('H_sas', shape = [par['batch_size'], par['n_latent'], \
				par['n_pol'],par['n_latent']], initializer=tf.zeros_initializer(), trainable = False)
			self.H_sar = tf.get_variable('H_sar', shape = [par['batch_size'], par['n_latent'], \
				par['n_pol'], par['n_unique_vals']], initializer=tf.zeros_initializer(), trainable = False)
			"""
			self.H_act_r = tf.get_variable('H_act_r', shape = [par['batch_size'], par['n_hopf_act'], \
				par['n_hopf_stim']], initializer=tf.zeros_initializer(), trainable = False)
			self.H_neuron = tf.get_variable('H_neuron', shape = [par['batch_size'], par['n_hopf_stim']], \
				initializer=tf.zeros_initializer(), trainable = False)
			"""

		#self.W_stim_write = tf.constant(par['W_stim_write'])
		#self.W_act_write = tf.constant(par['W_act_write'])
		#self.W_stim_read = tf.constant(par['W_stim_read'])
		#self.H_stim_mask = tf.constant(par['H_stim_mask'])


	def stimulus_encoding(self):

		"""
		Encode the stimulus into a sparse representation
		"""
		#self.stim_pl += tf.random_normal(tf.shape(self.stim_pl), 0, 0.1)
		x = tf.nn.relu(self.stim_pl @ self.var_dict['W_enc0'] + self.var_dict['b_enc0'])
		#x = tf.nn.relu(x @ self.var_dict['W_enc1'] + self.var_dict['b_enc1'])

		self.latent_mu = x @ self.var_dict['W_mu'] + self.var_dict['b_mu']
		self.latent_log_var = x @ self.var_dict['W_sigma'] + self.var_dict['b_sigma']
		self.latent_loss = -0.5*tf.reduce_sum(1 + self.latent_log_var \
			- tf.square(self.latent_mu) - tf.exp(self.latent_log_var))

		#self.latent = self.latent_mu + tf.exp(self.latent_log_var/2)* \
		#	tf.random_normal([par['batch_size'], par['n_latent']], 0, 1 , dtype=tf.float32)
		self.latent = self.latent_mu

		x_dec  = tf.nn.relu(self.latent @ self.var_dict['W_dec0'] + self.var_dict['b_dec0'])
		#x_dec  = tf.nn.relu(x_dec @ self.var_dict['W_dec1'] + self.var_dict['b_dec1'])
		self.stim_hat = x_dec @ self.var_dict['W_dec1'] + self.var_dict['b_dec1']




	def policy(self):

		"""
		Calculate the policy and value functions based on the latent
		representation and the read-out from the Hopfield network
		"""
		self.hopfield_read, h_hat = self.read_hopfield(self.latent_pl)
		self.encoding_out = tf.concat([self.hopfield_read, self.latent_pl, h_hat], axis = 1) # main output
		x1 = tf.nn.relu(self.encoding_out @ self.var_dict['W0'] + self.var_dict['b0'])
		x2 = tf.nn.relu(x1 @ self.var_dict['W1'] + self.var_dict['b1'])

		self.pol_out = tf.nn.softmax(x2 @ self.var_dict['W_pol'] + self.var_dict['b_pol'])
		self.val_out = x2 @ self.var_dict['W_val'] + self.var_dict['b_val']


	def calculate_encoder_grads(self):

		"""
		Calculate the gradient on the latent weights
		"""
		encoding_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'encoding')
		self.encoding_optimizer = AdamOpt.AdamOpt(encoding_vars, 0.0002)

		stim = self.stim_pl/(1e-9 + tf.sqrt(tf.reduce_sum(self.stim_pl**2,axis=1,keepdims = True)))
		# if the dot-product between stimuli is less than 0.95, consider them different
		s = tf.cast((stim @ tf.transpose(stim)) < 0.99, tf.float32)

		latent = self.latent_mu/(1e-9 + tf.sqrt(tf.reduce_sum(self.latent_mu**2,axis=1,keepdims = True)))
		c = latent @ tf.transpose(latent)
		c *= s
		self.sparsity_loss = tf.reduce_mean(tf.abs(c))

		self.reconstruction_loss = tf.reduce_mean(tf.square(self.stim_pl - self.stim_hat))
		self.loss = self.reconstruction_loss + par['latent_cost']*self.latent_loss \
			+ par['sparsity_cost']*self.sparsity_loss

		if par['train_encoder']:
			self.train_encoder = self.encoding_optimizer.compute_gradients(self.loss)
		else:
			self.train_encoder = tf.no_op()


	def calculate_policy_grads(self):

		"""
		Calculate the gradient on the policy/value weights
		"""
		RL_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='RL')
		self.RL_optimizer = AdamOpt.AdamOpt(RL_vars, par['learning_rate'])

		not_terminal_state = tf.cast(tf.equal(self.reward_pl, tf.constant(0.)), tf.float32)
		advantage = self.reward_pl + par['discount_rate']*self.future_val_pl*not_terminal_state - self.val_out
		self.val_loss = 0.5*tf.reduce_mean(tf.square(advantage))
		self.pol_loss = -tf.reduce_mean(tf.stop_gradient(advantage)*self.action_pl*tf.log(1e-9 + self.pol_out))
		self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.pol_out*tf.log(1e-9 + self.pol_out), axis = -1))

		self.loss = self.pol_loss + par['val_cost']*self.val_loss \
			- par['entropy_cost']*self.entropy_loss
		self.train_RL = self.RL_optimizer.compute_gradients(self.loss)


	def update_weights(self):

		"""
		Apply the weight changes
		"""
		if par['train_encoder']:
			self.update_weights = tf.group(*[self.encoding_optimizer.update_weights(), \
				self.RL_optimizer.update_weights()])
		else:
			self.update_weights = tf.group(self.RL_optimizer.update_weights())


	def read_hopfield(self, x):

		n_steps = 6
		gamma = 0.9
		x = self.latent_pl/(1e-9 + tf.sqrt(tf.reduce_sum(self.latent_pl**2, axis = 1, keepdims=True)))
		action_reward = tf.einsum('ijkm,ij->ikm', self.H_sar, x)
		action_reward_reshaped = tf.reshape(action_reward, [par['batch_size'], -1])

		total_rewards = []
		for i in range(par['n_pol']):
			rewards_policy = []
			for j in range(n_steps):
				action_stim 	= tf.einsum('ijkm,ij->ikm', self.H_sas, self.latent_pl)
				action_reward 	= tf.einsum('ijkm,ij->ikm', self.H_sar, self.latent_pl)
				if j == 0:
					action = tf.constant(par['action_template'][i])
				else:
					action_dist 	= tf.reduce_sum(action_stim, axis = 2)
					action_index = tf.multinomial(tf.log(1e-5 + action_dist), 1)
					action       = tf.one_hot(tf.squeeze(action_index), par['n_pol'])

				next_stim 		= tf.einsum('ijk,ij->ik', action_stim, action)
				reward 			= tf.einsum('ijk,ij->ik', action_reward, action)
				rewards_policy.append(reward*(gamma**j))
			rewards_policy = tf.stack(rewards_policy, axis = 0)

			total_rewards.append(tf.reduce_sum(rewards_policy, axis = 0))

		total_rewards = tf.stack(total_rewards, axis = 2)
		total_rewards = tf.reshape(total_rewards, [par['batch_size'], -1])
		print('total_rewards', total_rewards)

		return action_reward_reshaped, total_rewards



	def write_hopfield(self):

		norm_lambda  = lambda x : 1e-9 + tf.sqrt(tf.reduce_sum(x**2, axis=1, keepdims=True))

		latent       = self.hop_latent_pl / norm_lambda(self.hop_latent_pl)
		prev_latent  = self.prev_latent_pl / norm_lambda(self.prev_latent_pl)

		state_action = tf.einsum('ij,ik->ijk', prev_latent, self.prev_action_pl)
		H_sas_grad   = tf.einsum('ijk,im->ijkm', state_action, latent)
		H_sar_grad   = tf.einsum('ijk,im->ijkm', state_action, self.prev_reward_pl)

		alpha = 0.9999
		reset_ops  = [\
			tf.assign(self.H_sas, 0.*self.H_sas), \
			tf.assign(self.H_sar, 0.*self.H_sar)  ]
		update_ops = [\
			tf.assign(self.H_sas, alpha*self.H_sas + H_sas_grad), \
			tf.assign(self.H_sar, alpha*self.H_sar + H_sar_grad)  ]

		self.update_hopfield = tf.group(*update_ops)
		self.reset_hopfield = tf.group(*reset_ops)

		with tf.control_dependencies([self.train_RL, self.hopfield_read]):
			self.update_hopfield_with_dep = self.update_hopfield


def main(gpu_id = None):

	# Print out context
	print_important_params()

	# Select GPU
	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	# Reduce memory consumption for GPU 0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) \
		if gpu_id == '3' else tf.GPUOptions()

	# Initialize stimulus environment
	environment = stimulus.Stimulus()

	# Reset graph and designate placeholders
	tf.reset_default_graph()
	stim_pl     	= tf.placeholder(tf.float32, [par['batch_size'], par['n_input']], 'stim')
	prev_latent_pl  = tf.placeholder(tf.float32, [par['batch_size'], par['n_latent']], 'prev_latent')
	latent_pl  		= tf.placeholder(tf.float32, [par['batch_size'], par['n_latent']], 'latent')
	hop_latent_pl   = tf.placeholder(tf.float32, [par['batch_size'], par['n_latent']], 'hp_latent')
	prev_action_pl  = tf.placeholder(tf.float32, [par['batch_size'], par['n_pol']], 'prev_action')
	action_pl    	= tf.placeholder(tf.float32, [par['batch_size'], par['n_pol']], 'action')
	reward_pl 	 	= tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'reward')
	prev_reward_pl 	= tf.placeholder(tf.float32, [par['batch_size'], par['n_unique_vals']], 'prev_reward')
	prev_val_pl 	= tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'prev_val')
	future_val_pl 	= tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'future_val')

	# Start TensorFlow session
	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		# Set up and initialize model on desired device
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(stim_pl, prev_latent_pl, latent_pl, hop_latent_pl, prev_reward_pl, \
				reward_pl, prev_action_pl, action_pl, prev_val_pl, future_val_pl)

		sess.run(tf.global_variables_initializer())

		# Start training loop
		for i in range(par['num_batches']):

			# Reset environment at the start of each iteration
			environment.reset_agents()
			environment.reset_rewards()
			x_read_list = []

			# Pre-allocate prev_val and total_reward
			prev_latent        = np.zeros((par['batch_size'], par['n_latent']), dtype=np.float32)
			prev_action        = np.zeros((par['batch_size'], par['n_pol']), dtype=np.float32)
			prev_reward_scalar = np.zeros((par['batch_size'], par['n_val']), dtype=np.float32)
			prev_reward_matrix = np.zeros((par['batch_size'], par['n_unique_vals']), dtype=np.float32)
			prev_val           = np.zeros((par['batch_size'], par['n_val']), dtype=np.float32)
			total_reward       = np.zeros((par['batch_size'], 1), dtype=np.float32)

			action_record  = []
			H_stim_record  = []
			H_act_f_record = []

			# Iterate through time
			for t in range(par['num_time_steps']):

				# Make inputs
				stim_in = environment.make_inputs()

				# Generate the latent, train encoder weights
				_, reconstruction_loss, latent_loss, latent, latent_mu, latent_log_var, sparsity_loss = \
					sess.run([model.train_encoder, model.reconstruction_loss, model.latent_loss, \
					model.latent, model.latent_mu, model.latent_log_var, model.sparsity_loss], \
					feed_dict = {stim_pl: stim_in})

				agent_location = environment.get_agent_locs()

				# Generate the policy and value functions
				pol, val, hopfield_read = sess.run([model.pol_out, model.val_out, model.hopfield_read], \
					feed_dict = {latent_pl: latent})

				# Choose action, calculate reward and determine next state
				action = np.array([np.random.multinomial(1, pol[t,:]-1e-6) for t in range(par['batch_size'])])
				agent_locs = environment.get_agent_locs()
				reward = environment.agent_action(action)
				reward_one_hot = np.zeros((par['batch_size'], par['n_unique_vals']), dtype = np.float32)
				reward_one_hot[np.arange(par['batch_size']), np.int8(np.squeeze(reward))] = 1.

				if t<0:
					latent /= (1e-9+np.sqrt(np.sum(latent**2, axis = 1, keepdims = True)))
					prev_latent /= (1e-9+np.sqrt(np.sum(prev_latent**2, axis = 1, keepdims = True)))
					z = np.sum(latent*prev_latent,axis = 1)
					print(t, agent_location[0,:], action[0,:], ' Dot prod ', z[0])
					plt.imshow(np.reshape(hopfield_read[0,:],(4,2)), aspect='auto')
					plt.colorbar()
					plt.title('Location ' + str(agent_location[0,:]))
					plt.show()

				#action_record.append(action)

				# Update total reward and prev_val
				total_reward += reward

				# Update the Hopfield network
				sess.run([model.update_hopfield_with_dep, model.train_RL, model.hopfield_read], \
					feed_dict = {latent_pl: prev_latent, action_pl: prev_action, reward_pl: prev_reward_scalar, \
					future_val_pl: val, prev_latent_pl : prev_latent, hop_latent_pl : latent, \
					prev_action_pl: prev_action, prev_reward_pl : prev_reward_matrix})

				prev_latent = latent+0.
				prev_reward_matrix = reward_one_hot+0.
				prev_reward_scalar = reward+0.
				prev_action = action+0.
				prev_val = val+0.

				if t == -1:
					z_read = np.stack(x_read_list, axis = 0)
					fig, ax = plt.subplots(2, 2, figsize=[8,8])
					#ax[0,0].imshow(z_read[:, 0, :], aspect = 'auto')
					#ax[0,1].imshow(z_read[:, 1, :], aspect = 'auto')
					#ax[1,0].imshow(z_read[:, 2, :], aspect = 'auto')
					#ax[1,1].imshow(z_read[:, 3, :], aspect = 'auto')

					ax[0,0].imshow(H_stim_record[0], aspect = 'auto')
					ax[0,1].imshow(H_stim_record[100], aspect = 'auto')
					ax[1,0].imshow(H_stim_record[300], aspect = 'auto')
					ax[1,1].imshow(H_stim_record[-1], aspect = 'auto')

					fig, ax = plt.subplots(2, 2, figsize=[8,8])
					ax[0,0].imshow(H_act_f_record[0], aspect = 'auto')
					ax[0,1].imshow(H_act_f_record[100], aspect = 'auto')
					ax[1,0].imshow(H_act_f_record[300], aspect = 'auto')
					ax[1,1].imshow(H_act_f_record[-1], aspect = 'auto')

					plt.show()

				# Reset agents that have obtained a reward
				environment.reset_agents(reward != 0.)



			# Analyze actions
			#action_record = np.concatenate(action_record, axis=0)
			#action_record = np.round(np.mean(action_record, axis=0), 2).tolist()

			# Output network performance
			if i%10==0:
				H_sas, H_sar = sess.run([model.H_sas, model.H_sar])
				#print('Iter {:>4} | Mean Reward: {:6.3f} | Recon Loss: {:8.6f} | Sparsity Loss: {:8.6f} | Sim act.: {:8.6f} | Action Dist: {}'.format(\
				#	i, np.mean(total_reward), rec_loss, sparsity_loss, sim_active, action_record))
				print('Iter {:>4} | Mean Reward: {:6.3f} | Recon Loss: {:8.6f} | Latent Loss: {:8.6f} |'.format(\
					i, np.mean(total_reward), reconstruction_loss, latent_loss))
				print('          | Latent Mu: {:8.6f} | Latent Var: {:8.6f}  | Sparisty Loss: {:8.6f}'.format(\
					np.mean(latent_mu), np.mean(latent_log_var), sparsity_loss))
				print('          | H_sas', np.mean(H_sas**2), '| H_sar', np.mean(H_sar**2))
			if i%200==0 and par['train_encoder'] and i>0:
				weights = sess.run(model.var_dict)
				results = {'weights': weights, 'latent': latent}
				pickle.dump(results, open('./savedir/VAE_8x8_model_weights3.pkl', 'wb'))

			# Update model weights
			sess.run(model.update_weights)
			sess.run(model.reset_hopfield)

def print_important_params():

	keys = ['save_fn', 'learning_rate', 'n_hidden', 'n_latent', 'hopf_multiplier', \
		'hopf_alpha', 'hopf_neuron_alpha', 'hopf_beta', 'hopf_cycles', 'covariance_method', \
		'drop_rate', 'discount_rate', 'num_time_steps', 'num_batches', 'batch_size', 'rewards', \
		'room_width', 'room_height', 'sparsity_cost', 'weight_cost', 'entropy_cost', \
		'val_cost', 'latent_cost']

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
