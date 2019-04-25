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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model modules
from AdamOpt import AdamOpt
import atari_stimulus as stimulus
from atari_parameters import par
import atari_encoder as ae
import striatum

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def dense_layer(x, n_out, name, activation=tf.nn.relu):
	""" Build a dense layer with RELU activation
		x		: input tensor to propagate
		n_out	: number of neurons in layer
		name	: name of the layer
	"""

	n_in = x.shape.as_list()[-1]
	W = tf.get_variable('W_'+name, shape=[n_in, n_out])
	b = tf.get_variable('b_'+name, shape=[1, n_out])

	y = x @ W + b

	return activation(y)


class Model:

	def __init__(self, stim, gate, reward, action, future_val, terminal_state):

		# Gather placeholders
		self.stim = stim
		self.gate = gate
		self.reward = reward
		self.action = action
		self.future_val = future_val
		self.terminal_state = terminal_state

		# Run encoder
		self.latent, _ = ae.encoder(self.stim, par['n_latent'], \
			var_dict=par['loaded_var_dict'], trainable=par['train_encoder'])

		# Project the latent encoding to the policy and value outputs
		h0 = self.gate*dense_layer(self.latent, 1024, 'h0')

		self.pol = dense_layer(h0, par['n_pol'], 'pol', activation=tf.identity)
		self.val = dense_layer(h0, par['n_val'], 'val', activation=tf.identity)

		# Apply softmax to policy
		self.pol = tf.nn.softmax(self.pol, axis=1)

		# Run optimizer
		self.optimize()

	def optimize(self):

		epsilon = 1e-6

		# Collect and list all variables in the model
		var_list = tf.trainable_variables()
		self.var_dict = {var.op.name : var for var in var_list}
		print('Variables:')
		[print(var.op.name.ljust(20), ':', var.shape) for var in var_list]
		print()

		# Make optimizer
		# opt = tf.train.AdamOptimizer(par['learning_rate'])
		opt = AdamOpt(tf.trainable_variables(), par['learning_rate'])

		# Calculate RL quantities
		pred_val  = self.reward + par['discount_rate']*self.future_val*(1-self.terminal_state)
		advantage = pred_val - self.val

		# Stop gradients where necessary
		advantage_static = tf.stop_gradient(advantage)
		pred_val_static = tf.stop_gradient(pred_val)

		# Calculate RL losses
		self.pol_loss     = -tf.reduce_mean(advantage_static*self.action*tf.log(self.pol + epsilon))
		self.val_loss     =  tf.reduce_mean(tf.square(self.val - pred_val_static))
		self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.pol*tf.log(self.pol + epsilon), axis=1))

		total_loss = self.pol_loss + par['val_cost'] * self.val_loss - par['entropy_cost'] * self.entropy_loss

		# Make update operations for gradient applications
		self.update_grads = opt.compute_gradients(total_loss)
		self.grads = opt.return_delta_grads()

		# Make apply operations for gradient applications
		self.apply_grads = opt.update_weights()


def main(gpu_id=None):

	print_key_params()

	# Select GPU
	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	# Reduce memory consumption for GPU 0
	base_frac = 0.45
	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=base_frac/2) \
	# 	if gpu_id == '3' else tf.GPUOptions(per_process_gpu_memory_fraction=base_frac)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=base_frac)

	# Initialize stimulus environment and obtain first observations
	environment = stimulus.Stimulus()
	obs = environment.reset_environments()

	# Reset graph and designate placeholders
	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['batch_size'], 100, 84, 4], 'input')
	g = tf.placeholder(tf.float32, [par['batch_size'], 1024], 'gate')
	a = tf.placeholder(tf.float32, [par['batch_size'], par['n_pol']], 'action')
	f = tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'future_val')
	t = tf.placeholder(tf.float32, [par['batch_size'], 1], 'terminal_state')
	r = tf.placeholder(tf.float32, [par['batch_size'], 1], 'reward')

	# Start TensorFlow session
	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		# Set up and initialize model on desired device
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, g, r, a, f, t)
		sess.run(tf.global_variables_initializer())

		# Make lists for recording model performance
		recon_loss_record  = []
		latent_loss_record = []
		reward_list = []

		# Start training loop
		print('Starting training.\n')

		for fr in range(par['num_frames']):

			# Update dropout gating to simulate changing to a new agent
			# Update weights after changing gates
			if fr%1000 == 0:
				drop_rate = 0.5
				gate = np.random.choice([0,1/(1-drop_rate)], \
				p=[drop_rate, 1-drop_rate], \
				size=[par['batch_size'], 1024])
				
				if fr != 0:
					print('Applying accumulated gradients.\n')
					sess.run(model.apply_grads)

			# Run the model to obtain policy and value
			pol, val = sess.run([model.pol, model.val], feed_dict={x:obs, g:gate})

			# Choose action
			action = np.array(np.stack([np.random.multinomial(1, pol[i,:]-1e-6) for i in range(par['batch_size'])]))

			# Generate next four frames
			reward = np.zeros([par['batch_size'], 1], dtype=np.float32)
			done   = np.zeros([par['batch_size'], 1], dtype=np.float32)
			for _ in range(par['k_skip']):
				new_obs, reward_frame, done_frame = environment.agent_action(action)
				reward += reward_frame
				done   += done_frame
				reward_list.append(reward)

			# Make sure [done <= 1]
			done = np.minimum(1., done)

			# Only record the last 1000 entries in the reward list
			if len(reward_list) >= 1000: reward_list.pop(0)

			# Calculate the future value function of then ext four frames
			future_val = sess.run(model.val, feed_dict={x:new_obs, g:gate})

			# Train the model
			_, pol_loss, val_loss, ent_loss = \
				sess.run([model.update_grads, model.pol_loss, model.val_loss, model.entropy_loss], \
				feed_dict={x:obs, g:gate, a:action, r:reward, f:future_val, t:done})

			# Update observation window
			obs = new_obs

			# Display model performance
			if fr%500 == 0:
				print('Frame {:>7} | Hist Rew: {:6.3f} | Frame Rew: {:6.3f} | Mean Policy: {} |'.format(\
					fr, np.mean(reward_list), np.mean(reward), np.round(np.mean(pol, axis=0), 2)))
				print('              | Pol Loss: {:5.3f} | Val Loss: {:5.3f} | Ent Loss: {:5.3f} | Future Val: {:6.3f}'.format(\
					pol_loss, val_loss, ent_loss, np.mean(future_val)))



def print_key_params():

	key_params = ['savefn', 'learning_rate', 'discount_rate',\
		'entropy_cost', 'val_cost', 'batch_size']
	print('Key Parameters:\n'+'-'*60)
	for k in key_params:
		print(k.ljust(30), ':', par[k])
	print('-'*60)


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
