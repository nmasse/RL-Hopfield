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
import atari_stimulus as stimulus
from atari_parameters import par
import atari_encoder as ae
import striatum
import time

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)
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

	def __init__(self, stim, reward, action, future_val, terminal_state, gate, step):

		# Gather placeholders
		self.stim = stim
		self.reward = reward
		self.action = action
		self.future_val = future_val
		self.terminal_state = terminal_state
		self.gate = gate
		self.step = step

		#self.striatum = striatum.Network()

		# Run encoder
		flat, conv_shapes = ae.encoder(self.stim, par['n_latent'], \
			var_dict=par['loaded_var_dict'], trainable=par['train_encoder'])
		z = dense_layer(flat, par['n_latent'], 'out')
		z = self.gate * z

		self.pol = dense_layer(z, par['n_pol'], 'pol', activation = tf.identity)
		self.pol = tf.nn.softmax(self.pol, axis = 1)
		self.val = dense_layer(z, par['n_val'], 'val', activation = tf.identity)

		# Run optimizer
		self.optimize()


	def optimize(self):

		epsilon = 1e-6

		# Collect all variables in the model and list them out
		var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		self.var_dict = {var.op.name : var for var in var_list}
		print('Variables:')
		[print(var.op.name.ljust(20), ':', var.shape) for var in var_list]
		print()

		# Make optimizer
		opt = AdamOpt.AdamOpt(var_list, learning_rate = par['learning_rate'])

		pred_val = self.reward + (par['discount_rate']**self.step)*self.future_val*(1. - self.terminal_state)
		advantage = pred_val - self.val

		pol_loss = -tf.reduce_mean(tf.stop_gradient(advantage)*self.action*tf.log(self.pol + epsilon))

		val_loss = tf.reduce_mean(tf.square(advantage))

		entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.pol*tf.log(self.pol + epsilon), axis = 1))

		loss = pol_loss + par['val_cost'] * val_loss - par['entropy_cost'] * entropy_loss

		self.update_grads = opt.compute_gradients(loss, apply_gradients = False)

		self.update_weights = opt.update_weights()




def main(gpu_id=None):

	print_key_params()

	# Select GPU
	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	# Reduce memory consumption for GPU 0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)# \
#		if gpu_id == '3' else tf.GPUOptions()

	# Initialize stimulus environment and obtain first observations
	environment = stimulus.Stimulus()
	obs = environment.reset_environments()

	# Reset graph and designate placeholders
	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['batch_size'], 100, 84, 4], 'input')
	r = tf.placeholder(tf.float32, [par['batch_size'], 1], 'reward')
	a = tf.placeholder(tf.float32, [par['batch_size'], par['n_pol']], 'action')
	f = tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'future_val')
	g = tf.placeholder(tf.float32, [par['batch_size'], par['n_latent']], 'gate')
	ts = tf.placeholder(tf.float32, [par['batch_size'], 1], 'terminal_state')
	s = tf.placeholder(tf.float32, [], 'step')

	# Start TensorFlow session
	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		# Set up and initialize model on desired device
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, r, a, f, ts, g, s)
		sess.run(tf.global_variables_initializer())


		# Start training loop
		print('Starting training.\n')
		reward_list_full = []

		high_score        = np.zeros([par['batch_size'], 1])
		agent_score       = np.zeros([par['batch_size'], 1])
		final_agent_score = np.zeros([par['batch_size'], 1])
		reward_list = []
		obs_list = []
		action_list = []
		value_list = []
		done_list = []

		for fr in range(par['num_frames']//par['k_skip']):

			if fr%200==0:
				gate = np.random.choice([0. , 1/(1-par['drop_rate'])], size = [par['batch_size'], \
					par['n_latent']], p=[par['drop_rate'], 1 - par['drop_rate']])
				#gate = np.ones_like(gate)

			# Run the model
			pol, val = sess.run([model.pol, model.val], feed_dict = {x : obs, g:gate})
			obs_list.append(obs)

			# choose action, determine reward
			action = np.array(np.stack([np.random.multinomial(1, pol[i,:]-1e-6) for i in range(par['batch_size'])]))
			action_list.append(action)

			# Generate next four frames
			reward = np.zeros((par['batch_size'], 1))
			done   = np.zeros((par['batch_size'], 1))
			for _ in range(par['k_skip']):
				obs, reward_frame, reward_sign_frame, done_frame = environment.agent_action(action)
				#reward += reward_frame
				reward += reward_sign_frame
				done   += done_frame

				# Update the score by adding the current reward
				agent_score += reward_frame

				# Update final agent score and zero out agent score if the
				# environment resets
				final_agent_score = final_agent_score*(1-done_frame) + agent_score*done_frame
				agent_score *= (1-done_frame)

				# Record overall high scores for each agent
				high_score = np.maximum(high_score, agent_score)

			reward_list.append(reward)
			reward_list_full.append(reward)
			done_list.append( np.minimum(1., done))

			if len(obs_list) == par['n-step']+1:
				for n in range(par['n-step']):

					reward = np.zeros((par['batch_size'], 1))
					done = np.zeros((par['batch_size'], 1))
					for k in range(n+1):
						done += done_list[par['n-step']-k-1]
						reward += reward_list[par['n-step']-k-1]*par['discount_rate']**(n-k)
					done = np.minimum(1., done)

					# train the model
					sess.run(model.update_grads, feed_dict = {x : obs_list[-2-n], \
						a: action_list[-2-n], r : reward, f: val, ts: done, g:gate, s:n+1})

				sess.run(model.update_weights)

				obs_list = []
				action_list = []
				reward_list = []
				done_list = []


			if len(reward_list_full) >= 1000:
				reward_list_full = reward_list_full[1:]
			if fr%100==0:
				print('Frame {:>7} | Policy {} | Reward {:5.3f} | Overall HS: {:>4} | Current HS: {:>4} | Mean Final HS: {:7.2f}'.format(\
					fr, np.round(np.mean(pol,axis=0),2), np.mean(reward_list_full), int(high_score.max()), int(agent_score.max()), np.mean(final_agent_score)))


def display_data(obs, W_pos, W_neg, W_trace_pos, W_trace_neg, pol, reward, reward_list, y, t):

	"""
	plt.imshow(W_trace_pos[0,:,:], aspect = 'auto')
	plt.colorbar()
	plt.title('Weight traces')
	plt.show()

	plt.imshow(W_pos, aspect = 'auto')
	plt.colorbar()
	plt.title('Weights')
	plt.show()


	W_trace_pos = np.mean(W_trace_pos,axis=0)
	W_trace_neg = np.mean(W_trace_neg,axis=0)
	print('mean pol ', np.mean(pol, axis = 0), 'mean reward ', np.mean(reward))
	print('Frame {:>4} | y>0: {:6.3f} | W_pos: {:6.3f} | W_neg {:6.3f} | W_t_pos: {:6.3f} | W_t_neg {:6.3f} | MR: {:6.3f}'.format(\
		t*par['k_skip'], np.mean(y>0), np.mean(W_pos>0),np.mean(W_neg>0),np.mean(W_trace_pos>0),\
		np.mean(W_trace_neg>0), np.mean(reward_list)))


	fig, ax = plt.subplots(2,4,figsize=(12,8))
	ax[0,0].imshow(obs[0,...,0], aspect='auto', cmap='gray', clim=(obs[0].min(),obs[0].max()))
	ax[0,1].imshow(obs[0,...,1], aspect='auto', cmap='gray', clim=(obs[0].min(),obs[0].max()))
	ax[0,2].imshow(obs[0,...,2], aspect='auto', cmap='gray', clim=(obs[0].min(),obs[0].max()))
	ax[0,3].imshow(obs[0,...,3], aspect='auto', cmap='gray', clim=(obs[0].min(),obs[0].max()))
	ax[1,0].imshow(W_pos, aspect='auto', cmap='gray', clim=(W_pos.min(),W_pos.max()))
	ax[1,1].imshow(W_neg, aspect='auto', cmap='gray', clim=(W_neg.min(),W_neg.max()))

	ax[1,2].imshow(W_trace_pos, aspect='auto', cmap='gray', clim=(W_trace_pos.min(),W_trace_pos.max()))
	ax[1,3].imshow(W_trace_neg, aspect='auto', cmap='gray', clim=(W_trace_neg.min(),W_trace_neg.max()))


	plt.suptitle('Frame {} Striatum'.format(t*par['k_skip']))
	plt.savefig(par['plotdir']+par['savefn']+'_recon.png', bbox_inches='tight')
	plt.clf()
	plt.close()
	"""
def print_key_params():

	key_params = ['savefn', 'striatum_th', 'trace_th', 'learning_rate', 'discount_rate',\
		'entropy_cost', 'val_cost', 'prop_top', 'drop_rate','batch_size','n-step']
	print('Key parameters...')
	for k in key_params:
		print(k, ': ', par[k])


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
