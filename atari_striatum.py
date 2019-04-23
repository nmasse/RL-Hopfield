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

	def __init__(self, stim, boost_d, reward, action, future_val, terminal_state):

		# Gather placeholders
		self.stim = stim
		self.boost_d = boost_d
		self.reward = reward
		self.action = action
		self.future_val = future_val
		self.terminal_state = terminal_state

		self.striatum = striatum.Network()

		# Run encoder
		self.latent, conv_shapes = ae.encoder(self.stim, par['n_latent'], \
			var_dict=par['loaded_var_dict'], trainable=par['train_encoder'])

		# Calculate post-boost top_k
		boosted_latent = self.latent * \
			tf.exp(par['boost_level']*(par['num_k']/par['n_latent'] - self.boost_d))
		top_k, _ = tf.nn.top_k(boosted_latent, k=par['num_k'])

		# Filter the latent encoding
		boost_cond = boosted_latent >= top_k[:,par['num_k']-1:par['num_k']]
		self.binary = tf.where(boost_cond, tf.ones(self.latent.shape), tf.zeros(self.latent.shape))
		self.latent = tf.where(boost_cond, self.latent, tf.zeros(self.latent.shape))

		self.y, self.update_traces, self.update_weights, \
			self.normalize_weights = self.striatum.run(self.latent, self.reward)
		#z = dense_layer(y, par['n_out'], 'out')
		z = dense_layer(tf.concat([0.*self.y, self.latent], axis = 1), par['n_out'], 'out')
		z = tf.layers.dropout(z, rate = par['drop_rate'], training = True)
		self.pol = dense_layer(z, par['n_pol'], 'pol', activation = tf.identity)
		self.pol = tf.nn.softmax(self.pol, axis = 1)
		self.val = dense_layer(z, par['n_val'], 'val', activation = tf.identity)

		self.W_pos, self.W_neg, self.W_trace_pos, self.W_trace_neg = self.striatum.return_weights()

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
		opt = tf.train.AdamOptimizer(par['learning_rate'])

		pred_val = self.reward + par['discount_rate']*self.val*(1-self.terminal_state)
		advantage = pred_val - self.future_val

		pol_loss = -tf.reduce_mean(advantage*self.action*tf.log(self.pol + epsilon))

		val_loss = tf.reduce_mean(tf.square(advantage))

		entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.pol*tf.log(self.pol + epsilon), axis = 1))

		loss = pol_loss + par['val_cost'] * val_loss - par['entropy_cost'] * entropy_loss

		self.train = opt.minimize(loss)


def main(gpu_id=None):

	print_key_params()

	# Select GPU
	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	# Reduce memory consumption for GPU 0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) \
		if gpu_id == '3' else tf.GPUOptions()

	# Initialize stimulus environment and obtain first observations
	environment = stimulus.Stimulus()
	obs = environment.reset_environments()

	# Reset graph and designate placeholders
	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['batch_size'], 100, 84, 4], 'input')
	d = tf.placeholder(tf.float32, [1, par['n_latent']], 'boost_d')
	r = tf.placeholder(tf.float32, [par['batch_size'], 1], 'reward')
	a = tf.placeholder(tf.float32, [par['batch_size'], par['n_pol']], 'action')
	f = tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'future_val')
	ts = tf.placeholder(tf.float32, [par['batch_size'], 1], 'terminal_state')

	# Start TensorFlow session
	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		# Set up and initialize model on desired device
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, d, r, a, f, ts)
		sess.run(tf.global_variables_initializer())

		# Make lists for recording model performance
		recon_loss_record  = []
		latent_loss_record = []

		# Start training loop
		print('Starting training.\n')
		duty = (par['num_k']/par['n_latent'])*np.ones([1, par['n_latent']]).astype(np.float32)
		reward = np.zeros([par['batch_size'], 1], np.float32)
		#val = np.zeros([par['batch_size'], 1], np.float32)

		reward_list = []

		for fr in range(par['num_frames']//par['k_skip']):

			# Run the model
			pol, val, binary, y, _, _, _ = sess.run([model.pol, model.val, model.binary, \
				model.y, model.update_traces, model.update_weights, model.normalize_weights], \
				feed_dict = {x : obs, d : duty, r : reward})

			# Update boost duty cycle calculation
			duty = (1-par['boost_alpha']) * duty  \
				+ par['boost_alpha'] * np.mean(binary, axis = 0, keepdims = True)

			# choose action, determine reward
			#print('pol', pol.shape)
			action = np.array(np.stack([np.random.multinomial(1, pol[i,:]-1e-6) for i in range(par['batch_size'])]))

			# Generate next four frames
			reward = np.zeros((par['batch_size'], 1))
			done = np.zeros((par['batch_size'], 1))
			for _ in range(par['k_skip']):
				new_obs, reward_frame, done_frame = environment.agent_action(action)
				reward += reward_frame
				done += done_frame
				reward_list.append(reward)

			done = np.minimum(1., done)
			if len(reward_list) >= 1000:
				reward_list = reward_list[1:]

			# calculate the value function of the next four frames
			future_val = sess.run(model.val, feed_dict = {x : new_obs, d : duty})

			# train the model
			sess.run([model.train], feed_dict = {x : obs, d : duty, a: action, \
				r : reward, f: future_val, ts: done})

			obs = new_obs
			if fr%200==0:
				W_pos, W_neg, W_trace_pos, W_trace_neg = \
					sess.run([model.W_pos, model.W_neg, model.W_trace_pos, model.W_trace_neg])
				display_data(obs, W_pos, W_neg, W_trace_pos, W_trace_neg, pol, \
					reward, reward_list, y, fr)



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
	"""

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

def print_key_params():

	key_params = ['savefn', 'striatum_th', 'trace_th', 'learning_rate', 'discount_rate',\
		'entropy_cost', 'val_cost', 'prop_top', 'drop_rate','batch_size']
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
