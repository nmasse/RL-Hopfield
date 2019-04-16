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

	def __init__(self, stim, boost_d):

		# Gather placeholders
		self.stim = stim
		self.boost_d = boost_d

		# Run encoder
		self.latent, conv_shapes = ae.encoder(self.stim, par['n_latent'], var_dict=None, trainable=par['train_autoencoder'])

		# Calculate post-boost top_k
		boosted_latent = self.latent * tf.exp(par['boost_level']*(par['num_k']/par['n_latent'] - self.boost_d))
		top_k, _ = tf.nn.top_k(boosted_latent, k=par['num_k'])

		# Filter the latent encoding
		self.binary = tf.where(boosted_latent >= top_k[:,par['num_k']-1:par['num_k']], tf.ones(self.latent.shape), tf.zeros(self.latent.shape))
		self.latent = tf.where(boosted_latent >= top_k[:,par['num_k']-1:par['num_k']], self.latent, tf.zeros(self.latent.shape))

		# Randomly select an action
		self.action = tf.random_uniform([par['batch_size'], 6], 0, 1)

		# Concatenate the latent encoding and the action
		latent_vec = tf.concat([self.latent, self.action], axis=-1)

		# Run decoder
		self.recon = ae.decoder(latent_vec, conv_shapes, var_dict=None, trainable=par['train_autoencoder'])

		# Run optimizer
		self.optimize()


	def optimize(self):

		# Collect all variables in the model and list them out
		var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		self.var_dict = {var.op.name : var for var in var_list}
		print('Variables:')
		[print(var.op.name.ljust(20), ':', var.shape) for var in var_list]
		print()

		# Make optimizer
		opt = tf.train.AdamOptimizer(par['learning_rate'])

		# Calculate losses
		self.recon_loss = tf.reduce_mean(tf.square(self.stim - self.recon))
		self.latent_loss = 1e-3*tf.reduce_mean(tf.square(self.latent))

		# Aggregate loss and run optimizer
		total_loss = self.recon_loss + self.latent_loss
		self.train = opt.minimize(total_loss)


def main(gpu_id=None):

	print('Saving to:', par['savefn'])

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
	x = tf.placeholder(tf.float32, [par['batch_size'], 100, 84, 4], 'input')
	d = tf.placeholder(tf.float32, [par['batch_size'], par['n_latent']], 'boost_d')

	# Start TensorFlow session
	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		# Set up and initialize model on desired device
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, d)
		sess.run(tf.global_variables_initializer())

		# Start training loop
		print('Starting training.\n')
		for i in range(par['num_batches']):
			
			reward_list = []
			obs = environment.reset_environments()
			t_range = par['frames_per_iter'] // par['k_skip']
			
			stored_action = np.random.rand(par['batch_size'], 6)
			duty = (par['num_k']/par['n_latent'])*np.ones([par['batch_size'], par['n_latent']]).astype(np.float32)
			for t in range(t_range):

				# Run the model
				_, latent, rec, binary, recon_loss, latent_loss, action = \
					sess.run([model.train, model.latent, model.recon, model.binary, \
						model.recon_loss, model.latent_loss, model.action], \
						feed_dict = {x : obs, d : duty})

				# Update boost duty cycle calculation
				duty = (1-par['boost_alpha']) * binary + par['boost_alpha'] * duty

				# Have first four frames
				last_obs = obs

				# Generate next four frames
				for _ in range(par['k_skip']):
					obs, reward = environment.agent_action(stored_action)
					reward_list.append(reward)

				# Do a sess.run here to train the predictive autoencoder
				# _, pred = sess.run([model.train_enc, model.pred], feed_dict = {x : last_obs})


			print('Iter {:>4} | MR: {:6.3f} | Recon Loss: {:7.5f} | Latent Loss: {:7.5f} |'.format(\
				i, np.mean(reward_list), recon_loss, latent_loss))

			c = np.sum(binary, axis=0)
			uniques, counts = np.unique(c, return_counts=True)
			for u, c in zip(uniques.astype(np.int32), counts):
				print('Frq: {:>4}/{:} | Occ: {:>3}'.format(u,par['batch_size'],c))
			print('')

			if i%5 == 0:

				obs = last_obs

				fig, ax = plt.subplots(2,4,figsize=(12,8))
				ax[0,0].imshow(obs[0,...,0], aspect='auto', cmap='gray', clim=(obs[0].min(),obs[0].max()))
				ax[0,1].imshow(obs[0,...,1], aspect='auto', cmap='gray', clim=(obs[0].min(),obs[0].max()))
				ax[0,2].imshow(obs[0,...,2], aspect='auto', cmap='gray', clim=(obs[0].min(),obs[0].max()))
				ax[0,3].imshow(obs[0,...,3], aspect='auto', cmap='gray', clim=(obs[0].min(),obs[0].max()))
				ax[1,0].imshow(rec[0,...,0], aspect='auto', cmap='gray', clim=(rec[0].min(),rec[0].max()))
				ax[1,1].imshow(rec[0,...,1], aspect='auto', cmap='gray', clim=(rec[0].min(),rec[0].max()))
				ax[1,2].imshow(rec[0,...,2], aspect='auto', cmap='gray', clim=(rec[0].min(),rec[0].max()))
				ax[1,3].imshow(rec[0,...,3], aspect='auto', cmap='gray', clim=(rec[0].min(),rec[0].max()))

				ax[0,0].set_title('Frame {}'.format(t*par['k_skip']-3))
				ax[0,1].set_title('Frame {}'.format(t*par['k_skip']-2))
				ax[0,2].set_title('Frame {}'.format(t*par['k_skip']-1))
				ax[0,3].set_title('Frame {}'.format(t*par['k_skip']))

				ax[0,0].set_ylabel('Observation')
				ax[1,0].set_ylabel('Reconstruction')

				for j in range(8):
					ax[j//4,j%4].set_xticks([])
					ax[j//4,j%4].set_yticks([])

				plt.savefig('./savedir/'+par['savefn']+'_iter{:0>5}_recon.png'.format(i), bbox_inches='tight')
				plt.clf()
				plt.close()



if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
