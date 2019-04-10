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
from parameters import *
import atari_stimulus as stimulus
import atari_encoder as ae

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

	def __init__(self, stim):

		t0 = time.time()

		self.stim = stim

		self.latent, conv_shapes = \
			ae.encoder(self.stim, par['n_latent'])

		top_k, _ = tf.nn.top_k(self.latent, k=np.int32(par['n_latent']*0.05))
		self.latent = tf.where(self.latent < top_k[:,-2:-1], tf.zeros(self.latent.shape), self.latent)
		self.binary = tf.where(self.latent < top_k[:,-2:-1], tf.zeros(self.latent.shape), tf.ones(self.latent.shape))
		# self.latent = self.latent * (0.1+self.binary)

		print('\nMany frames with bottleneck, using linear\n')

		self.action = tf.random_uniform([par['batch_size'], 6], 0, 1)

		latent_vec = tf.concat([self.latent, self.action], axis=-1)
		self.recon = ae.decoder(latent_vec, conv_shapes)

		self.optimize()


	def optimize(self):

		var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		self.var_dict = {var.op.name : var for var in var_list}

		self.recon_loss = tf.reduce_mean(tf.square(self.stim - self.recon))
		self.latent_loss = 0.01*tf.reduce_mean(tf.abs(self.latent))

		opt = tf.train.AdamOptimizer(5e-4)
		self.train = opt.minimize(self.recon_loss + self.latent_loss)



def main(gpu_id = None):

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
	x = tf.placeholder(tf.float32, [par['batch_size'], 100, 84, 4], 'input')

	# Start TensorFlow session
	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		# Set up and initialize model on desired device
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x)

		sess.run(tf.global_variables_initializer())

		# Start training loop
		print('Starting training.\n')
		for i in range(par['num_batches']):
			
			reward_list = []
			obs = environment.reset_environments()
			
			for t in range(2000):

				_, latent, rec, binary, recon_loss, latent_loss, action = \
					sess.run([model.train, model.latent, model.recon, model.binary, \
						model.recon_loss, model.latent_loss, model.action], \
						feed_dict = {x : obs})

				obs, reward = environment.agent_action(action)
				reward_list.append(reward)

			print('Iter {:>4} | Mean Reward: {:6.3f} | Recon Loss: {:7.5f} | Latent Loss: {:7.5f} |'.format(\
				i, np.mean(reward_list), recon_loss, latent_loss))
			# print(np.reshape(np.sum(binary, axis=0)[:128], [8, 16]).astype(np.int32))

			fig, ax = plt.subplots(1,2)
			ax[0].imshow(obs[0,...,-1], aspect='auto', cmap='gray')
			ax[1].imshow(rec[0,...,-1], aspect='auto', cmap='gray')
			plt.savefig('./savedir/2000_frames_bottle_linear_iter{}_recon.png'.format(i))
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
