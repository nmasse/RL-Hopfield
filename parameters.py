### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
import pickle
import matplotlib.pyplot as plt
from itertools import product

print('\n--> Loading parameters...')

##############################
### Independent parameters ###
##############################

par = {
	# Setup parameters
	'savedir'				: './savedir/',
	'save_fn'				: 'maze_4x4_test',
	'LSTM_init'				: 0.02,
	'w_init'				: 0.02,
	'RL_method'				: 'policy',

	# Network shape
	'num_nav_tuned'			: 4,		# Must be 4 for maze task (replaced motion neurons)
	'num_fix_tuned'			: 0,
	'num_rule_tuned'		: 0,
	'num_rew_tuned'			: 1,		# For reward vector in maze task
	'n_hidden'				: [200, 200],
	'n_encoding'			: [64, 128],
	'n_pol'					: 4,
	'n_val'					: 1,
	'n_latent'				: 128,
	'n_dend'				: 25,
	'act_mult'				: 4,
	'dend_th'				: 8,

	# Hopfield configuration
	'hopf_alpha'			: 0.95,
	'hopf_neuron_alpha'		: 0.9,
	'hopf_beta'				: 1.,
	'hopf_cycles'			: 6,
	'covariance_method'		: True,

	# Timings and rates
	'learning_rate'			: 5e-5,
	'drop_rate'				: 0.5,
	'discount_rate'			: 0.9,

	# Task specs
	'num_time_steps'		: 1,
	'num_batches'			: 10000000,

	# Maze task specs
	'rewards'				: [1.],
	'num_actions'			: 4,		# The number of different actions available to the agent
	'room_width'			: 8,
	'room_height'			: 8,
	'use_default_rew_locs'	: False,
	'movement_penalty'		: 0.,

	# Cost values
	'sparsity_cost'         : 2e-1, # was 1e-2
	'weight_cost'           : 1e-6,  # was 1e-6
	'entropy_cost'          : 0.05,
	'val_cost'              : 0.01,
	'latent_cost'			: 0.00000,

	# Training specs
	'batch_size'			: 1,
	'train_encoder'			: True,
	'load_encoder_weights'	: False,

}


############################
### Dependent parameters ###
############################

def update_parameters(updates, verbose=True, update_deps=True):
	""" Updates parameters based on a provided
		dictionary, then updates dependencies """

	par.update(updates)
	if verbose:
		print('Updating parameters:')
		for (key, val) in updates.items():
			print('{:<24} --> {}'.format(key, val))

	if update_deps:
		update_dependencies()



def load_encoder_weights():

	fn = './savedir/VAE_8x8_n32_model_weights.pkl'
	results = pickle.load(open(fn, 'rb'))


	print('Weight keys ', results['weights'].keys())
	par['W_enc0_init'] = results['weights']['W_enc0']
	par['W_enc1_init'] = results['weights']['W_enc1']
	par['W_dec0_init'] = results['weights']['W_dec0']
	par['W_dec1_init'] = results['weights']['W_dec1']
	par['W_dec2_init'] = results['weights']['W_dec2']
	par['W_mu_init'] = results['weights']['W_mu']
	par['b_enc0_init'] = results['weights']['b_enc0']
	par['b_enc1_init'] = results['weights']['b_enc1']
	par['b_dec0_init'] = results['weights']['b_dec0']
	par['b_dec1_init'] = results['weights']['b_dec1']
	par['b_dec2_init'] = results['weights']['b_dec2']
	par['b_mu_init'] = results['weights']['b_mu']



def update_weights(var_dict):

	print('Setting weight values manually; disabling training and weight saving.')
	par['train'] = False
	par['save_weights'] = False
	for key, val in var_dict['weights'].items():
		print(key, val.shape)
		if not 'A_' in key:
			par[key+'_init'] = val


def update_dependencies():
	""" Updates all parameter dependencies """

	# Set input and output sizes
	par['n_input']  = par['num_nav_tuned'] + par['num_fix_tuned'] + par['num_rew_tuned'] + par['num_rule_tuned']
	par['n_pol'] = par['num_actions']

	par['n_unique_vals'] = len(par['rewards']) + 1
	par['n_val'] = 4 if par['RL_method'] == 'Q-learning' else 1


	# Specify one-hot vectors matching with each reward for maze task
	condition = True
	while condition:
		par['reward_vectors'] = np.random.choice([0,1], size=[len(par['rewards']), par['num_rew_tuned']])
		condition = (np.mean(np.std(par['reward_vectors'], axis=0)) == 0.) and len(par['rewards']) > 1

	c = 0.2

	if par['load_encoder_weights']:
		load_encoder_weights()
	else:
		par['W_enc0_init'] = np.random.uniform(-c, c, size=[par['n_input'], par['n_encoding'][0]]).astype(np.float32)
		par['W_enc1_init'] = np.random.uniform(-c, c, size=[par['n_encoding'][0], par['n_encoding'][1]]).astype(np.float32)
		par['W_mu_init'] = np.random.uniform(-c, c, size=[par['n_encoding'][1], par['n_latent']]).astype(np.float32)
		#par['W_mu_init'] = np.random.uniform(-c, c, size=[par['n_encoding'][0], par['n_latent']]).astype(np.float32)
		par['W_sigma_init'] = np.random.uniform(-c, c, size=[par['n_encoding'][1], par['n_latent']]).astype(np.float32)
		par['W_dec0_init'] = np.random.uniform(-c, c, size=[par['n_latent'], par['n_encoding'][0]]).astype(np.float32)
		par['W_dec1_init'] = np.random.uniform(-c, c, size=[par['n_encoding'][0], par['n_encoding'][1]]).astype(np.float32)
		#par['W_dec1_init'] = np.random.uniform(-c, c, size=[par['n_encoding'][0], par['n_input']]).astype(np.float32)
		par['W_dec2_init'] = np.random.uniform(-c, c, size=[par['n_encoding'][1], par['n_input']]).astype(np.float32)
		#par['W_dec2_init'] = np.random.uniform(-c, c, size=[par['n_encoding'][0], par['n_input']]).astype(np.float32)
		par['b_enc0_init'] = np.zeros([1, par['n_encoding'][0]], dtype=np.float32)
		par['b_enc1_init'] = np.zeros([1, par['n_encoding'][1]], dtype=np.float32)
		par['b_dec0_init'] = np.zeros([1, par['n_encoding'][0]], dtype=np.float32)
		par['b_dec1_init'] = np.zeros([1, par['n_encoding'][1]], dtype=np.float32)
		#par['b_dec1_init'] = np.zeros([1, par['n_input']], dtype=np.float32)
		par['b_dec2_init'] = np.zeros([1, par['n_input']], dtype=np.float32)
		par['b_mu_init'] = np.zeros([1, par['n_latent']], dtype=np.float32)
		par['b_sigma_init'] = np.zeros([1, par['n_latent']], dtype=np.float32)

	N = (len(par['rewards']) + 1)*par['n_pol'] + par['n_latent'] + 8 + 8
	par['W0_init'] = np.random.uniform(-c, c, size=[N, par['n_hidden'][0]]).astype(np.float32)
	par['W1_init'] = np.random.uniform(-c, c, size=[par['n_hidden'][0], par['n_hidden'][1]]).astype(np.float32)
	par['b0_init'] = np.zeros([1, par['n_hidden'][0]], dtype=np.float32)
	par['b1_init'] = np.zeros([1, par['n_hidden'][1]], dtype=np.float32)


	par['W_pol_init'] = np.random.uniform(-c, c, size=[par['n_hidden'][1], par['n_pol']]).astype(np.float32)
	par['b_pol_init'] = np.zeros([1,par['n_pol']], dtype=np.float32)
	par['W_val_init'] = np.random.uniform(-c, c, size=[par['n_hidden'][1], par['n_val']]).astype(np.float32)
	par['b_val_init'] = np.zeros([1,par['n_val']], dtype=np.float32)

	"""
	par['W_stim_write'] = np.zeros([par['num_time_steps'], par['n_latent'], par['n_hopf_stim']], dtype=np.float32)
	for i,j in product(range(par['num_time_steps']), range(par['n_latent'])):
		for _ in range(1):
			#k = np.random.randint(par['hopf_multiplier'])
			k = i%par['hopf_multiplier']
			par['W_stim_write'][i, j, k + j*par['hopf_multiplier']] = 1.

	N = 2*par['n_pol']
	par['W_act_write'] = np.zeros([par['num_time_steps'], N, par['n_hopf_act']], dtype=np.float32)
	for i,j in product(range(par['num_time_steps']), range(N)):
		for _ in range(1):
			#k = np.random.randint(par['hopf_multiplier'])
			k = i%par['hopf_multiplier']
			par['W_act_write'][i, j, k + j*par['hopf_multiplier']] = 1.

	par['W_stim_read'] = np.float32(np.mean(par['W_stim_write'], axis = 0) > 0)


	par['H_stim_mask'] = np.ones((par['n_hopf_stim'], par['n_hopf_stim']), dtype = np.float32) \
		- np.eye((par['n_hopf_stim']), dtype = np.float32)
	#for i,j,k in product(range(par['n_latent']), range(par['hopf_multiplier']), range(par['hopf_multiplier'])):
	#	m = i*par['hopf_multiplier']
	#	par['H_stim_mask'][m+j,m+k] = 0.
	"""
	"""
	plt.imshow(par['W_stim_write'][0,:,:], aspect = 'auto')
	plt.colorbar()
	plt.show()
	plt.imshow(par['W_stim_read'][:,:], aspect = 'auto')
	plt.colorbar()
	plt.show()
	"""

	par['action_template'] = []
	for i in range(par['n_pol']):
		p = np.zeros((par['batch_size'], par['n_pol']), dtype = np.float32)
		p[:, i] = 1.
		par['action_template'].append(p)



update_dependencies()
print('--> Parameters successfully loaded.\n')
