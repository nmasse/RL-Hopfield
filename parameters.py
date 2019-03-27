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
	'save_fn'				: 'maze_v1',
	'LSTM_init'				: 0.02,
	'w_init'				: 0.02,

	# Network shape
	'num_nav_tuned'			: 4,		# Must be 4 for maze task (replaced motion neurons)
	'num_fix_tuned'			: 1,
	'num_rule_tuned'		: 0,
	'num_rew_tuned'			: 10,		# For reward vector in maze task
	'n_hidden'				: [100, 100],
	'n_pol'					: 5,
	'n_val'					: 1,
	'n_latent'				: 40,
	'hopf_multiplier'		: 10,

	# Hopfield configuration
	'hopf_alpha'			: 0.999,
	'hopf_neuron_alpha'		: 0.9,
	'hopf_beta'				: 1.,
	'hopf_cycles'			: 6,
	'covariance_method'		: False,

	# Timings and rates
	'learning_rate'			: 3e-4,
	'drop_rate'				: 0.5,
	'discount_rate'			: 0.95,

	# Task specs
	'num_time_steps'		: 1000,
	'num_batches'			: 1000,

	# Maze task specs
	'rewards'				: [1.],
	'num_actions'			: 5,		# The number of different actions available to the agent
	'room_width'			: 6,
	'room_height'			: 4,
	'use_default_rew_locs'	: False,


	# Cost values
	'sparsity_cost'         : 1e-3, # was 1e-2
	'rec_cost'				: 1e-3,  # was 1e-2
	'weight_cost'           : 1e-2,  # was 1e-6
	'entropy_cost'          : 0.01,
	'val_cost'              : 0.01,

	# Training specs
	'batch_size'			: 64,

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

	fn = './savedir/14_tasks_base_medium_model_weights.pkl'
	results = pickle.load(open(fn, 'rb'))
	print('Weight keys ', results['weights'].keys())
	par['W0_init'] = results['weights']['W0']
	par['W1_init'] = results['weights']['W1']
	par['b0_init'] = results['weights']['b0']


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

	par['num_reward_types'] = len(par['rewards']) + 1

	par['n_hopf_stim'] = par['n_latent']*par['hopf_multiplier']
	par['n_hopf_act'] = 2*par['n_pol']*par['hopf_multiplier']

	# Specify one-hot vectors matching with each reward for maze task
	condition = True
	while condition:
		par['reward_vectors'] = np.random.choice([0,1], size=[len(par['rewards']), par['num_rew_tuned']])
		condition = (np.mean(np.std(par['reward_vectors'], axis=0)) == 0.) and len(par['rewards']) > 1

	c = 0.02

	par['W_enc_init'] = np.random.uniform(-c, c, size=[par['n_input'], par['n_latent']]).astype(np.float32)
	par['W_dec_init'] = np.random.uniform(-c, c, size=[par['n_latent'], par['n_input']]).astype(np.float32)
	par['b_enc_init'] = np.zeros([1, par['n_latent']], dtype=np.float32)

	N = (len(par['rewards']) + 1)*par['n_pol'] + par['n_latent']
	par['W0_init'] = np.random.uniform(-c, c, size=[N, par['n_hidden'][0]]).astype(np.float32)
	par['W1_init'] = np.random.uniform(-c, c, size=[par['n_hidden'][0], par['n_hidden'][1]]).astype(np.float32)
	par['b0_init'] = np.zeros([1, par['n_hidden'][0]], dtype=np.float32)
	par['b1_init'] = np.zeros([1, par['n_hidden'][1]], dtype=np.float32)

	par['W_pol_init'] = np.random.uniform(-c, c, size=[par['n_hidden'][1], par['n_pol']]).astype(np.float32)
	par['b_pol_init'] = np.zeros([1,par['n_pol']], dtype=np.float32)
	par['W_val_init'] = np.random.uniform(-c, c, size=[par['n_hidden'][1], 1]).astype(np.float32)
	par['b_val_init'] = np.zeros([1,1], dtype=np.float32)

	par['W_stim_write'] = np.zeros([par['hopf_multiplier'], par['n_latent'], par['n_hopf_stim']], dtype=np.float32)
	for i,j in product(range(par['hopf_multiplier']), range(par['n_latent'])):
		k = np.random.randint(par['hopf_multiplier'])
		par['W_stim_write'][i, j, k + j*par['hopf_multiplier']] = 1.

	N = 2*par['n_pol']
	par['W_act_write'] = np.zeros([par['hopf_multiplier'], N, par['n_hopf_act']], dtype=np.float32)
	for i,j in product(range(par['hopf_multiplier']), range(N)):
		k = np.random.randint(par['hopf_multiplier'])
		par['W_act_write'][i, j, i + j*par['hopf_multiplier']] = 1.

	par['W_stim_read'] = np.sum(par['W_stim_write'], axis = 0)


	par['H_stim_mask'] = np.ones((par['n_hopf_stim'], par['n_hopf_stim']), dtype = np.float32) \
		- np.eye((par['n_hopf_stim']), dtype = np.float32)
	par['H_old_new_mask'] = np.ones((par['n_hopf_stim'], par['n_hopf_stim']), dtype = np.float32)
	par['H_old_new_mask'] -= np.tril(par['H_old_new_mask'], -1)

	# load_encoder_weights()

update_dependencies()
print('--> Parameters successfully loaded.\n')
