import pickle
import numpy as np
from itertools import product
import gym

print('\n--> Loading parameters...')

##############################
### Independent parameters ###
##############################

par = {
	# Setup parameters
	'savedir'				: './savedir/',
	'plotdir'				: './plotdir/',
	'RL_method'				: 'policy',

	# Network shape
	'n_hidden'				: [200, 200],
	'n_encoding'			: [64, 128],
	'n_pol'					: 4,
	'n_val'					: 1,
	'n_latent'				: 1024,
	'n_dend'				: 25,
	'act_mult'				: 4,
	'dend_th'				: 8,

	'prop_top'				: 0.05,
	'boost_level'			: 200.,
	'boost_alpha'			: 5e-3,

	# Hopfield configuration
	'hopf_alpha'			: 0.95,
	'hopf_neuron_alpha'		: 0.9,
	'hopf_beta'				: 1.,
	'hopf_cycles'			: 6,
	'covariance_method'		: True,

	# Timings and rates
	'learning_rate'			: 5e-4,
	'drop_rate'				: 0.5,
	'discount_rate'			: 0.9,

	# Task specs
	'num_time_steps'		: 1,
	'num_batches'			: 10000000,

	# Task specs
	'gym_env'				: 'SpaceInvaders-v0',
	'task'					: 'atari',
	'frames_per_iter'		: 1000,
	'k_skip'				: 4,
	'rewards'				: [-1.,1.],
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
	'batch_size'			: 64,
	'train_encoder'			: False,
	'load_weights'			: True,

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

	fn = par['savedir'] + '200boost_2x2filter_test' + '.pkl'
	var_dict = pickle.load(open(fn, 'rb'))['weights']
	return var_dict


def update_dependencies():
	""" Updates all parameter dependencies """


	par['savefn'] = '{}boost_2x2filter_test'.format(int(par['boost_level']))

	par['num_k'] = int(par['n_latent'] * par['prop_top'])

	par['action_template'] = []
	for i in range(par['n_pol']):
		p = np.zeros((par['batch_size'], par['n_pol']), dtype = np.float32)
		p[:, i] = 1.
		par['action_template'].append(p)

	if par['load_weights']:
		par['loaded_var_dict'] = load_encoder_weights()
	else:
		par['loaded_var_dict'] = None



update_dependencies()
print('--> Parameters successfully loaded.\n')
