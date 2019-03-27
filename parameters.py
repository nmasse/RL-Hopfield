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
	'n_hidden'				: 200,
	'n_val'					: 1,
	'n_latent'				: 40,
	'n_hopf_stim'			: 40*12,
	'n_hopf_act'			: 9*4*12,

	# Read/write configuration
	'A_alpha_init'			: 0.99995,
	'A_beta_init'			: 1.5,
	'inner_steps'			: 1,
	'batch_norm_inner'		: False,

	# Timings and rates
	'learning_rate'			: 3e-4,
	'drop_rate'				: 0.5,

	# Task specs
	'trial_length'			: 1200,
	'dead_time'				: 100,
	'dt'					: 100,
	'temporal_div'			: 1,
	'rewards'				: [1.],
	'num_actions'			: 5,		# The number of different actions available to the agent
	'room_width'			: 6,
	'room_height'			: 4,
	'use_default_rew_locs'	: False,

	# RL parameters
	'fix_break_penalty'     : -1.,
	'wrong_choice_penalty'  : -0.01, #-0.01,
	'correct_choice_reward' : 1.,
	'discount_rate'         : 0.9,

	# Tuning function data
	'num_motion_dirs'		: 8,
	'tuning_height'			: 4.0,

	# Cost values
	'sparsity_cost'         : 1e-3, # was 1e-2
	'rec_cost'				: 1e-3,  # was 1e-2
	'weight_cost'           : 1e-2,  # was 1e-6
	'entropy_cost'          : 0.01,
	'val_cost'              : 0.01,
	'stim_cost'				: 1e-1,

	# Training specs
	'batch_size'			: 64,
	'n_batches'				: 3000000,		# 1500 to train straight cortex

	'share_hippocampus'		: False,
	'covariance_method'		: False,

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

	# Reward map, for hippocampus reward one-hot conversion
	par['reward_map'] = {
		par['fix_break_penalty']		: 0,
		par['wrong_choice_penalty']		: 1,
		0.								: 2,
		par['correct_choice_reward']	: 3
	}

	par['num_reward_types'] = len(par['reward_map'].keys())
	par['reward_map_matrix'] = np.zeros([par['num_reward_types'],1]).astype(np.float32)
	for key, val in par['reward_map'].items():
		par['reward_map_matrix'][val,:] = key

	# Set input and output sizes
	par['n_input']  = par['num_nav_tuned'] + par['num_fix_tuned'] + par['num_rew_tuned'] + par['num_rule_tuned']
	par['n_output'] = par['num_actions']
	par['n_pol'] = par['num_motion_dirs'] + 1

	# Set trial step length
	par['num_time_steps'] = par['trial_length']//par['dt']

	# Specify one-hot vectors matching with each reward for maze task
	condition = True
	while condition:
		par['reward_vectors'] = np.random.choice([0,1], size=[len(par['rewards']), par['num_rew_tuned']])
		condition = (np.mean(np.std(par['reward_vectors'], axis=0)) == 0.) and len(par['rewards']) > 1

	# Set up standard LSTM weights and biases
	LSTM_weight = lambda size : np.random.uniform(-par['LSTM_init'], par['LSTM_init'], size=size).astype(np.float32)

	# option 1
	#for p in ['Wf', 'Wi', 'Wo', 'Wc']: par[p+'_init'] = LSTM_weight([par['n_input']+par['n_hidden'], par['n_hidden']])
	#par['W_write_init'] = LSTM_weight([par['n_input']+par['n_val']+par['n_pol'], par['n_latent']])
	# option 2
	#for p in ['Wf', 'Wi', 'Wo', 'Wc']: par[p+'_init'] = LSTM_weight([par['n_input'], par['n_hidden'][0]])
	#for j in range(par['n_modules']):
	#	for p in ['Wf', 'Wi', 'Wo', 'Wc']: par[p+str(j)+'_init'] = LSTM_weight([par['n_input'], par['n_hidden'][j]])
	#	for p in ['Uf', 'Ui', 'Uo', 'Uc']: par[p+str(j)+'_init'] = LSTM_weight([par['n_hidden'][0], par['n_hidden'][j]])
	#	for p in ['bf', 'bi', 'bo', 'bc']: par[p+str(j)+'_init'] = np.zeros([1, par['n_hidden'][j]], dtype=np.float32)
	par['W0_init'] = np.random.uniform(-0.0, 0.02, size=[par['n_input'], par['n_latent']]).astype(np.float32)
	par['W1_init'] = np.random.uniform(-0.0, 0.02, size=[par['n_latent'], par['n_input']]).astype(np.float32)
	par['b0_init'] = np.zeros([1, par['n_latent']], dtype=np.float32)

	N = par['n_latent']
	M = par['num_time_steps']//par['temporal_div']
	par['W_stim_write'] = np.zeros([M, N, par['n_hopf_stim']], dtype=np.float32)
	for i,j in product(range(M), range(N)):
		par['W_stim_write'][i, j, i + j*M] = 1.

	par['W_stim_read'] = np.sum(par['W_stim_write'], axis = 0)

	"""
	plt.imshow(par['W_stim_read'], aspect = 'auto')
	plt.show()
	plt.imshow(par['W_stim_write'][1,:,:], aspect = 'auto')
	plt.show()
	plt.imshow(par['W_stim_write'][2,:,:], aspect = 'auto')
	plt.show()
	"""

	N = par['num_reward_types']*par['n_pol']
	par['W_act_write'] = np.zeros([M, N, par['n_hopf_act']], dtype=np.float32)
	for i,j in product(range(M), range(N)):
		par['W_act_write'][i, j, i + j*M] = 1.



	# V0
	#n_input_ctl = par['n_pol']*par['num_reward_types']*par['num_time_steps']//par['temporal_div'] \
	# 	+ par['n_pol'] + par['n_val'] + par['n_input']
	n_input_ctl = par['n_pol']*par['num_reward_types'] \
		+ par['n_pol'] + par['n_val'] + par['n_input']

	n_input_ctl = par['n_pol']*par['num_reward_types'] + par['n_latent']
	#n_input_ctl = 33 + par['n_pol'] + par['n_val'] + par['n_pol']*par['n_val']
	# V1
	#n_input_ctl = par['n_module_out']*par['n_modules'] + par['n_pol'] + par['n_val'] + par['n_pol']*par['n_val']
	#n_input_ctl = par['n_input'] + par['n_module_out']*par['n_modules'] + par['n_pol']*par['n_val']

	for p in ['Wf', 'Wi', 'Wo', 'Wc']: par[p+'_init'] = LSTM_weight([n_input_ctl, par['n_hidden']])
	for p in ['Uf', 'Ui', 'Uo', 'Uc']: par[p+'_init'] = LSTM_weight([par['n_hidden'], par['n_hidden']])
	for p in ['bf', 'bi', 'bo', 'bc']: par[p+'_init'] = np.zeros([1, par['n_hidden']], dtype=np.float32)

	N0 = 200
	#N2 = 200
	par['W_ff0_init'] = LSTM_weight([n_input_ctl, N0])
	par['W_ff1_init'] = LSTM_weight([N0, par['n_hidden']])
	par['b_ff0_init'] = np.zeros([1, N0], dtype=np.float32)
	par['b_ff1_init'] = np.zeros([1, par['n_hidden']], dtype=np.float32)

	# LSTM posterior distribution weights
	#for p in ['Pf', 'Pi', 'Po', 'Pc']: par[p+'_init'] = LSTM_weight([par['n_tasks'], par['n_hidden']])

	# Cortex RL weights and biases
	par['W_pol_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_hidden'], par['n_pol']]).astype(np.float32)
	par['b_pol_init'] = np.zeros([1,par['n_pol']], dtype=np.float32)
	par['W_val_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_hidden'], 1]).astype(np.float32)
	par['b_val_init'] = np.zeros([1,1], dtype=np.float32)

	par['H_stim_mask'] = np.ones((par['n_hopf_stim'], par['n_hopf_stim']), dtype = np.float32) \
		- np.eye((par['n_hopf_stim']), dtype = np.float32)
	par['H_old_new_mask'] = np.ones((par['n_hopf_stim'], par['n_hopf_stim']), dtype = np.float32)
	par['H_old_new_mask'] -= np.tril(par['H_old_new_mask'], -1)
	#plt.imshow(par['H_old_new_mask'], aspect = 'auto')
	#plt.show()
	#par['H_old_new_mask'] = par['H_old_new_mask'][np.newaxis, ...]



	# load_encoder_weights()

update_dependencies()
print('--> Parameters successfully loaded.\n')
