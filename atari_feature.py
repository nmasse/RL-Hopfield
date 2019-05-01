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
import atari_capsule as ae
import striatum
import time

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

	def __init__(self, stim, reward, action, future_val, future_capsule, terminal_state, gate, step, lr_multiplier):

		# Gather placeholders
		self.stim           = stim
		self.reward         = reward
		self.action         = action
		self.future_val     = future_val
		self.future_capsule = future_capsule
		self.terminal_state = terminal_state
		self.gate           = gate
		self.step           = step
		self.lr_multiplier  = lr_multiplier

		# Run model
		self.run_model()

		# Optimize
		self.optimize()


	def run_model(self):

		# Set the number of features and properties
		n_features      = 16
		n_preproperties = 20
		n_properties    = 5

		# Run encoder to get latent vector
		self.caps = ae.encoder(self.stim, n_features, n_preproperties, n_properties, \
			var_dict=par['loaded_var_dict'], trainable=par['train_encoder'])

		# Process action such that it may be added to each set of feature properties
		reshaped_action = tf.reshape(self.action, [par['batch_size'],1,1,1,par['n_pol']])
		tiled_action    = tf.tile(reshaped_action, multiples=[1,*self.caps.shape.as_list()[1:-1],1])
			
		# Combine capsule properties and action
		source = tf.concat([self.caps, tiled_action], axis=-1)

		# Make variables for capsule prediction
		W_pred = tf.get_variable('W_pred', \
			shape=[n_features,n_properties+par['n_pol'],n_properties])
		b_pred = tf.get_variable('b_pred', shape=[1,1,1,n_features,n_properties])

		# Project from properties + action to strict properties prediction
		# [batch, x, y, feature, prop + action], [feature, prop + action, prop]
		#    --> [batch, x, y, feature, prop]
		self.pred = tf.einsum('bxyfi,fij->bxyfj', source, W_pred) + b_pred

		# Flatten capsule output to get latent state and project
		# through hidden layers
		self.latent = tf.reshape(self.caps, [par['batch_size'], -1])
		h0 = dense_layer(self.latent, 2048, 'h0')
		h1 = dense_layer(h0, 2048, 'h1')

		# Calculate policy and value outputs
		self.pol = dense_layer(h1, par['n_pol'], 'pol', activation=tf.identity)
		self.val = dense_layer(h1, par['n_val'], 'val', activation=tf.identity)

		# Normalize policy for entropy calculations
		self.pol = tf.nn.softmax(self.pol, axis = 1)


	def optimize(self):

		epsilon = 1e-6

		# Collect all variables in the model and list them out
		var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		self.var_dict = {var.op.name : var for var in var_list}
		print('Variables:')
		[print(var.op.name.ljust(20), ':', var.shape) for var in var_list]
		print()

		# Make optimizer
		opt = AdamOpt.AdamOpt(var_list, algorithm='rmsprop', learning_rate=par['learning_rate'])

		# Calculate RL quantities
		pred_val = self.reward + (par['discount_rate']**self.step)*self.future_val*(1. - self.terminal_state)
		advantage = pred_val - self.val

		# Calculate RL losses
		pol_loss     = -tf.reduce_mean(tf.stop_gradient(advantage)*self.action*tf.log(self.pol + epsilon))
		val_loss     =  tf.reduce_mean(tf.square(advantage))
		entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.pol*tf.log(self.pol + epsilon), axis = 1))

		# Calculate state prediction loss
		self.pred_loss =  tf.reduce_mean(tf.square(self.pred - self.future_capsule))

		loss = pol_loss + par['val_cost'] * val_loss - par['entropy_cost'] * entropy_loss \
			+ par['pred_cost'] * self.pred_loss

		# Make update operations for gradient applications
		self.update_grads = opt.compute_gradients_rmsprop(loss)

		# Make apply operations for gradient applications
		self.update_weights = opt.update_weights_rmsprop(lr_multiplier = self.lr_multiplier)




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
	x  = tf.placeholder(tf.float32, [par['batch_size'], 100, 84, 4], 'input')
	r  = tf.placeholder(tf.float32, [par['batch_size'], 1], 'reward')
	a  = tf.placeholder(tf.float32, [par['batch_size'], par['n_pol']], 'action')
	f  = tf.placeholder(tf.float32, [par['batch_size'], par['n_val']], 'future_val')
	fc = tf.placeholder(tf.float32, [par['batch_size'], 25, 21, 16, 5], 'future_capsule')
	g  = tf.placeholder(tf.float32, [par['batch_size'], par['n_latent']], 'gate')
	ts = tf.placeholder(tf.float32, [par['batch_size'], 1], 'terminal_state')
	s  = tf.placeholder(tf.float32, [], 'step')
	lr = tf.placeholder(tf.float32, [], 'learning_rate_mult')

	# Start TensorFlow session
	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		# Set up and initialize model on desired device
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, r, a, f, fc, ts, g, s, lr)
		sess.run(tf.global_variables_initializer())

		# Make scoreboard
		high_score        = np.zeros([par['batch_size'], 1])
		agent_score       = np.zeros([par['batch_size'], 1])
		final_agent_score = np.zeros([par['batch_size'], 1])

		# Make lists for recording performanc
		reward_list_full = []
		reward_list      = []
		obs_list         = []
		state_list       = []
		action_list      = []
		value_list       = []
		done_list        = []

		# Make initial states for model
		gate = np.random.choice([0. , 1/(1-par['drop_rate'])], size = [par['batch_size'], \
			par['n_latent']], p=[par['drop_rate'], 1 - par['drop_rate']])
		pred_loss_total = -1.

		# Start training loop
		print('Starting training.\n')
		for fr in range(par['num_frames']//par['k_skip']):

			lr_multiplier = np.maximum(0.001, 0.99995**fr) # 999995
			lr_multiplier = 1.

			# Run the model
			pol, val, caps = sess.run([model.pol, model.val, model.caps], feed_dict = {x:obs, g:gate})
			obs_list.append(obs)
			state_list.append(caps)

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

			if len(obs_list) >= par['n_step']+1:
				pred_loss_total = 0.
				for t0 in range(par['n_step']):
					reward = np.zeros((par['batch_size'], 1))
					done = np.zeros((par['batch_size'], 1))
					for t1 in range(t0, par['n_step']):
						done += done_list[t1]
						reward += reward_list[t1]*par['discount_rate']**(t1-t0)
						#print(t0, t1, len(obs_list))
					done = np.minimum(1., done)

					# train the model
					_, pred_loss = sess.run([model.update_grads, model.pred_loss], feed_dict = {x : obs_list[t0], \
						a: action_list[t0], r : reward, f: val, fc: state_list[t0+1], \
						ts: done, g:gate, s:t1-t0+1, lr: lr_multiplier})
					pred_loss_total += pred_loss

				obs_list    = obs_list[1:]
				state_list  = state_list[1:]
				action_list = action_list[1:]
				reward_list = reward_list[1:]
				done_list   = done_list[1:]

			if fr%par['n_step'] == 0 and fr >= par['n_step']+1:
				# sess.run(model.update_weights, feed_dict = {lr: lr_multiplier})
				pass

			if fr%par['gate_reset']==0 and fr>0:
				sess.run(model.update_weights, feed_dict = {lr: lr_multiplier})
				gate = np.random.choice([0. , 1/(1-par['drop_rate'])], size = [par['batch_size'], \
					par['n_latent']], p=[par['drop_rate'], 1 - par['drop_rate']])
				#gate = np.ones_like(gate)

			if len(reward_list_full) >= 1000:
				reward_list_full = reward_list_full[1:]
			if fr%1000==0:
				print('Frame {:>7} | Policy {} | Reward {:5.3f} | Overall HS: {:>4} | Current HS: {:>4} | Mean Final HS: {:7.2f}'.format(\
					fr, np.round(np.mean(pol,axis=0),2), np.mean(reward_list_full), int(high_score.max()), int(agent_score.max()), np.mean(final_agent_score)))
				print(' '*14 + '| Pred Loss {:5.3f}'.format(pred_loss_total))
				weights = sess.run(model.var_dict)
				fn = './savedir/weights_batch{}_drop5{}_reset{}_iter0.pkl'.format(par['batch_size'], \
					int(par['drop_rate']*10), par['gate_reset'])
				pickle.dump(weights, open(fn,'wb'))

			if fr%20000 == 0 and fr != 0:

				obs_list = []
				action_list = []
				reward_list = []
				done_list = []

				N = 10
				render_done       = np.zeros([par['batch_size'], N], dtype=np.float32)
				render_reward     = np.zeros([par['batch_size'], N], dtype=np.float32)
				render_best_score = np.zeros([par['batch_size'], N], dtype=np.float32)

				for k in range(N):
					render_fr_count = 1

					if k == 0:
						print('\nRendering video...')
						dirname = environment.start_render(fr)
					else:
						environment.start_render(fr, render=False)

					render_obs = environment.reset_environments()

					while np.any(render_done[:,k] == 0.):

						render_pol = sess.run(model.pol, feed_dict={x:render_obs, g:np.ones_like(gate)})
						render_action = np.array(np.stack([np.random.multinomial(1, render_pol[i,:]-1e-6) for i in range(par['batch_size'])]))

						for _ in range(par['k_skip']):
							render_fr_count += 1
							render_obs, render_reward_frame, _, render_done_frame = environment.agent_action(render_action)

							render_reward_frame = np.squeeze(render_reward_frame)
							render_done_frame = np.squeeze(render_done_frame)

							# Update record states
							render_reward[:,k] += render_reward_frame
							render_done[:,k]   += render_done_frame

							# Update high score
							render_best_score[:,k] = np.maximum(render_best_score[:,k], render_reward[:,k])

							# End environments as necessary
							render_reward[:,k] *= (1-render_done_frame)

						# Stop rendering if too many frames have elapsed
						if render_fr_count > 50000:
							break

					if k == 0:
						environment.stop_render()
					else:
						environment.stop_render(render=False)

				render_mean_score = np.mean(render_best_score)
				render_high_score = int(render_best_score.max())
				render_best_agent = np.where(np.squeeze(render_best_score[:,0]==render_best_score[:,0].max()))[0]

				line0 = 'Recorded at Training Frame {}'.format(fr)
				line1 = 'High Score: {}       [Agent(s) {}]'.format(render_high_score, render_best_agent)
				line2 = 'Mean Score: {}'.format(render_mean_score)
				line3 = 'Agent ID / Personal Best'
				lines = ['{:>8} / {}'.format(int(ag), int(bs)) for ag, bs \
					in enumerate(np.squeeze(render_best_score[:,0]))]

				text = [line0, line1, line2, line3] + lines
				text = '\n'.join(text)
				with open(dirname+'performance_record.txt', 'w') as tfile:
					tfile.write(text)

				print('Rendered {} frames with a high score of {} (mean {}).'.format(\
					render_fr_count, render_high_score, render_mean_score))
				print('Rendering complete.\n')


def print_key_params():

	key_params = ['savefn', 'striatum_th', 'trace_th', 'learning_rate', 'discount_rate',\
		'entropy_cost', 'val_cost', 'prop_top', 'drop_rate','batch_size','n_step','gate_reset']
	
	print('Key Parameters:\n'+'-'*60)
	for k in key_params:
		print(k.ljust(30), ':', par[k])
	print('-'*60 + '\n')


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
