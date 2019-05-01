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
#import striatum
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
	W = tf.get_variable('W_'+name, shape=[n_in, n_out], initializer = tf.variance_scaling_initializer(scale=2.))
	b = tf.get_variable('b_'+name, shape=[1, n_out])

	y = x @ W + b

	return activation(y)


class Model:

	def __init__(self, stim, reward, action, future_val, terminal_state, gate, step, lr_multiplier):

		# Gather placeholders
		self.stim = stim
		self.reward = reward
		self.action = action
		self.future_val = future_val
		self.terminal_state = terminal_state
		self.gate = gate
		self.step = step
		self.lr_multiplier = lr_multiplier

		# Run encoder
		flat, conv_shapes = ae.encoder(self.stim, par['n_latent'], \
			var_dict=par['loaded_var_dict'], trainable=par['train_encoder'])
		z = dense_layer(flat, par['n_latent'], 'out')
		#top_k, _ = tf.nn.top_k(z, k = 50)
		#top_cond = z >= top_k[:,-50:-49]
		#z = tf.where(top_cond, z, tf.zeros(z.shape))
		self.latent = self.gate * z
		self.val = dense_layer(self.latent, par['n_val'], 'val', activation = tf.identity)

		#self.striatum_interaction()
		#self.latent = tf.concat([self.latent, self.striatum_out], axis = 1)

		self.pol = dense_layer(self.latent, par['n_pol'], 'pol', activation = tf.identity)
		self.pol = tf.nn.softmax(self.pol, axis = 1)


		# Run optimizer
		self.optimize()



	def striatum_interaction(self):

		self.striatum = striatum.Network()

		pred_val = self.reward + (par['discount_rate']**self.step)*self.future_val*(1. - self.terminal_state)
		delta_value = pred_val - self.val
		self.update_striatum = self.striatum.write_striatum(self.latent, self.action, delta_value)

		striatum_out = self.striatum.read_striatum(self.latent)
		self.striatum_out = dense_layer(striatum_out, 100, 'striatum_out')
		#self.pred_action = dense_layer(self.striatum_out, par['n_pol'], 'striatum_pred', activation = tf.identity)
		#self.pred_action = tf.nn.softmax(self.pred_action , axis = 1)

		self.U = self.striatum.U
		self.W = self.striatum.W


	def optimize(self):

		epsilon = 1e-6

		# Collect all variables in the model and list them out
		var_list_all = tf.trainable_variables()
		var_list = [var for var in var_list_all if not 'striatum' in var.op.name]
		var_list = var_list_all

		var_list_striatum = [var for var in var_list_all if 'striatum' in var.op.name]
		self.var_dict = {var.op.name : var for var in var_list}
		print('Variables:')
		[print(var.op.name.ljust(20), ':', var.shape) for var in var_list]
		print()
		print('Striatum Variables:')
		[print(var.op.name.ljust(20), ':', var.shape) for var in var_list_striatum]
		print()

		# Make optimizer
		opt = AdamOpt.AdamOpt(var_list, algorithm = 'rmsprop', learning_rate = par['learning_rate'])
		opt_striatum = AdamOpt.AdamOpt(var_list_striatum, algorithm = 'rmsprop', learning_rate = par['learning_rate'])

		pred_val = self.reward + (par['discount_rate']**self.step)*self.future_val*(1. - self.terminal_state)
		advantage = pred_val - self.val

		pol_loss = -tf.reduce_mean(tf.stop_gradient(advantage)*self.action*tf.log(self.pol + epsilon))

		val_loss = tf.reduce_mean(tf.square(advantage))

		entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.pol*tf.log(self.pol + epsilon), axis = 1))
		#entropy_loss = -tf.reduce_mean(tf.reduce_mean(self.pol*tf.log(self.pol + epsilon), axis = 1))

		loss = pol_loss + par['val_cost'] * val_loss - par['entropy_cost'] * entropy_loss

		self.update_grads = opt.compute_gradients_rmsprop(loss)

		self.update_weights = opt.update_weights_rmsprop(lr_multiplier = self.lr_multiplier)
		"""
		self.loss_striatum = -tf.reduce_mean(self.action*tf.log(self.pred_action + epsilon))

		self.update_striatum_grads = opt_striatum.compute_gradients_rmsprop(self.loss_striatum)

		self.update_striatum_weights = opt_striatum.update_weights_rmsprop(lr_multiplier = self.lr_multiplier)
		"""



def main(gpu_id=None):

	print_key_params()

	# Select GPU
	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	# Reduce memory consumption for GPU 0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)# \
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
	lr = tf.placeholder(tf.float32, [], 'step')

	# Start TensorFlow session
	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		# Set up and initialize model on desired device
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, r, a, f, ts, g, s, lr)
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
		loss_striatum = 0.
		U = 0.
		W = 0.
		gate = np.random.choice([0. , 1/(1-par['drop_rate'])], size = [par['batch_size'], \
			par['n_latent']], p=[par['drop_rate'], 1 - par['drop_rate']])

		for fr in range(par['num_frames']//par['k_skip']):

			lr_multiplier = np.maximum(0.001, 0.99995**fr) # 999995
			lr_multiplier = 1.

			# Run the model
			pol, val = sess.run([model.pol, model.val], feed_dict = {x : obs, g:gate})
			obs_list.append(obs)

			# choose action, determine reward
			action = np.array(np.stack([np.random.multinomial(1, pol[i,:]-1e-6) for i in range(par['batch_size'])]))
			action_list.append(action)

			# Generate next four frames
			reward = np.zeros((par['batch_size'], 1))
			done   = np.zeros((par['batch_size'], 1))
			for _ in range(par['action_repeat']):
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
				for t0 in range(par['n_step']):
					reward = np.zeros((par['batch_size'], 1))
					done = np.zeros((par['batch_size'], 1))
					for t1 in range(t0, par['n_step']):
						done += done_list[t1]
						reward += reward_list[t1]*par['discount_rate']**(t1-t0)
						#print(t0, t1, len(obs_list))
					done = np.minimum(1., done)

					# train the model
					sess.run(model.update_grads, feed_dict = {x : obs_list[t0], \
						a: action_list[t0], r : reward, f: val, ts: done, g:gate, \
						s:t1-t0+1, lr: lr_multiplier})
					"""
					if t0 == par['n_step']-1:
						#_, _, loss_striatum = sess.run([model.update_striatum_grads, \
						#	model.update_striatum_weights, model.loss_striatum], feed_dict = {x : obs_list[t0], \
						#	a: action_list[t0], r : reward, f: val, ts: done, g:gate, \
						#	s:1, lr: lr_multiplier})
						sess.run(model.update_striatum, feed_dict = {x : obs_list[t0], \
							a: action_list[t0], r : reward, f: val, ts: done, g:gate, \
							s:1, lr: lr_multiplier})
						U, W = sess.run([model.U, model.W])
					"""


				obs_list = obs_list[1:]
				action_list = action_list[1:]
				reward_list = reward_list[1:]
				done_list = done_list[1:]

			if fr%par['n_step'] == 0 and fr >= par['n_step']+1:
				sess.run(model.update_weights, feed_dict = {lr: lr_multiplier})

			if fr%par['gate_reset']==0 and fr>0:
				#sess.run(model.update_weights, feed_dict = {lr: lr_multiplier})
				gate = np.random.choice([0. , 1/(1-par['drop_rate'])], size = [par['batch_size'], \
					par['n_latent']], p=[par['drop_rate'], 1 - par['drop_rate']])
				#gate = np.ones_like(gate)



			if len(reward_list_full) >= 1000:
				reward_list_full = reward_list_full[1:]
			if fr%1000==0:
				print('Frame {:>7} | Policy {} | Str. Loss {:5.3f}| Reward {:5.3f}  | Mean Final HS: {:7.2f}'.format(\
					fr, np.round(np.mean(pol,axis=0),2), loss_striatum, np.mean(reward_list_full), int(high_score.max()), int(agent_score.max()), np.mean(final_agent_score)))
				weights = sess.run(model.var_dict)
				fn = './savedir/{}_weights_batch{}_drop{}_reset{}_iter0.pkl'.format(par['task_name'], \
					par['batch_size'], int(par['drop_rate']*10), par['gate_reset'])
				pickle.dump(weights, open(fn,'wb'))
				#print('U', np.mean(np.abs(U)), ' W', np.mean(np.abs(W)))

			if fr%20000 == 0 and fr != 0:

				obs_list = []
				action_list = []
				reward_list = []
				done_list = []

				N = 2
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

						for _ in range(par['action_repeat']):
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

	key_params = ['gym_env', 'savefn', 'learning_rate', 'discount_rate',\
		'entropy_cost', 'val_cost', 'drop_rate','batch_size','n_step',\
		'gate_reset', 'batch_size','k_skip','action_repeat']
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
