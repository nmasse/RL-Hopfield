### Authors: Nicolas Y. Masse, Gregory D. Grant
import tensorflow as tf
from parameters import par


def read(latent_inp, H_sas, H_sar):

	n_steps = 6
	gamma = 0.9

	x = latent_inp/(1e-9 + tf.sqrt(tf.reduce_sum(latent_inp**2, \
		axis=1, keepdims=True)))

	action_reward     = tf.einsum('ijkm,ij->ikm', H_sar, x)
	action_reward_rsh = tf.reshape(action_reward, [par['batch_size'], -1])

	total_rewards = []
	for i in range(par['n_pol']):
		rewards_policy = []

		for j in range(n_steps):
			action_stim   = tf.einsum('ijkm,ij->ikm', H_sas, latent_inp)
			action_reward = tf.einsum('ijkm,ij->ikm', H_sar, latent_inp)

			if j == 0:
				action = tf.constant(par['action_template'][i])
			else:
				action_dist  = tf.reduce_sum(action_stim, axis=2)
				action_index = tf.multinomial(tf.log(1e-5 + action_dist), 1)
				action       = tf.one_hot(tf.squeeze(action_index), par['n_pol'])

			next_stim = tf.einsum('ijk,ij->ik', action_stim, action)
			reward    = tf.einsum('ijk,ij->ik', action_reward, action)
			rewards_policy.append(reward * (gamma**j))

		rewards_policy = tf.stack(rewards_policy, axis=0)
		total_rewards.append(tf.reduce_sum(rewards_policy, axis=0))

	total_rewards = tf.stack(total_rewards, axis=2)
	total_rewards = tf.reshape(total_rewards, [par['batch_size'], -1])

	return action_reward_rsh, total_rewards


def write(hop_latent, prev_latent, H_sas, H_sar):

	norm_lambda  = lambda x : 1e-9 + tf.sqrt(tf.reduce_sum(x**2, axis=1, keepdims=True))

	latent       = hop_latent / norm_lambda(hop_latent)
	prev_latent  = prev_latent / norm_lambda(prev_latent)

	state_action = tf.einsum('ij,ik->ijk', prev_latent, prev_action)
	H_sas_grad   = tf.einsum('ijk,im->ijkm', state_action, latent)
	H_sar_grad   = tf.einsum('ijk,im->ijkm', state_action, prev_reward)

	alpha = 0.9999
	print('Hopfield writing currently broken.  Replace with latest code!')
	reset_ops  = [\
		tf.assign(H_sas, 0.*H_sas), \
		tf.assign(H_sar, 0.*H_sar)  ]
	update_ops = [\
		tf.assign(H_sas, alpha*H_sas + H_sas_grad), \
		tf.assign(H_sar, alpha*H_sar + H_sar_grad)  ]

	update_hopfield = tf.group(*update_ops)
	reset_hopfield = tf.group(*reset_ops)

	return update_hopfield, reset_hopfield