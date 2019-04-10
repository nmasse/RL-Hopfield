import tensorflow as tf
import numpy as np
from parameters import par
import stimulus


class HopfieldNetwork:

	def __init__(self):

		N = par['n_latent'] + par['n_pol']
		self.H  = tf.Variable(tf.zeros([N, par['n_dend'], N]), False)
		self.Hr = tf.Variable(tf.zeros([N, par['n_dend'], par['n_unique_vals']]), False)


	def _read_threshold(self, z, H):

		# Determine whether activity is above the threshold
		dend_act = tf.einsum('j,jkm->km', z, tf.minimum(1., H))
		dend_out = dend_act - par['dend_th']

		# Check dendrite activation against the reference
		above_th = tf.cast(dend_out > 0., tf.float32)

		# Calculate neuron response
		neur_out = tf.minimum(1., tf.reduce_sum(above_th, axis=0))

		return neur_out


	def _write_threshold(self, z, H):

		# Determine whether activity is above the threshold
		dend_act = tf.einsum('j,jkm->km', z, tf.minimum(1., H))
		dend_out = dend_act - par['dend_th']

		# Check dendrite activation against reference activation
		above_th = tf.reduce_sum(tf.cast(dend_out > 0.5, tf.float32), axis=0) > 0.
		above_th = tf.cast(above_th, tf.int64)

		return above_th, dend_act


	def _update_H(self, ind, H, z):

		update_ops = []
		for i in range(ind.shape.as_list()[0]):
			el = H[:,ind[i],i]
			op = el.assign(el + z[:,i])
			update_ops.append(op)

		return update_ops


	def read_hopfield(self, latent):

		reward_vector = tf.constant([-1.,0.,1.])    # Shape = [1,3]
		n_steps = 15
		gamma   = 0.9
		total_rewards = []

		for i in range(par['n_pol']):
			action         = tf.constant(par['action_template'][i][0,:])
			next_lat       = latent
			rewards_policy = []

			for j in range(n_steps):

				z = tf.concat([next_lat, par['act_mult']*action], axis=0)

				# Estimate next state and action
				neur_out = self._read_threshold(z, self.H)

				next_pol = neur_out[tf.newaxis,-par['n_pol']:]
				next_lat = neur_out[:-par['n_pol']]

				action   = tf.one_hot(tf.random.categorical(next_pol, 1), par['n_pol'])
				action   = tf.reshape(action, [-1])

				# Estimate reward
				neur_out = self._read_threshold(z, self.Hr)

				rew_pol  = tf.reduce_sum(reward_vector*neur_out, axis=0)*(gamma**j)
				rewards_policy.append(rew_pol)

			rewards_policy = tf.stack(rewards_policy, axis=0)
			total_rewards.append(tf.reduce_sum(rewards_policy, axis=0))

		total_rewards = tf.stack(total_rewards, axis=0)
		return total_rewards


	def write_hopfield(self, latent, action, prev_latent, prev_action, prev_reward):

		# Collect latent states and actions for previous and current steps
		z0 = tf.concat([prev_latent, prev_action], axis=0)
		z1 = tf.concat([latent, action], axis=0)
		zz = tf.einsum('j,k->jk', z0, z1)
		zr = tf.einsum('j,k->jk', z0, prev_reward)

		# Restate previous step with amplified action
		z0 = tf.concat([prev_latent, par['act_mult']*prev_action], axis=0)

		# Check whether activity meets threshold for H
		above_th, dend_act = self._write_threshold(z0, self.H)

		# Find indices for maximum and minimum dendrite activity
		ind_max = tf.argmax(dend_act, axis=0)
		ind_min = tf.argmin(dend_act, axis=0)

		ind = ind_max*above_th + ind_min*(1-above_th)

		# Update H
		H_updates = self._update_H(ind, self.H, zz)

		# Check whether activity meets threshold for Hr
		above_th, dend_act = self._write_threshold(z0, self.Hr)

		# Find indices for maximum and minimum dendrite activity
		ind_max = tf.argmax(dend_act, axis=0)
		ind_min = tf.argmin(dend_act, axis=0)

		ind = ind_max*above_th + ind_min*(1-above_th)

		# Update Hr
		Hr_updates = self._update_H(ind, self.Hr, zr)

		# Collect update ops
		update_ops = H_updates + Hr_updates

		return update_ops


class EventLists:

	def __init__(self):
		
		self.latent_list = []
		self.action_list = []
		self.reward_list = []
		self.max_size    = 20
		self.write_data  = False


	def write_hopfield(self, network):

		update_ops = []

		for i in range(self.max_size - 1):
			ops = network.write_hopfield(self.latent_list[i+1], self.action_list[i+1], \
				self.latent_list[i], self.action_list[i], self.reward_list[i])
			update_ops += ops

		return update_ops


	def add_event(self, network, latent, action, reward, write_data):

		if len(self.latent_list) >= self.max_size:
			self.latent_list = self.latent_list[1:]
			self.action_list = self.action_list[1:]
			self.reward_list = self.reward_list[1:]

		self.latent_list.append(latent)
		self.action_list.append(action)
		self.reward_list.append(reward)

		if self.write_data:
			update_ops = self.write_hopfield(network)
			self.write_data = False
		else:
			update_ops = [tf.no_op]

		self.write_data = True if write_data else False

		return update_ops