import numpy as np
from scipy.misc import imresize
from collections import deque
import gym
from atari_parameters import par
import time


class Stimulus:

	def __init__(self):

		self.envs        = [gym.make(par['gym_env']) for _ in range(par['batch_size'])]
		self.render_envs = [gym.make(par['gym_env']) for _ in range(par['batch_size'])]

		print('Game:', par['gym_env'])
		print('Game action space:', self.envs[0].action_space)
		print('Actions:', self.envs[0].unwrapped.get_action_meanings(), '\n')

		self.cached_nn = False

	def start_render(self, frame, render=True):
		""" Push the standard environments into a backup, replace with
			environments that can be rendered to file """

		# Designate a directory name for this set of saves
		d = './viddir/{}_fr{}_t{}/'.format(par['savefn'], frame, int(time.time()))

		# Shunt training environments to the back burner
		self.envs_backup  = self.envs
		self.frmbf_backup = self.framebuffer

		# Make a set of evaluation environments environments
		if render:
			self.envs = [gym.wrappers.Monitor(env, d+'agent{}/'.format(i), force=True) \
				for i, env in enumerate(self.render_envs)]
		else:
			self.envs = self.render_envs

		# Return directory name
		return d

	def stop_render(self, render=True):
		""" Close the rendering environments and
			reinvoke the training environments """

		# Close the rendering environments
		if render:
			[env.close() for env in self.envs]

		# Bring the training environments back into focus
		self.envs = self.envs_backup
		self.framebuffer = self.frmbf_backup


	def nn_interpolate(self, x, new_size):

		# Assumes x is of shape [batch x pix_x x pix_y x RGB]
		# Assumes new_size is list/tuple [new_size_x, new_size_y]

		# Checks whether the interpolation algorithm has already been run
		# If so, skips straight to the downsampling step
		# If not, calculates the indices to do nearest-neighbor downsampling
		if not self.cached_nn:

			# Get the size of the source image
			old_size = x.shape[1:3]

			# Set up index collecting
			idx = []

			# Iterate over the dimensions of the image
			for d in range(len(old_size)):

				# Find the ratio between the old and new sizes for this dimension
				r = new_size[d]/old_size[d]

				# Determine the nearest-neighbor indices for each of the new pixels
				inds = np.ceil(np.arange(1, 1+new_size[d])/r - 1).astype(int)

				# Record this set of indices
				idx.append(inds)

			# Aggregate the indices into a grid for easy indexing
			self.nn_ind_set = np.meshgrid(*idx, sparse=False, indexing='ij')

			# Declare the interpolation indexing as complete and raise the
			# flag to retrieve the cached result next time
			self.cached_nn = True

		# Index into the source image to generate the interpolated image
		return x[:,self.nn_ind_set[0],self.nn_ind_set[1],:]


	def preprocess(self, frame_list):

		# Combine batch of frames into an array
		frames = np.stack(frame_list, axis=0)

		# Downsample frames to 100 x 84
		frames = self.nn_interpolate(frames, [100, 84])

		# Simultaneously convert to grayscale (luminosity method)
		# and convert integer color values to a range of 0 to 1
		frames = (0.21/255)*frames[...,0] \
			+ (0.72/255)*frames[...,1] \
			+ (0.07/255)*frames[...,2]

		# Return preprocessed frames, converted to float32
		return frames.astype(np.float32)


	def update_framebuffer(self, update, initialize=False):

		if initialize:
			self.framebuffer = deque([update]*par['k_skip'])
		else:
			self.framebuffer.popleft()
			self.framebuffer.append(update)

		return np.stack(self.framebuffer, axis=-1)


	def reset_environments(self):

		# Reset all environments
		obs = [e.reset() for e in self.envs]

		self.obs_shape = obs[0].shape

		# Preprocess original observations
		obs = self.preprocess(obs)

		# Initialize framebuffer with four sets of this frame
		obs = self.update_framebuffer(obs, initialize=True)

		return obs


	def agent_action(self, action):
		""" Takes in a vector of actions of size [batch_size, n_pol] """

		# Convert to indices
		action = np.argmax(action, axis=-1) # to [batch_size]

		obs    = [] # Aggregate observations
		reward = [] # Aggregate rewards
		done   = [] # Aggregate completion states

		# Iterate over trials per batch
		for i in range(par['batch_size']):

			r = 0.
			d = 0
			o = np.zeros(self.obs_shape)
			for k in range(par['frame_skip']):
				if d == 0:
					o0, r0, d0, _ = self.envs[i].step(action[i])
				r += r0
				d += d0

				if k >= par['frame_skip']-2:
					o += o0

			# Apply the desired action
			#o, r, d, _ = self.envs[i].step(action[i])

			# Reset the environment the episode is complete
			if d > 0:
				o = self.envs[i].reset()

			# Collect observation and reward
			obs.append(o/2)
			reward.append(r)
			done.append(np.minimum(1, d))

		# Preprocess batch of frames
		obs = self.preprocess(obs)

		# Update and obtain framebuffer
		obs = self.update_framebuffer(obs)

		# Clarify reward structure
		reward = np.array(reward).astype(np.float32)
		reward_sign = np.sign(reward)

		# Clarify done vector structure
		done = np.array(done)

		# Reshape rewards and termination vector
		reward_sign = reward_sign.reshape(-1,1)
		reward = reward.reshape(-1,1)
		done = done.reshape(-1,1)

		# Return observation and reward as float32 arrays
		return obs, reward, reward_sign, done



if __name__ == '__main__':
	import time
	import matplotlib.pyplot as plt

	s = Stimulus()
	obs = s.reset_environments()
	print('Stimulus loaded and reset.')

	t0 = time.time()
	# rew_hist = []
	# high_score = np.zeros([par['batch_size'], 1])
	for i in range(1000):
		act = np.random.rand(par['batch_size'], 6)
		# s.envs[0].render()
		o, r, rs, d = s.agent_action(act)

		# high_score += r
		# high_score *= (1-d)

		# print(i, np.squeeze(high_score).astype(np.int32))
		# rew_hist.append(r)
		# time.sleep(0.1)

	print('Elapsed:', time.time() - t0)
	# print(np.mean(rew_hist))

	# for i in range(10):
	# 	plt.imshow(np.mean(o[i,...], axis=-1), aspect='auto', cmap='gray')
	# 	plt.show()
