import numpy as np
from scipy.misc import imresize
from collections import deque
import gym
from parameters import par


class Stimulus:

	def __init__(self):

		self.envs = [gym.make(par['gym_env']) for _ in range(par['batch_size'])]

	def preprocess(self, frame_list):

		# Downsample frames to 100 x 84
		frames = [imresize(f, (100, 84, 3)) for f in frame_list]

		# Combine batch of frames into an array
		frames = np.stack(frames, axis=0).astype(np.float32)

		# Convert values to a range of 0 to 1
		frames = frames/255

		# Convert to grayscale (luminosity method)
		frames = 0.21*frames[...,0] + 0.72*frames[...,1] + 0.07*frames[...,2]

		return frames


	def update_framebuffer(self, update, initialize=False):

		if initialize:
			self.framebuffer = deque([update]*4)
		else:
			self.framebuffer.popleft()
			self.framebuffer.append(update)

		return np.stack(self.framebuffer, axis=-1)


	def reset_environments(self):

		# Reset all environments
		obs = [e.reset() for e in self.envs]

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

		# Iterate over trials per batch
		for i in range(par['batch_size']):

			# Apply the desired action
			o, r, done, _ = self.envs[i].step(action[i])

			# Reset the environment the episode is complete
			if done:
				o = self.envs[i].reset()

			# Collect observation and reward
			obs.append(o)
			reward.append(r)

		# Preprocess batch of frames
		obs = self.preprocess(obs)

		# Update and obtain framebuffer
		obs = self.update_framebuffer(obs)

		# Clarify reward structure
		r = np.sign(np.array(r).astype(np.float32))

		# Return observation and reward as float32 arrays
		return obs, r



if __name__ == '__main__':
	import time
	import matplotlib.pyplot as plt

	s = Stimulus()
	obs = s.reset_environments()
	print('Stimulus loaded and reset.')

	t0 = time.time()
	for i in range(100):
		act = np.random.rand(par['batch_size'], 6)
		# s.envs[0].render()
		o, r = s.agent_action(act)
	print(time.time() - t0)

	for i in range(10):
		plt.imshow(np.mean(o[i,...], axis=-1), aspect='auto', cmap='gray')
		plt.show()