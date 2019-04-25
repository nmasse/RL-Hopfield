import numpy as np
from scipy.misc import imresize
from collections import deque
import gym
from atari_parameters import par


class Stimulus:

	def __init__(self):

		self.envs = [gym.make(par['gym_env']) for _ in range(par['batch_size'])]
		print('Game:', par['gym_env'])
		print('Game action space:', self.envs[0].action_space)
		print('Actions:', self.envs[0].unwrapped.get_action_meanings(), '\n')


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
			self.framebuffer = deque([update]*par['k_skip'])
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
		done   = [] # Aggregate completion states

		# Iterate over trials per batch
		for i in range(par['batch_size']):

			# Apply the desired action
			o, r, d, _ = self.envs[i].step(action[i])

			# Reset the environment the episode is complete
			if d:
				o = self.envs[i].reset()

			# Collect observation and reward
			obs.append(o)
			reward.append(r)
			done.append(d)

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
	rew_hist = []
	high_score = np.zeros([par['batch_size'], 1])
	for i in range(1000):
		act = np.random.rand(par['batch_size'], 6)
		s.envs[0].render()
		o, r, rs, d = s.agent_action(act)

		high_score += r
		high_score *= (1-d)

		print(i, np.squeeze(high_score).astype(np.int32))
		rew_hist.append(r)
		time.sleep(0.1)

	print(time.time() - t0)
	print(np.mean(rew_hist))

	# for i in range(10):
	# 	plt.imshow(np.mean(o[i,...], axis=-1), aspect='auto', cmap='gray')
	# 	plt.show()
