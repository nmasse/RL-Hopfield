### Authors: Nicolas Y. Masse, Gregory D. Grant
import numpy as np
from parameters import par
import copy

# Actions that can be taken
#   Move up, down, left, right
#   Pick up reward


class Stimulus:


	def __init__(self):

		self.place_rewards()
		self.place_agents()

		self.rewards = par['rewards']


	def reset_rewards(self, trial_completion_vector=None):

		if trial_completion_vector is not None:
			trial_completion_vector = trial_completion_vector.astype(np.bool)
			for t in range(par['batch_size']):
				if trial_completion_vector[t]:
					self.place_rewards(agent_id=t)
		else:
			self.place_rewards()


	def reset_agents(self, trial_completion_vector=None):

		if trial_completion_vector is not None:
			trial_completion_vector = trial_completion_vector.astype(np.bool)
			for t in range(par['batch_size']):
				if trial_completion_vector[t]:
					self.place_agents(agent_id=t)
		else:
			self.place_agents()


	def make_reward_locations(self):

		inds = np.random.choice(par['room_width']*par['room_height'], size=len(par['rewards']), replace=False)

		rew_locs = []
		for i in range(len(par['rewards'])):
			rew_loc = [int(inds[i]//par['room_width']), int(inds[i]%par['room_width'])]
			rew_locs.append(tuple(rew_loc))

		return rew_locs


	def place_rewards(self, agent_id='all'):

		if agent_id is 'all':
			self.reward_locations = []
			for i in range(par['batch_size']):
				self.reward_locations.append(self.make_reward_locations())
		else:
			self.reward_locations[agent_id] = self.make_reward_locations()


	def place_agents(self, agent_id='all'):


		if agent_id is 'all':
			xs = np.random.choice(par['room_width'],size=par['batch_size'])
			ys = np.random.choice(par['room_height'],size=par['batch_size'])
			self.agent_loc = [[int(ys[i]), int(xs[i])] for i in range(par['batch_size'])]
		else:
			x = np.random.choice(par['room_width'])
			y = np.random.choice(par['room_height'])
			self.agent_loc[agent_id] = [int(y), int(x)]


	def identify_reward(self, location, agent_id):

		if tuple(location) in self.reward_locations[agent_id]:
			reward_index = self.reward_locations[agent_id].index(tuple(location))
			return self.rewards[reward_index], par['reward_vectors'][reward_index,:]
		else:
			return 0, None


	def make_inputs(self):

		# Inputs contain information for batch x (d1, d2, d3, d4, on_stim)
		inputs = np.zeros([par['batch_size'], par['n_input']])
		inputs[:,0] = [agent[0] for agent in self.agent_loc]
		inputs[:,1] = [agent[1] for agent in self.agent_loc]
		inputs[:,2] = [(par['room_height']-1) - agent[0] for agent in self.agent_loc]
		inputs[:,3] = [(par['room_width']-1) - agent[1] for agent in self.agent_loc]

		for i in range(par['batch_size']):
			_, vec = self.identify_reward(self.agent_loc[i], i)
			if vec is not None:
				inputs[i,par['num_nav_tuned']:par['num_nav_tuned']+par['num_rew_tuned']] = vec

		return np.float32(inputs)


	def agent_action(self, action, mask=np.ones(par['batch_size'])):
		""" Takes in a vector of actions of size [batch_size, n_pol] """

		action = np.argmax(action, axis=-1) # to [batch_size]
		reward = np.zeros(par['batch_size'], dtype=np.float32)

		for i, a in enumerate(action):

			# If the network has found a reward for this trial, cease movement
			if mask[i] == 0.:
				continue

			if a == 0 and self.agent_loc[i][1] != par['room_width']-1:
				# Input 0 = Move Up (visually right)
				self.agent_loc[i][1] += 1
			elif a == 1 and self.agent_loc[i][1] != 0:
				# Input 1 = Move Down (visually left)
				self.agent_loc[i][1] -= 1
			elif a == 2 and self.agent_loc[i][0] != par['room_height']-1:
				# Input 2 = Move Right (visually down)
				self.agent_loc[i][0] += 1
			elif a == 3 and self.agent_loc[i][0] != 0:
				# Input 3 = Move Left (visually up)
				self.agent_loc[i][0] -= 1
			elif a == 4:
				# Input 5 = Pick Reward
				rew, _ = self.identify_reward(self.agent_loc[i], i)
				if rew is not None:
					reward[i] = rew

		return reward[:, np.newaxis]


	def get_agent_locs(self):
		return np.array(self.agent_loc).astype(np.int32)


	def get_reward_locs(self):
		return np.array(self.reward_locations).astype(np.int32)


if __name__ == '__main__':

	### Diagnostics
	r = Stimulus()
	changes = np.random.choice([0,1],size=[par['batch_size']])
	changes[0] = 1
	changes[1] = 0

	import matplotlib.pyplot as plt
	for i in range(10):

		if i < 5:
			r.reset_agents(changes)
		else:
			r.reset_rewards(changes)

		stim_in = r.make_inputs()
		print(stim_in[:2,:])

		agent_locs = r.get_agent_locs()
		reward_locs = r.get_reward_locs()

		fig, ax = plt.subplots(1,2)
		for t in range(2):
			demo_room = np.zeros([par['room_width'],par['room_height']])
			demo_room[agent_locs[t,1],agent_locs[t,0]] = 1
			demo_room[reward_locs[t][0][1],reward_locs[t][0][0]] = -1

			ax[t].imshow(demo_room)
			ax[t].set_title('Iteration: {} | Room: {}'.format(i, t))
		plt.show()
