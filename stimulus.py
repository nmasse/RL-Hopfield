### Authors: Nicolas Y. Masse, Gregory D. Grant
import numpy as np
from parameters import par
import copy

# Actions that can be taken
#   Move up, down, left, right
#   Pick up reward

print('Using \'Room Stim\' stimulus file.')

class RoomStimulus:

	def __init__(self):

		self.initialize_rooms()
		self.place_agents()

		self.rewards = par['rewards']
		

	def initialize_rooms(self):

		# Two sets of reward locations:  Random and default
		rand_locs = np.random.choice(par['room_width']*par['room_height'], size=len(par['rewards']), replace=False)
		default_locs = [[1,1], [par['room_height']-2,par['room_width']-2], [1,par['room_width']-2], [par['room_height']-2],1]

		# Assign one stimulus location per reward
		self.stim_loc = []
		for i in range(len(par['rewards'])):

			if par['use_default_rew_locs']:
				if i >= len(default_locs):
					raise Exception('Implement more default reward locations!')
				rew_loc = default_locs[i]
			else:
				rew_loc = [int(rand_locs[i]//par['room_width']), int(rand_locs[i]%par['room_width'])]

			self.stim_loc.append(rew_loc)

		# One locations are assigned, place rewards at those locations
		self.place_rewards()


	def place_rewards(self):

		self.reward_locations = []
		for i in range(par['batch_size']):
			trial_set = {}
			for r, loc in enumerate([self.stim_loc[ind] for ind in np.random.permutation(len(par['rewards']))]):
				trial_set[tuple(loc)] = {'rew':par['rewards'][r], 'vec':par['reward_vectors'][r]}
			self.reward_locations.append(trial_set)


	def place_agents(self):

		xs = np.random.choice(par['room_width'],size=par['batch_size'])
		ys = np.random.choice(par['room_height'],size=par['batch_size'])
		self.agent_loc = [[int(ys[i]), int(xs[i])] for i in range(par['batch_size'])]

		self.loc_history = [self.agent_loc]


	def identify_reward(self, location, i):

		if tuple(location) in self.reward_locations[i].keys():
			data = self.reward_locations[i][tuple(location)]
			return data['rew'], data['vec']
		else:
			return None, None


	def make_inputs(self):

		# Inputs contain information for batch x (d1, d2, d3, d4, on_stim)
		inputs = np.zeros([par['batch_size'], par['n_input']])
		inputs[:,0] = [agent[0] for agent in self.agent_loc]
		inputs[:,1] = [agent[1] for agent in self.agent_loc]
		inputs[:,2] = [par['room_height'] - agent[0] for agent in self.agent_loc]
		inputs[:,3] = [par['room_width'] - agent[1] for agent in self.agent_loc]

		for i in range(par['batch_size']):
			_, vec = self.identify_reward(self.agent_loc[i], i)
			if vec is not None:
				inputs[i,par['num_nav_tuned']:par['num_nav_tuned']+par['num_rew_tuned']] = vec

		return np.float32(inputs)


	def agent_action(self, action, mask):
		""" Takes in a vector of actions of size [batch_size, n_output] """

		action = np.argmax(action, axis=-1) # to [batch_size]
		reward = np.zeros(par['batch_size'])

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

		self.loc_history.append(copy.deepcopy(self.agent_loc))

		return np.float32(reward)


	def get_agent_locs(self):
		return np.array(self.agent_loc).astype(np.int32)


	def get_reward_locs(self):
		return [list(rew.keys()) for rew in self.reward_locations]

if __name__ == '__main__':

	### Diagnostics
	r = RoomStimulus()

	inp = r.make_inputs()
	inpsum = np.sum(inp[:,4:], axis=1)

	act = np.zeros([par['batch_size'], par['n_output']])
	act[:,4] = 1
	rew = r.agent_action(act, np.ones(par['batch_size']))

	print(np.mean(np.minimum(1, inpsum)))       # Check placement
	print(np.mean(rew==1.), np.mean(rew==2.))   # Check rewards

	agent_locs = r.get_agent_locs()
	reward_locs = r.get_reward_locs()

	demo_room = np.zeros([par['room_width'],par['room_height']])
	t = 0
	demo_room[agent_locs[t,1],agent_locs[t,0]] = 1
	demo_room[reward_locs[t][0][1],reward_locs[t][0][0]] = -1
	# demo_room[reward_locs[t][1][1],reward_locs[t][1][0]] = -2

	import matplotlib.pyplot as plt
	plt.imshow(demo_room)
	plt.show()
	