
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import deque
from parameters import *
import stimulus

class Network:

    def __init__(self):

        self.load_encoder_weights()

    def load_encoder_weights(self):

        fn = './savedir/VAE_8x8_n128_model_weights.pkl'
        results = pickle.load(open(fn, 'rb'))
        self.weights = results['weights']

        N = par['n_latent'] + par['n_pol']
        self.H = np.zeros((1, N, par['n_dend'],N), dtype=np.float32)
        self.Hr = np.zeros((1, N, par['n_dend'],par['n_unique_vals']), dtype=np.float32)


    def stimulus_encoding(self, stim):
        """
        Encode the stimulus into a sparse representation
        """
        N = 5
        x = np.maximum(0, stim @ self.weights['W_enc0'] + self.weights['b_enc0'])
        x = np.maximum(0, x @ self.weights['W_enc1'] + self.weights['b_enc1'])
        latent = x @ self.weights['W_mu'] + self.weights['b_mu']
        latent_sort = np.sort(latent, axis = 1)
        latent = np.where(latent < latent_sort[:, -N], \
            np.zeros_like(latent), np.ones_like(latent))

        return latent

    def stimulus_decoding(self, latent):

    	"""
    	Encode the stimulus into a sparse representation
    	"""
    	x = np.maximum(0, latent @ self.weights['W_dec0'] + self.weights['b_dec0'])
    	x = np.maximum(0, x @ self.weights['W_dec1'] + self.weights['b_dec1'])

    	return x @ self.weights['W_dec2'] + self.weights['b_dec2']

    def read_hopfield(self, latent):

        reward_vector = np.reshape(np.array([0., 1.]), (1,2))
        n_steps = 15
        gamma = 0.9
        total_rewards = []



        for i in range(par['n_pol']):
            action = par['action_template'][i]+0
            next_latent = latent + 0.
            rewards_policy = []
            for j in range(n_steps):
                # next state and action
                #print('Next lat', i,j,next_latent, np.sum(next_latent))
                z = np.concatenate([next_latent, par['act_mult']*action], axis = 1)
                dend_activity = np.einsum('ij,ijkm->ikm', z, np.minimum(1., self.H))
                #print('DEND',i,j, np.max(dend_activity, axis = 1))
                dend_output   = np.maximum(0, dend_activity - par['dend_th'])
                above_th = np.float32(dend_output > 0.)
                neuron_output = np.minimum(1., np.sum(above_th, axis = 1))
                pol_output = neuron_output[:, -par['n_pol']:]
                next_latent = neuron_output[:, :-par['n_pol']]

                next_policy = pol_output/(1e-9 + np.sum(pol_output, axis = 1))
                action = np.random.multinomial(1, next_policy[0,:] - 1e-6)
                action = np.reshape(action, (1, -1))
                #print('action ',i, j,action, pol_output)


                # reward
                dend_activity = np.einsum('ij,ijkm->ikm', z, np.minimum(1., self.Hr))
                dend_output   = np.maximum(0, dend_activity - par['dend_th'])
                above_th = np.float32(dend_output > 0.)
                neuron_output = np.minimum(1., np.sum(above_th, axis = 1))
                rewards_policy.append(np.sum(reward_vector*neuron_output, axis=1)*(gamma**j))

            rewards_policy = np.stack(rewards_policy, axis = 0)
            #print(rewards_policy.shape)
            total_rewards.append(np.sum(rewards_policy, axis = 0))

        total_rewards = np.stack(total_rewards, axis = 1)
        total_rewards = np.reshape(total_rewards, [1, -1])
        #print(total_rewards)

        return total_rewards


    def write_hopfield(self, latent, action, prev_latent, prev_action, prev_reward):

        #latent       = np.float32(latent > 0)
		#prev_latent  = np.float32(prev_latent > 0)
		#prev_action  = np.float32(prev_action > 0)
        #prev_reward  = np.float32(prev_reward > 0)

        z0 = np.concatenate([prev_latent, prev_action], axis = 1)
        z1 = np.concatenate([latent, action], axis = 1)
        zz = np.einsum('ij,ik->ijk', z0, z1)
        zr = np.einsum('ij,ik->ijk', z0, prev_reward)

        z0[:, -par['n_pol']] *= par['act_mult']

        #print('zz', np.sum(zz))

        dend_activity = np.einsum('ij,ijkm->ikm', z0, np.minimum(1., self.H))
        dend_output   = np.maximum(0, dend_activity - par['dend_th'])
        above_th = np.int16(np.sum(dend_output > 0.5, axis = 1) > 0.)
        ind_max = np.argmax(dend_activity, axis = 1)
        ind_min = np.argmin(dend_activity, axis = 1)
        ind = np.int16(ind_max*above_th + ind_min*(1 - above_th))

        for i, j in enumerate(ind[0]):
            self.H[:, :, j, i] += zz[:, :, i]
            #plt.imshow(self.H[0, i, j, :])


        dend_activity = np.einsum('ij,ijkm->ikm', z0, np.minimum(1., self.Hr))
        dend_output   = np.maximum(0, dend_activity - par['dend_th'])
        above_th = np.int16(np.sum(dend_output > 0.5, axis = 1) > 0.)
        ind_max = np.argmax(dend_activity, axis = 1)
        ind_min = np.argmin(dend_activity, axis = 1)
        ind = np.int16(ind_max*above_th + ind_min*(1 - above_th))
        for i, j in enumerate(ind[0]):
            self.Hr[:, :, j, i] += zr[:, :, i]

        #print('REW ', prev_reward, np.sum(zr[0, :, :], axis=0))


class EventLists:

    def __init__(self):

        self.latent_list = []
        self.action_list = []
        self.reward_list = []
        self.max_size = 50
        self.write_data = False

    def add_event(self, network, latent, action, reward, write_data):

        if len(self.latent_list) >= self.max_size:
            self.latent_list = self.latent_list[1:]
            self.action_list = self.action_list[1:]
            self.reward_list = self.reward_list[1:]

        self.latent_list.append(latent)
        self.action_list.append(action)
        self.reward_list.append(reward)

        if self.write_data:
            self.write_hopfield(network)
            self.write_data = False

        self.write_data = True if write_data else False

    def write_hopfield(self, network):
        for i in range(self.max_size - 1):
            network.write_hopfield(self.latent_list[i+1], self.action_list[i+1], \
                self.latent_list[i], self.action_list[i], self.reward_list[i])



def main():

    network = Network()
    environment = stimulus.Stimulus()
    par['batch_size'] = 1
    par['room_width'] = 8
    par['room_height'] = 8

    environment.reward_locations[0][0] = tuple([7, 7])
    environment.agent_loc[0] = [0, 0]
    new_location = environment.get_agent_locs()

    event_lists = EventLists()

    noise_moves(network, environment, event_lists)
    reward_moves(network, environment, event_lists)
    noise_moves(network, environment, event_lists)
    test_moves(network, environment)


def test_moves(network, environment):

    environment.agent_loc[0] = [0, 2]
    stim = environment.make_inputs()
    latent = network.stimulus_encoding(stim)
    network.read_hopfield(latent)


def noise_moves(network, environment, event_lists):

    num_moves = 250
    for i in range(num_moves):
        make_move = False
        location = environment.get_agent_locs()
        stim = environment.make_inputs()
        prev_latent = network.stimulus_encoding(stim)
        while not make_move:
            environment.agent_loc[0] = location[0]
            action = np.random.multinomial(1, [0.25, 0.25, 0.25, 0.25])
            action = np.reshape(action, (1,4))
            reward = environment.agent_action(action)
            new_location = environment.get_agent_locs()
            make_move = False if tuple(location[0]) in environment.reward_locations[0] else True
            #if not make_move:
            #    print(i, new_location, action, reward)
        stim = environment.make_inputs()
        latent = network.stimulus_encoding(stim)
        reward_one_hot = np.zeros((1, par['n_unique_vals']), dtype = np.float32)
        reward_one_hot[0, np.int8(np.squeeze(reward))] = 1.

        event_lists.add_event(network, latent, action, reward_one_hot, reward > 0)



def reward_moves(network, environment, event_lists):

    environment.agent_loc[0] = [7, 0]
    right_action = np.reshape(np.array([1,0,0,0]), (1,4))
    up_action = np.reshape(np.array([0,0,1,0]), (1,4))
    down_action = np.reshape(np.array([0,0,0,1]), (1,4))
    stim = environment.make_inputs()
    prev_latent = network.stimulus_encoding(stim)

    for _ in range(7):
        reward = environment.agent_action(down_action)
        stim = environment.make_inputs()
        latent = network.stimulus_encoding(stim)
        reward_one_hot = np.zeros((1, par['n_unique_vals']), dtype = np.float32)
        reward_one_hot[0, np.int8(np.squeeze(reward))] = 1.
        event_lists.add_event(network, latent, down_action, reward_one_hot, reward > 0)
        prev_latent = latent + 0.
        new_location = environment.get_agent_locs()

    for _ in range(7):
        reward = environment.agent_action(right_action)
        stim = environment.make_inputs()
        latent = network.stimulus_encoding(stim)
        reward_one_hot = np.zeros((1, par['n_unique_vals']), dtype = np.float32)
        reward_one_hot[0, np.int8(np.squeeze(reward))] = 1.
        event_lists.add_event(network, latent, right_action, reward_one_hot, reward > 0)
        prev_latent = latent + 0.
        new_location = environment.get_agent_locs()
        #print(new_location, reward)

    for _ in range(8):
        reward = environment.agent_action(up_action)
        #print('UP', reward)
        stim = environment.make_inputs()
        latent = network.stimulus_encoding(stim)
        reward_one_hot = np.zeros((1, par['n_unique_vals']), dtype = np.float32)
        reward_one_hot[0, np.int8(np.squeeze(reward))] = 1.
        event_lists.add_event(network, latent, up_action, reward_one_hot, reward > 0)
        prev_latent = latent + 0.
        new_location = environment.get_agent_locs()
        #print('XXX', new_location, reward)

    reward_location = environment.get_reward_locs()
    #print('XX', new_location, reward_location)
