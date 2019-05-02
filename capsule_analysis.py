import pickle
import numpy as np
import tensorflow as tf
import os, sys, time

import matplotlib.pyplot as plt
from itertools import product

from atari_parameters import par, update_parameters
from atari_stimulus import Stimulus
from atari_capsule import encoder

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU setup
gpu_id = sys.argv[1] if len(sys.argv) >= 2 else None
if gpu_id is not None: os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

###############################################################################

def load_capsule(weights, gpu_id):

	# Select device, make placeholder, and generate session
	device = '/cpu:0' if gpu_id is None else '/gpu:0'
	x_pl   = tf.placeholder(tf.float32, [par['batch_size'], 100, 84, 4], 'input')
	sess   = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

	# Load capsule encoder into the graph
	with tf.device(device):
		capsule_out = encoder(x_pl, par['kernel_size'], par['n_features'], \
			par['n_preproperties'], par['n_properties'], \
			var_dict=weights, trainable=False)

	# Initialize
	sess.run(tf.global_variables_initializer())

	# Make a function to run the capsule encoder for a given input
	def run_capsule(x):
		return sess.run(capsule_out, feed_dict={x_pl:x})

	# Return the run function
	return run_capsule


###############################################################################

print('Loading data...')
fn   = 'capsule_weights_batch32_5drop20_kernel8_predcost0001.pkl'
data = pickle.load(open('./savedir/'+fn, 'rb'))

print('Updating parameters...')
update_parameters(data['par'], verbose=False, update_deps=False)
weights = data['weights']

print('Loading complete.  Starting analysis.\n')

s = Stimulus()
obs = s.reset_environments()

print('Kernel size:         {}'.format(par['kernel_size']))
print('Prediction cost:     {}'.format(par['pred_cost']))
print('Num features:        {}'.format(par['n_features']))
print('Num pre-properties:  {}'.format(par['n_preproperties']))
print('Num properties:      {}'.format(par['n_properties']))
print('Input shape:         {}'.format(obs.shape))

# Get convolution weight:
# [b x 4 x 100 x 84 x 1] --> [b x 1 x 25 x 21 x (features x pre_properties)]
W_conv = weights['encoder/conv1_filter']	# [steps, kernel, kernel, channel, features x pre_props]
W_proj= weights['encoder/conv1_transf']		# [features, pre-properties, properties]

print('\nConvolution filter, to analyze patches:')
print('\t', W_conv.shape)
print('Projection matrix, to go from {} proto-properties to {} properties:'.format(\
	par['n_preproperties'], par['n_properties']))
print('\t', W_proj.shape)

print('\n' + '-'*40 + '\n')

# Load capsule encoder into TensorFlow and get run function
run_capsule = load_capsule(weights, gpu_id)

# Get an example of a capsule output
demo = run_capsule(obs)
print('Capsule output shape:', demo.shape)

# Iterate over capsule locations to observe feature/property values
x_locs = range(25)
y_locs = range(21)
for x, y in product(x_locs, y_locs):
	plt.imshow(demo[0,x,y,:,:], aspect='auto')
	plt.xlabel('Properties')
	plt.ylabel('Features')
	plt.colorbar()
	plt.title('Location ({}, {})'.format(x, y))
	plt.savefig('./plotdir/capsule_x{:0>2}_y{:0>2}.png'.format(x,y), bbox_inches='tight')
	plt.clf()