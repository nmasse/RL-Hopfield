### Authors: Nicolas Y. Masse, Gregory D. Grant

### To use:
###		# atari_frames.shape   == [trials x height x width x channels]
###		# reconstruction.shape == [trials x height x width x channels]
###		# latent.shape     == [trials x n_latent]
###		# new_latent.shape == [trials x ?]
###
###		latent, loss, conv_shapes = encoder(atari_frames, n_latent)
###		<operate on latent, including resizing if desired>
###		reconstruction            = decoder(new_latent, conv_shapes)


# Required packages
import tensorflow as tf


def filter_init(size):
	""" Using the given size, make the initialization
		for a filter variable """

	return tf.random_uniform(size, -0.1, 0.1)


def dense_layer(x, n_out, name, activation=tf.nn.relu):
	""" Build a dense layer with RELU activation
		x		: input tensor to propagate
		n_out	: number of neurons in layer
		name	: name of the layer
	"""

	n_in = x.shape.as_list()[-1]
	W = tf.get_variable('W_'+name, shape=[n_in, n_out])
	b = tf.get_variable('b_'+name, shape=[1, n_out])

	y = x @ W + b

	return activation(y)


def conv_layer(x, n_filter, n_kernel, stride, name, activation=tf.nn.relu):
	""" Build a convolutional layer from the provided data
		x			: input tensor to be convolved
		n_filter	: number of output filters
		n_kernel	: size of the kernel
		stride		: stride length of the convolution
		name		: name of the layer
	"""

	# Get number of input channels for making filter
	in_channels = x.shape.as_list()[-1]

	# Make convolutional filter variable
	f = tf.get_variable(name, \
		initializer=filter_init([n_kernel, n_kernel, in_channels, n_filter]))
	
	# Generate convolution operation
	strides = [1, stride, stride, 1]
	y = activation(tf.nn.conv2d(x, f, strides, 'SAME'))
	
	return y


def deconv_layer(x, n_filter, n_kernel, stride, shape, name, activation=tf.nn.relu):
	""" Build a convolutional layer from the provided data
		x			: input tensor to be deconvolved
		n_filter	: number of output filters
		n_kernel	: size of the kernel
		stride		: stride length of the deconvolution
		shape		: output shape of the deconvolution
		name		: name of the layer
	"""

	# Get number of input channels for making filter
	in_channels = x.shape.as_list()[-1]

	# Make deconvolutional filter variable
	f = tf.get_variable(name, \
		initializer=filter_init([n_kernel, n_kernel, n_filter, in_channels]))

	# Generate deconvolution operation
	strides = [1, stride, stride, 1]
	y = activation(tf.nn.conv2d_transpose(x, f, shape, strides, 'SAME'))

	return y


def encoder(data0, n_latent):
	""" Convolve the provided data to generate a latent representation.
		Based on the DQN architecture.

		Input must be of size [trials x height x width x channels]
		Output will be of size [trials x n_latent]
	"""

	# Obtain batch size
	batch_size = data0.shape.as_list()[0]

	# Run encoder
	with tf.variable_scope('encoder'):

		# Run convolutional layers to compress input data
		conv0  = conv_layer(data0, n_filter=32, n_kernel=2, stride=1, name='conv0')
		conv1  = conv_layer(conv0, n_filter=32, n_kernel=8, stride=4, name='conv1')
		conv2  = conv_layer(conv1, n_filter=64, n_kernel=4, stride=2, name='conv2')
		conv3  = conv_layer(conv2, n_filter=64, n_kernel=4, stride=2, name='conv3')

		# Flatten convolution output, apply dense layers to make latent vector
		flat0  = tf.reshape(conv3, [batch_size, -1])
		dense0 = dense_layer(flat0,  n_out=2048,     name='dense0')
		dense1 = dense_layer(dense0, n_out=2048,     name='dense1')
		latent = dense_layer(dense1, n_out=n_latent, name='latent0', activation=tf.identity)

	# Collect the convolutional shapes for later decovolution
	conv_shapes = [v.shape.as_list() for v in [data0, conv0, conv1, conv2, conv3]]

	return latent, conv_shapes


def decoder(latent, conv_shapes):
	""" Deconvolve the provided latent vector to generate a reconstruction.
		Based on the inverse of the DQN architecture.

		Input must be of size [trials x n_latent]
		Output will be of size [trials x height x width x channels]
	"""

	# Get the number of elements to produce from the dense layer
	n = conv_shapes[-1][1] * conv_shapes[-1][2] * conv_shapes[-1][3]

	# Convert convolutional shapes into output shapes for
	# the convolutional layers
	s = conv_shapes[-2::-1]

	# Run decoder
	with tf.variable_scope('decoder'):

		# Apply dense layers from latent vector, reshape for deconvolution
		dense0  = dense_layer(latent, n_out=2048, name='dense0')
		dense1  = dense_layer(latent, n_out=2048, name='dense1')
		dense2  = dense_layer(dense0, n_out=n,    name='dense2')
		unflat0 = tf.reshape(dense2, conv_shapes[-1])

		# Run deconvolutional layers to produce reconstruction
		deconv0 = deconv_layer(unflat0, n_filter=64, n_kernel=4, stride=2, shape=s[0], name='deconv0')
		deconv1 = deconv_layer(deconv0, n_filter=32, n_kernel=4, stride=2, shape=s[1], name='deconv1')
		deconv2 = deconv_layer(deconv1, n_filter=32, n_kernel=8, stride=4, shape=s[2], name='deconv2')
		recon   = deconv_layer(deconv2, n_filter=4 , n_kernel=2, stride=1, shape=s[3], name='deconv3', activation=tf.identity)
		# recon   = deconv_layer(deconv1, n_filter=4 , n_kernel=8, stride=4, shape=s[2], name='deconv3', activation=tf.identity)

	return recon