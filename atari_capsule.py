import tensorflow as tf

"""
CNN:
 - 8x8 image patch
 - Project into N features
 - Results in feature representation

Capsule:
 - 8x8x4 patch (4=frames)
 - Project into an N x M (features x properties) matrix
 - For example, N x 5, where columns are [presence, x_pos, y_pos, x_vel, y_vel]
 - Presence = sigmoid(column), others = tanh(columns)
 - 8x8x4 patch --> N x 5 cell | 100x84x4 --> 25x21xNx5 result
    - Alternatively, 100x84x4 --> 25x21x(N*5), then reshape

 - Convolve capsule across image (stride 4?)
 - Combine with current action in a dense/flat layer
 - Predict next 25x21xNx5 result
 """


def dense_layer(x, n_out, name, activation=tf.nn.relu, var_dict=None, trainable=True):
	""" Build a dense layer
		x		: input tensor to propagate
		n_out	: number of neurons in `layer
		name	: name of the layer
	"""

	if var_dict is None:
		n_in = x.shape.as_list()[-1]
		W = tf.get_variable('W_'+name, shape=[n_in, n_out], trainable=trainable)
		b = tf.get_variable('b_'+name, shape=[1, n_out], trainable=trainable)
	else:
		W = tf.get_variable('W_'+name, initializer=var_dict['W_'+name], trainable=trainable)
		b = tf.get_variable('b_'+name, initializer=var_dict['b_'+name], trainable=trainable)

	y = x @ W + b

	return activation(y)


def capsule_conv_layer(x, n_features, n_preproperties, n_properties, n_kernel, stride, \
		name, var_dict=None, trainable=True):

	# Use the number of features and properties together as "filters"
	# in the convolutional layer
	n_prefilter = n_features * n_preproperties
	n_filter    = n_features * n_properties

	# No color channels - set to 1
	in_channels = 1

	# Make convolutional filter variable either by creating a new one or
	# using a variable provided through var_dict
	if var_dict is None:
		f0 = tf.get_variable(name+'_filter', trainable=trainable, shape=[4, n_kernel, n_kernel, in_channels, n_prefilter], \
			initializer=tf.variance_scaling_initializer(scale=2.))
		f1 = tf.get_variable(name+'_transf', trainable=trainable, shape=[n_features, n_preproperties, n_properties], \
			initializer=tf.variance_scaling_initializer(scale=2.))
	else:
		f0 = tf.get_variable(name+'_filter', initializer=var_dict[name+'_filter'], trainable=trainable)
		f1 = tf.get_variable(name+'_transf', initializer=var_dict[name+'_transf'], trainable=trainable)

	# Generate first 3D convolution operation
	strides = [1, 4, stride, stride, 1]
	y = tf.nn.conv3d(x, f0, strides, 'SAME')

	# Reshape into [b x 25 x 21 x features x pre_properties]
	y_shape = y.shape.as_list()[:-1]
	y = tf.reshape(y, y_shape + [n_features, n_preproperties])
	y = tf.squeeze(y)

	# Project from 20 units to 5, with separate projects for each feature
	# [batch, x, y, feature, pre_prop], [feature, pre_prop, prop]
	#    --> [batch, x, y, feature, prop]
	y = tf.einsum('bxyfi,fij->bxyfj', y, f1)

	return y


def capsule_conv_activation(x):
	""" Apply different activation functions to different capsule properties """

	# axis=-1 has elements [feature presence, x_pos, y_pos, x_vel, y_vel]
	# Apply sigmoid to feature presence, tanh to all others

	# Apply desired activations on slices of the convolved data
	feature = tf.nn.sigmoid(x[...,0:1])
	others  = tf.nn.tanh(x[...,1:5])

	# Combine the slices
	y = tf.concat([feature, others], axis=-1)
	
	return y


def encoder(x, n_kernel, n_features, n_preproperties, n_properties, var_dict=None, trainable=True):
	""" Convolve the provided data to generate a latent representation.
		Based roughly on Hinton's capsule concept.  Uses a 3d convolution
		to generate convolution across time as well as space.

		Input must be of size [trials x height x width x steps]
		Output will be of size [trials x n_latent]
	"""

	# Isolate encoder variables if provided
	if var_dict is not None:
		vd = {n.split('/')[1] : v for n, v in var_dict.items() if 'encoder' in n}
	else:
		vd = None
		if not trainable:
			print('Encoder is manually set to be untrainable,')
			print('but no pre-trained variables are provided.')
			print('--> Setting trainable to true.\n')
			trainable = True

	# Obtain batch size
	batch_size = x.shape.as_list()[0]

	# Convert to [batch x depth x width x height x channels]
	# x.shape = [b, 4, 100, 84, 1]
	x = tf.transpose(x, [0, 3, 1, 2])[...,tf.newaxis]

	# Run encoder
	with tf.variable_scope('encoder'):

		# Run convolutional layers to compress input data
		conv0 = capsule_conv_layer(x, n_features, n_preproperties, n_properties, \
			n_kernel, 4, name='conv1', var_dict=vd, trainable=trainable)
		caps0 = capsule_conv_activation(conv0)

		# Flatten convolution
		# flat0 = tf.reshape(conv0, [batch_size, -1])

	return caps0