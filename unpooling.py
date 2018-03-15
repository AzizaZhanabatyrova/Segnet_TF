import tensorflow as tf

def unravel_argmax(argmax, shape):

	with tf.device('/cpu:0'):
		output_list = []
		output_list.append(argmax // (shape[2] * shape[3]))
		output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
		return tf.stack(output_list)

def unpool_layer2x2_batch(x, argmax):
        '''
        Args:
            x: 4D tensor of shape [batch_size x height x width x channels]
            argmax: A Tensor of type Targmax. 4-D. The flattened indices of the max
            values chosen for each output.
        Return:
            4D output tensor of shape [batch_size x 2*height x 2*width x channels]
        '''

	with tf.device('/cpu:0'):
		x_shape = tf.shape(x)
		out_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]]

		batch_size = out_shape[0]
		height = out_shape[1]
		width = out_shape[2]
		channels = out_shape[3]

		argmax_shape = tf.to_int64([batch_size, height, width, channels])
		argmax = unravel_argmax(argmax, argmax_shape)

		t1 = tf.to_int64(tf.range(channels))
		t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
		t1 = tf.reshape(t1, [-1, channels])
		t1 = tf.transpose(t1, perm=[1, 0])
		t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
		t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

		t2 = tf.to_int64(tf.range(batch_size))
		t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
		t2 = tf.reshape(t2, [-1, batch_size])
		t2 = tf.transpose(t2, perm=[1, 0])
		t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

		t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

		t = tf.concat([t2, t3, t1], 4)
		indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

		x1 = tf.transpose(x, perm=[0, 3, 1, 2])
		values = tf.reshape(x1, [-1])

		delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))
		return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))
