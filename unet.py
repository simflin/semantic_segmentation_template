import tensorflow as tf
FUSED=False


def add_block(prev_layer_input, is_training, n_filters, kern_size=3, stride=1):
    for i in xrange(len(n_filters)):
        prev_layer_input = tf.layers.conv2d(prev_layer_input, filters=n_filters[i],
                kernel_size=kern_size, padding='same', strides=stride,
                activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="conv_{}".format(i), use_bias=False)
        if i == len(n_filters)-1:
            tf.add_to_collection('checkpoints', prev_layer_input)
        prev_layer_input = tf.layers.batch_normalization(prev_layer_input, momentum=0.9, fused=FUSED,
                                training=is_training, name="bn_{}".format(i))
        prev_layer_input = tf.nn.relu(prev_layer_input, name="relu_{}".format(i))
    return prev_layer_input

def psp_module(x, psp_level_sizes, n_channels, name, is_training):
    with tf.variable_scope(name):
        out = [x]
        out_size = x.get_shape()[1:3]
        print "PSP input shape", out_size
        for level_size in psp_level_sizes:
            with tf.variable_scope('psp_{}'.format(level_size)):
                #pooled_map = tf.image.resize_bilinear(x, [level_size,level_size], name="pool")
                if level_size == 1:
                    pooled_map = tf.reduce_mean(x,axis=[1,2], keep_dims=True, name="global_average_pooling")
                else:
                    s = out_size[0]/level_size
                    pooled_map = tf.layers.average_pooling2d(x, (s,s), (s,s), 'same', name="pool")
                conved = tf.layers.conv2d(pooled_map, filters=n_channels, kernel_size=1, padding='same', strides=1,
                        activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        name="conv")
                conved = tf.layers.batch_normalization(conved, momentum=0.9, fused=FUSED,
                                training=is_training, name="bn")
                conved = tf.nn.relu(conved, name="relu")
                out.append(tf.image.resize_nearest_neighbor(conved, out_size, name='upsampling'))
        out = tf.concat(out, axis=3)
    return out

def build_graph(x, is_training, n_output_channels):
    # x - 4d tensor
    ENCODER_CONV_SIZES = [[64, 64], [128, 128], [256, 256], [512, 512], [1024, 1024]]
    #sizes =            x    / 2     /2          / 2               /2      / 2
    #final size is   x / 32
    DECODER_CONV_SIZES = [[512, 512], [256, 256], [128, 128], [64, 64]]

    skip_ends = []
    layer_input = x
    with tf.variable_scope("encoder"):
        for i, block in enumerate(ENCODER_CONV_SIZES):
            with tf.variable_scope("block_{}".format(i)):
                layer_input = add_block(layer_input, is_training, block, kern_size=3, stride=1)
                if i != len(ENCODER_CONV_SIZES) - 1:
                    skip_ends.append(layer_input)
                    layer_input = tf.layers.max_pooling2d(layer_input, pool_size=(2,2), strides=2,
                                                        padding='same', name='pooling')

    layer_input = psp_module(layer_input, [1,2,3,6], 128, name='psp_module', is_training=is_training)
    conved = tf.layers.conv2d(layer_input, filters=ENCODER_CONV_SIZES[-1][-1], kernel_size=1,
                    padding='same', strides=1,
                    activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    name="psp_out_conv")
    conved = tf.layers.batch_normalization(conved, momentum=0.9, fused=FUSED,
                                training=is_training, name="psp_out_bn")
    layer_input = tf.nn.relu(conved, name="psp_out_relu")

    with tf.variable_scope("decoder"):
        for i, block in enumerate(DECODER_CONV_SIZES):
            with tf.variable_scope("block_{}".format(i)):
                layer_input = tf.layers.conv2d_transpose(layer_input, filters=block[0],
                            kernel_size=2, strides=2, padding='same', activation=None,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            name='deconv')
                layer_input = tf.concat([skip_ends[-1], layer_input], 3)
                skip_ends.pop()
                layer_input = add_block(layer_input, is_training, block, kern_size=3, stride=1)

        #output = []
        #for i, elem in enumerate(n_output_channels):
        #    output.append(tf.layers.conv2d(layer_input, filters=elem,
        #        kernel_size=1, padding='same', strides=1,
        #        activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        #        name="final_output_{}".format(i)))
        output = tf.layers.conv2d(layer_input, filters=n_output_channels,
                kernel_size=1, padding='same', strides=1,
                activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="final_output")
    return output

def build_graph2(x, is_training, n_output_channels):
    # x - 4d tensor
    ENCODER_CONV_SIZES = [[64, 64], [128, 128], [256, 256], [512, 512], [1024, 1024]]
    #sizes =            x    / 2     /2          / 2               /2      / 2
    #final size is   x / 32
    DECODER_CONV_SIZES = [[512, 512], [256, 256], [128, 128], [64, 64]]

    skip_ends = []
    layer_input = x
    with tf.variable_scope("encoder"):
        for i, block in enumerate(ENCODER_CONV_SIZES):
            with tf.variable_scope("block_{}".format(i)):
                layer_input = add_block(layer_input, is_training, block, kern_size=3, stride=1)
                if i != len(ENCODER_CONV_SIZES) - 1:
                    skip_ends.append(layer_input)
                    layer_input = tf.layers.max_pooling2d(layer_input, pool_size=(2,2), strides=2,
                                                        padding='same', name='pooling')

    conved = tf.layers.conv2d(layer_input, filters=ENCODER_CONV_SIZES[-1][-1], kernel_size=1,
                    padding='same', strides=1,
                    activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    name="psp_out_conv")
    conved = tf.layers.batch_normalization(conved, momentum=0.9, fused=FUSED,
                                training=is_training, name="psp_out_bn")
    layer_input = tf.nn.relu(conved, name="psp_out_relu")

    with tf.variable_scope("decoder"):
        for i, block in enumerate(DECODER_CONV_SIZES):
            with tf.variable_scope("block_{}".format(i)):
                layer_input = tf.layers.conv2d_transpose(layer_input, filters=block[0],
                            kernel_size=2, strides=2, padding='same', activation=None,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            name='deconv')
                layer_input = tf.concat([skip_ends[-1], layer_input], 3)
                skip_ends.pop()
                layer_input = add_block(layer_input, is_training, block, kern_size=3, stride=1)

        #output = []
        #for i, elem in enumerate(n_output_channels):
        #    output.append(tf.layers.conv2d(layer_input, filters=elem,
        #        kernel_size=1, padding='same', strides=1,
        #        activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        #        name="final_output_{}".format(i)))
        output = tf.layers.conv2d(layer_input, filters=n_output_channels,
                kernel_size=1, padding='same', strides=1,
                activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="final_output")
    return output
