import tensorflow as tf
import sys

BATCH_NORM_MOMENTUM = 0.9
FUSED=False

def prelu(x, name):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - tf.abs(x)) * 0.5

    return pos + neg


def initial_block(x, is_training, n_conv_channels=13):
    with tf.variable_scope('initial_block'):
        initial1 = tf.layers.max_pooling2d(x, pool_size=(2,2), strides=2, padding='same', name='pool')
        initial2 = tf.layers.conv2d(x, filters=n_conv_channels, kernel_size=3, padding='same', strides=2,
                activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="conv")
        x = tf.concat([initial1, initial2], 3)
        # bn + prelu moved to residual branch
        return x


def bottleneck(x, n_filters, internal_scale, downsample, t, is_training, name, dropout_prob,
                dilation_rate=None):
    with tf.variable_scope(name):
        projection_size = n_filters / internal_scale
        kern = 2 if downsample else 1

        # residual branch
        x_bn = tf.layers.batch_normalization(x, momentum=BATCH_NORM_MOMENTUM,
                                             fused=FUSED,
                                             training=is_training, name="initial_bn")
        x_bn = prelu(x_bn, "initial_prelu")

        projection = tf.layers.conv2d(x_bn, filters=projection_size, kernel_size=kern, padding='same',
                strides=kern,
                activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="conv_projection", use_bias=False)
        projection = tf.layers.batch_normalization(projection, momentum=BATCH_NORM_MOMENTUM,
                                                    fused=FUSED,
                                                    training=is_training, name="projection_bn")
        projection = prelu(projection, 'projection_relu')

        if t == 'dilated':
            conved = tf.layers.conv2d(projection, filters=projection_size, kernel_size=3, padding='same',
                    dilation_rate=dilation_rate, activation=None,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    name='dilation')
        elif t == 'asymetric':
            conved = tf.layers.conv2d(projection, filters=projection_size, kernel_size=(5,1), padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    name='asymmetric_5x1', use_bias=False)
            conved = tf.layers.conv2d(conved, filters=projection_size, kernel_size=(1,5), padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    name='asymmetric_1x5')
        elif t == 'regular':
            conved = tf.layers.conv2d(projection, filters=projection_size, kernel_size=3, padding='same',
                strides=1,
                activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="conv")
        elif t == 'deconv':
            conved = tf.layers.conv2d_transpose(projection, filters=projection_size, kernel_size=2,
                    strides=2, padding='same',
                    activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    name='deconv')
        else:
            assert False, name + " " + t
        conved = tf.layers.batch_normalization(conved, momentum=BATCH_NORM_MOMENTUM,
                                                fused=FUSED,
                                                training=is_training, name="main_conv_bn")
        conved = prelu(conved, "main_conv_prelu")

        expansion = tf.layers.conv2d(conved, filters=n_filters, kernel_size=1, padding='same',
                strides=1,
                activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="conv_expansion", use_bias=False)

        s = tf.shape(expansion)
        s = tf.slice(s, [0], [1])
        s = tf.concat([s, [1, 1, n_filters]], 0)
        dropout_prob = 1 - dropout_prob   # FIXME
        if is_training:
            expansion = tf.nn.dropout(expansion, keep_prob=dropout_prob, noise_shape=s)

        # skip branch
        if downsample:
            y = tf.layers.conv2d(x, filters=n_filters, kernel_size=2, padding='same',
                strides=2,
                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="main_downsampling", use_bias=False)
            y = tf.layers.batch_normalization(y, momentum=BATCH_NORM_MOMENTUM, training=is_training,
                                              fused=FUSED,
                                              name="main_downsampling_bn")
            y = prelu(y, 'main_downsampling_prelu')
        else:
            y = x
        x = expansion + y
    return x


def bottleneck_decoder(x, n_filters, internal_scale, upsample, is_training, name,
                        concat_output=None):
    with tf.variable_scope(name):
        # residual branch
        x_bn = tf.layers.batch_normalization(x, momentum=BATCH_NORM_MOMENTUM, training=is_training,
                                             fused=FUSED,
                                             name="initial_bn")
        x_bn = tf.nn.relu(x_bn) # FIXME


        projection_size = n_filters / internal_scale
        projection = tf.layers.conv2d(x_bn, filters=projection_size, kernel_size=1, padding='same',
                strides=1,
                activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="conv_projection", use_bias=False)
        projection = tf.layers.batch_normalization(projection, momentum=BATCH_NORM_MOMENTUM, fused=FUSED,
                                                   training=is_training, name="projection_bn")
        projection = tf.nn.relu(projection)  # FIXME prelu ?


        if upsample:
            conved = tf.layers.conv2d_transpose(projection,
                    filters=projection_size, kernel_size=3, padding='same', strides=2,
                     activation=None,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    name='dilation')
        else:
            conved = tf.layers.conv2d(projection, filters=projection_size, kernel_size=3, padding='same',
                strides=1,
                activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="conv")
        conved = tf.layers.batch_normalization(conved, momentum=BATCH_NORM_MOMENTUM, fused=FUSED,
                                               training=is_training, name="main_conv_bn")
        conved = tf.nn.relu(conved)  # FIXME prelu?


        expansion = tf.layers.conv2d(conved, filters=n_filters, kernel_size=1, padding='same',
                strides=1,
                activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="conv_expansion", use_bias=False)

        # main (skip) branch
        if upsample:
            y = tf.layers.conv2d_transpose(x,
                    filters=n_filters, kernel_size=2, padding='same', strides=2,
                     activation=None,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    name='main_dilation')
            #y = tf.slice(y, [0,0,0,0], tf.shape(concat_output))
            y = y[:,:tf.shape(concat_output)[1], :tf.shape(concat_output)[2]]
            #y.set_shape(concat_output.get_shape())
            y = tf.concat([y, concat_output], 3)
            y = tf.layers.batch_normalization(y, momentum=BATCH_NORM_MOMENTUM, training=is_training,
                                              fused=FUSED,
                                              name="main_bn")
            y = prelu(y, 'prelu_main')

            y = tf.layers.conv2d(y, filters=n_filters, kernel_size=1, padding='same',
                strides=1, activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="refinement", use_bias=False)
            expansion = tf.slice(expansion, [0,0,0,0], tf.shape(y))
        else:
            y = x

        x = expansion + y
    return x

def build_graph(x, is_training, n_output_channels):
    # x - 4d tensor
    input_h = tf.shape(x)[1]
    input_w = tf.shape(x)[2]

    x = initial_block(x, is_training, n_conv_channels=5)

    skip_connection_starts = []

    skip_connection_starts.append(x)
    x = bottleneck(x, n_filters=32, internal_scale=4, downsample=True, t='regular',
            is_training=is_training, name="bottleneck0.0", dropout_prob=0.01)
    for i in xrange(1,3):
        x = bottleneck(x, n_filters=32, internal_scale=4, downsample=False, t='regular',
                is_training=is_training, name="bottleneck0.{}".format(i), dropout_prob=0.01)

    skip_connection_starts.append(x)
    x = bottleneck(x, n_filters=64, internal_scale=4, downsample=True, t='regular',
            is_training=is_training, name="bottleneck1.0", dropout_prob=0.01)
    for i in xrange(1,5):
        x = bottleneck(x, n_filters=64, internal_scale=4, downsample=False, t='regular',
                is_training=is_training, name="bottleneck1.{}".format(i), dropout_prob=0.01)

    skip_connection_starts.append(x)
    x = bottleneck(x, n_filters=128, internal_scale=4, downsample=True, t='regular',
            is_training=is_training, name="bottleneck2.0", dropout_prob=0.1)
    for j in xrange(2,4):
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='regular',
                is_training=is_training, name="bottleneck{}.1".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated', dilation_rate=2,
                is_training=is_training, name="bottleneck{}.2".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='asymetric',
                is_training=is_training, name="bottleneck{}.3".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated', dilation_rate=4,
                is_training=is_training, name="bottleneck{}.4".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='regular',
                is_training=is_training, name="bottleneck{}.5".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated', dilation_rate=8,
                is_training=is_training, name="bottleneck{}.6".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='asymetric',
                is_training=is_training, name="bottleneck{}.7".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated', dilation_rate=16,
                is_training=is_training, name="bottleneck{}.8".format(j), dropout_prob=0.1)


    x = bottleneck_decoder(x, 64, internal_scale=4, upsample=True, is_training=is_training,
            name='decoder/bottleneck4.0', concat_output=skip_connection_starts[-1])
    skip_connection_starts.pop()
    for i in xrange(1,3):
        x = bottleneck_decoder(x, 64, internal_scale=4, upsample=False, is_training=is_training,
                name='decoder/bottleneck4.{}'.format(i))

    x = bottleneck_decoder(x, 32, internal_scale=4, upsample=True, is_training=is_training,
            name='decoder/bottleneck5.0', concat_output=skip_connection_starts[-1])
    skip_connection_starts.pop()
    x = bottleneck_decoder(x, 32, internal_scale=4, upsample=False, is_training=is_training,
            name='decoder/bottleneck5.1')

    x = bottleneck_decoder(x, 16, internal_scale=4, upsample=True, is_training=is_training,
            name='decoder/bottleneck6.0', concat_output=skip_connection_starts[-1])
    skip_connection_starts.pop()
    x = bottleneck_decoder(x, 16, internal_scale=4, upsample=False, is_training=is_training,
            name='decoder/bottleneck6.1')

    logits = tf.layers.conv2d_transpose(x, filters=n_output_channels, kernel_size=2, strides=2, padding='same',
            activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            name='conv_decoder_final')
    logits = logits[:,:input_h, :input_w]
    return logits

def build_graph2(x, is_training, n_output_channels):
    # x - 4d tensor
    input_h = tf.shape(x)[1]
    input_w = tf.shape(x)[2]

    x = initial_block(x, is_training, n_conv_channels=5)

    skip_connection_starts = []

    #skip_connection_starts.append(x)
    #x = bottleneck(x, n_filters=32, internal_scale=4, downsample=True, t='regular',
    #        is_training=is_training, name="bottleneck0.0", dropout_prob=0.01)
    #for i in xrange(1,3):
    #    x = bottleneck(x, n_filters=32, internal_scale=4, downsample=False, t='regular',
    #            is_training=is_training, name="bottleneck0.{}".format(i), dropout_prob=0.01)

    skip_connection_starts.append(x)
    x = bottleneck(x, n_filters=64, internal_scale=4, downsample=True, t='regular',
            is_training=is_training, name="bottleneck1.0", dropout_prob=0.01)
    for i in xrange(1,5):
        x = bottleneck(x, n_filters=64, internal_scale=4, downsample=False, t='regular',
                is_training=is_training, name="bottleneck1.{}".format(i), dropout_prob=0.01)

    skip_connection_starts.append(x)
    x = bottleneck(x, n_filters=128, internal_scale=4, downsample=True, t='regular',
            is_training=is_training, name="bottleneck2.0", dropout_prob=0.1)
    for j in xrange(2,4):
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='regular',
                is_training=is_training, name="bottleneck{}.1".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated', dilation_rate=2,
                is_training=is_training, name="bottleneck{}.2".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='asymetric',
                is_training=is_training, name="bottleneck{}.3".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated', dilation_rate=4,
                is_training=is_training, name="bottleneck{}.4".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='regular',
                is_training=is_training, name="bottleneck{}.5".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated', dilation_rate=8,
                is_training=is_training, name="bottleneck{}.6".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='asymetric',
                is_training=is_training, name="bottleneck{}.7".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated', dilation_rate=16,
                is_training=is_training, name="bottleneck{}.8".format(j), dropout_prob=0.1)


    x = bottleneck_decoder(x, 64, internal_scale=4, upsample=True, is_training=is_training,
            name='decoder/bottleneck4.0', concat_output=skip_connection_starts[-1])
    skip_connection_starts.pop()
    for i in xrange(1,3):
        x = bottleneck_decoder(x, 64, internal_scale=4, upsample=False, is_training=is_training,
                name='decoder/bottleneck4.{}'.format(i))

    x = bottleneck_decoder(x, 32, internal_scale=4, upsample=True, is_training=is_training,
            name='decoder/bottleneck5.0', concat_output=skip_connection_starts[-1])
    skip_connection_starts.pop()
    x = bottleneck_decoder(x, 32, internal_scale=4, upsample=False, is_training=is_training,
            name='decoder/bottleneck5.1')

    #x = bottleneck_decoder(x, 16, internal_scale=4, upsample=True, is_training=is_training,
    #        name='decoder/bottleneck6.0', concat_output=skip_connection_starts[-1])
    #skip_connection_starts.pop()
    #x = bottleneck_decoder(x, 16, internal_scale=4, upsample=False, is_training=is_training,
    #        name='decoder/bottleneck6.1')

    logits = tf.layers.conv2d_transpose(x, filters=n_output_channels, kernel_size=2, strides=2, padding='same',
            activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            name='conv_decoder_final')
    logits = logits[:,:input_h, :input_w]
    return logits
