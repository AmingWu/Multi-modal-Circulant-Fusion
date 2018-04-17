import tensorflow as tf

def fc(name, bottom, output_dim, bias_term=True, weights_initializer=None,
             biases_initializer=None):
    # flatten bottom input
    # input has shape [batch, in_height, in_width, in_channels]
    shape = bottom.get_shape().as_list()
    input_dim = 1
    for d in shape[1:]:
        input_dim *= d
    flat_bottom = tf.reshape(bottom, [-1, input_dim])
    
    # weights and biases variables
    with tf.variable_scope(name):
        if weights_initializer is None and biases_initializer is None:
            # initialize the variables
            if weights_initializer is None:
                weights_initializer = tf.random_normal_initializer()
            if bias_term and biases_initializer is None:
                biases_initializer = tf.constant_initializer(0.)

            # weights has shape [input_dim, output_dim]
            weights = tf.get_variable("weights", [input_dim, output_dim],
                initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", output_dim,
                    initializer=biases_initializer)

    if bias_term:
        fc = tf.nn.xw_plus_b(flat_bottom, weights, biases)
    else:
        fc = tf.matmul(flat_bottom, weights)
    return fc