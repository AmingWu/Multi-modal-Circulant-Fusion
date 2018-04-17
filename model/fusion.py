import tensorflow as tf

def recurent_matrix(text_f, visual_f):

    shape = text_f.get_shape().as_list()
    text_set = []
    text0 = tf.expand_dims(text_f, 2)
    text_set.append(text0)

    visual_set = []
    visual0 = tf.expand_dims(visual_f, 2)
    visual_set.append(visual0)

    for i in range(1,shape[1]):

        pre=text0[:,(shape[1]-i):,:]
        host=text0[:,0:(shape[1]-i),:]
        text1=tf.concat([pre,host],1)
        text_set.append(text1)

        pre=visual0[:,(shape[1]-i):,:]
        host=visual0[:,0:(shape[1]-i),:]
        visual1=tf.concat([pre,host],1)
        visual_set.append(visual1)

    text_vector=tf.concat(text_set, 2)
    visual_vector=tf.concat(visual_set, 2)

    text_vector=tf.transpose(text_vector, perm=[0,2,1])
    visual_vector=tf.transpose(visual_vector, perm=[0,2,1])

    return visual_vector, text_vector

def fusion_element_product(text_f, visual_f):

    visual_vector, text_vector = recurent_matrix(text_f, visual_f)

    text_f = tf.expand_dims(text_f, 1)
    visual_f = tf.expand_dims(visual_f, 1)

    fusion_visual = tf.reduce_mean(visual_vector * visual_f, 1)
    fusion_text = tf.reduce_mean(text_vector * text_f, 1)

    return fusion_visual, fusion_text

def fusion_matrix_multiplication(text_f, visual_f):

    visual_vector, text_vector = recurent_matrix(text_f, visual_f)

    text_f = tf.expand_dims(text_f, 2)
    visual_f = tf.expand_dims(visual_f, 2)

    fusion_text = tf.matmul(visual_vector, text_f)
    fusion_visual = tf.matmul(text_vector, visual_f)

    fusion_text = tf.nn.relu(fusion_text)
    fusion_visual = tf.nn.relu(fusion_visual)

    fusion_text = tf.squeeze(fusion_text, 2)
    fusion_visual = tf.squeeze(fusion_visual, 2)

    return fusion_visual, fusion_text