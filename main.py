import tensorflow as tf
from model.fusion import fusion_element_product
from model.init import fc
from model.eval import compute_loss_reg, calculate_IoU, nms_temporal, compute_IoU_recall_top_n_forreg
from model.eval import do_eval_slidingclips
from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import pooling_layer as pool
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from model.dataset import TestingDataSet
from model.dataset import TrainingDataSet
import os
import numpy as np
from six.moves import xrange
import time
from sklearn.metrics import average_precision_score
import pickle
import operator

initial_steps = 0
max_steps = 20000
batch_size = 64

test_batch_size = 1
vs_lr = 0.001
lambda_regression = 0.01
alpha = 1.0/batch_size
semantic_size = 1024 # the size of visual and semantic comparison size
sentence_embedding_size = 4800
visual_feature_dim = 4096*3

train_csv_path = "/home/wam/Action_Recognition/TACoS/train_clip-sentvec.pkl"
test_csv_path = "/home/wam/Action_Recognition/TACoS/test_clip-sentvec.pkl"
test_feature_dir="/home/wam/Action_Recognition/Interval128_256_overlap0.8_c3d_fc6/"
train_feature_dir = "/home/wam/Action_Recognition/Interval64_128_256_512_overlap0.8_c3d_fc6/"
train_set=TrainingDataSet(train_feature_dir, train_csv_path, batch_size)
test_set=TestingDataSet(test_feature_dir, test_csv_path, test_batch_size)

vs = tf.get_variable("vs", [1024, 96], initializer=tf.random_normal_initializer())
tx = tf.get_variable("tx", [1024, 96], initializer=tf.random_normal_initializer())
vs_f = tf.get_variable("vs_f", [96, 1024], initializer=tf.random_normal_initializer())

visual_featmap_ph_test = tf.placeholder(tf.float32, shape=(test_batch_size, visual_feature_dim))
sentence_ph_test = tf.placeholder(tf.float32, shape=(test_batch_size, sentence_embedding_size))

visual_featmap_ph_train = tf.placeholder(tf.float32, shape=(batch_size, visual_feature_dim))
sentence_ph_train = tf.placeholder(tf.float32, shape=(batch_size, sentence_embedding_size))
offset_ph = tf.placeholder(tf.float32, shape=(batch_size,2))

def fill_feed_dict_train_reg():
    image_batch, sentence_batch, offset_batch = train_set.next_batch_iou()
    input_feed = {
            visual_featmap_ph_train: image_batch,
            sentence_ph_train: sentence_batch,
            offset_ph: offset_batch
    }

    return input_feed

def modal_fusion(visual_f, text_f):

    visual_f = tf.matmul(visual_f, vs)
    text_f = tf.matmul(text_f, tx)

    #First stage
    fusion_visual, fusion_text = fusion_element_product(text_f, visual_f)
    #fusion

    fusion_visual = tf.matmul(fusion_visual, vs_f)
    fusion_text = tf.matmul(fusion_text, vs_f)

    return fusion_visual, fusion_text

def cross_modal_comb(visual_feat, sentence_embed, batch_size):

    visual_feat, sentence_embed = modal_fusion(visual_feat, sentence_embed)

    visual_feat = tf.nn.l2_normalize(visual_feat, dim=1)
    sentence_embed = tf.nn.l2_normalize(sentence_embed, dim=1)

    vv_feature = tf.reshape(tf.tile(visual_feat, [batch_size, 1]), [batch_size, batch_size, semantic_size])
    ss_feature = tf.reshape(tf.tile(sentence_embed,[1, batch_size]),[batch_size, batch_size, semantic_size])
    concat_feature = tf.reshape(tf.concat([vv_feature, ss_feature], 2),[batch_size, batch_size, semantic_size+semantic_size])

    mul_feature = vv_feature * ss_feature
    add_feature = vv_feature + ss_feature

    comb_feature = tf.reshape(tf.concat([mul_feature, add_feature, concat_feature], 2),[1, batch_size, batch_size, semantic_size*4])
    return comb_feature

def vs_multilayer(input_batch,name,middle_layer_dim=1000,reuse=False):
    with tf.variable_scope(name):
        if reuse==True:
            print name+" reuse variables"
            tf.get_variable_scope().reuse_variables()
        else:
            print name+" doesn't reuse variables"

        layer1 = conv_relu('layer1', input_batch,
                        kernel_size=1,stride=1,output_dim=middle_layer_dim)
        sim_score = conv('layer2', layer1,
                        kernel_size=1,stride=1,output_dim=3)
    return sim_score

def visual_semantic_infer(visual_feature_train, sentence_embed_train, visual_feature_test, sentence_embed_test):
    name="CTRL_Model"
    with tf.variable_scope(name):
        print "Building training network...............................\n"     
        transformed_clip_train = fc('v2s_lt', visual_feature_train, output_dim=semantic_size) 
        transformed_sentence_train = fc('s2s_lt', sentence_embed_train, output_dim=semantic_size)
        cross_modal_vec_train = cross_modal_comb(transformed_clip_train, transformed_sentence_train, batch_size)
        sim_score_mat_train = vs_multilayer(cross_modal_vec_train, "vs_multilayer_lt", middle_layer_dim=1000)
        sim_score_mat_train = tf.reshape(sim_score_mat_train,[batch_size, batch_size, 3])

        tf.get_variable_scope().reuse_variables()
        print "Building test network...............................\n" 
        transformed_clip_test = fc('v2s_lt', visual_feature_test, output_dim=semantic_size)
        transformed_sentence_test = fc('s2s_lt', sentence_embed_test, output_dim=semantic_size)
        cross_modal_vec_test = cross_modal_comb(transformed_clip_test, transformed_sentence_test, test_batch_size)
        sim_score_mat_test = vs_multilayer(cross_modal_vec_test, "vs_multilayer_lt", reuse=True, middle_layer_dim=1000)
        sim_score_mat_test = tf.reshape(sim_score_mat_test, [3])

        return sim_score_mat_train, sim_score_mat_test

if __name__ == '__main__':

    sim_reg_mat, sim_reg_mat_test = visual_semantic_infer(visual_featmap_ph_train, sentence_ph_train, visual_featmap_ph_test, sentence_ph_test)
    loss_align_reg, offset_pred, loss_reg = compute_loss_reg(sim_reg_mat, offset_ph)
    vs_eval_op = sim_reg_mat_test
    g_optim = tf.train.AdamOptimizer(vs_lr).minimize(loss_align_reg)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=30)
    ckpt = tf.train.get_checkpoint_state('checkpoints/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
        print "Restored Epoch ", epoch_n
    else:
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        init = tf.global_variables_initializer()
        sess.run(init)

    test_result_output=open("ctrl_test_results.txt", "w")
    for step in xrange(max_steps):
        start_time = time.time()
        feed_dict = fill_feed_dict_train_reg()
        _, loss_value, offset_pred_v, loss_reg_v = sess.run([g_optim, loss_align_reg, offset_pred, loss_reg], feed_dict=feed_dict)
        duration = time.time() - start_time

        if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))

        if (step+1) % 2000 == 0:
            print "Start to test:-----------------\n"
            saver.save(sess, 'checkpoints/model.ckpt', step)
            movie_length_info=pickle.load(open("/home/wam/Action_Recognition/TALL-master/video_allframes_info.pkl", 'rb'))
            do_eval_slidingclips(sess, vs_eval_op, movie_length_info, step+1, test_result_output, visual_featmap_ph_test, sentence_ph_test)