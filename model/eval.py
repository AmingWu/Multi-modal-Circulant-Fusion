import tensorflow as tf
import numpy as np
from six.moves import xrange
import time
from sklearn.metrics import average_precision_score
import pickle
import operator
from dataset import TestingDataSet
from dataset import TrainingDataSet
from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import pooling_layer as pool
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
import os

initial_steps = 0
max_steps = 20000
batch_size = 64
train_csv_path = "/home/wam/Action_Recognition/TACoS/train_clip-sentvec.pkl"
test_csv_path = "/home/wam/Action_Recognition/TACoS/test_clip-sentvec.pkl"
test_feature_dir="/home/wam/Action_Recognition/Interval128_256_overlap0.8_c3d_fc6/"
train_feature_dir = "/home/wam/Action_Recognition/Interval64_128_256_512_overlap0.8_c3d_fc6/"

test_batch_size = 1
vs_lr = 0.001
lambda_regression = 0.01
alpha = 1.0/batch_size
semantic_size = 1024 # the size of visual and semantic comparison size
sentence_embedding_size = 4800
visual_feature_dim = 4096*3
train_set=TrainingDataSet(train_feature_dir, train_csv_path, batch_size)
test_set=TestingDataSet(test_feature_dir, test_csv_path, test_batch_size)

def compute_loss_reg(sim_reg_mat, offset_label):
    sim_score_mat, p_reg_mat, l_reg_mat = tf.split(sim_reg_mat, 3, 2)
    sim_score_mat = tf.reshape(sim_score_mat, [batch_size, batch_size])
    l_reg_mat = tf.reshape(l_reg_mat, [batch_size, batch_size])
    p_reg_mat = tf.reshape(p_reg_mat, [batch_size, batch_size])
    # unit matrix with -2
    I_2 = tf.diag(tf.constant(-2.0, shape=[batch_size]))
    all1 = tf.constant(1.0, shape=[batch_size, batch_size])
    mask_mat = tf.add(I_2, all1)
    # loss cls, not considering iou
    I = tf.diag(tf.constant(1.0, shape=[batch_size]))
    I_half = tf.diag(tf.constant(0.5, shape=[batch_size]))
    batch_para_mat = tf.constant(alpha, shape=[batch_size, batch_size])
    para_mat = tf.add(I,batch_para_mat)
    loss_mat = tf.log(tf.add(all1, tf.exp(mask_mat * sim_score_mat)))
    loss_mat = loss_mat * para_mat
    loss_align = tf.reduce_mean(loss_mat)
    # regression loss
    l_reg_diag = tf.matmul(l_reg_mat * I, tf.constant(1.0, shape=[batch_size, 1]))
    p_reg_diag = tf.matmul(p_reg_mat * I, tf.constant(1.0, shape=[batch_size, 1]))
    offset_pred = tf.concat((p_reg_diag, l_reg_diag), 1)
    loss_reg = tf.reduce_mean(tf.abs(offset_pred - offset_label))

    loss=tf.add(lambda_regression * loss_reg, loss_align)
    return loss, offset_pred, loss_reg

def calculate_IoU(i0,i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick

    union = map(operator.sub, x2, x1) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick

def compute_IoU_recall_top_n_forreg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips):
    correct_num = 0.0
    for k in range(sentence_image_mat.shape[0]):
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k,:,0]]
        ends = [e for e in sentence_image_reg_mat[k,:,1]]
        picks = nms_temporal(starts,ends, sim_v, iou_thresh-0.05)
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        if top_n<len(picks): picks=picks[0:top_n]
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
            if iou>=iou_thresh:
                correct_num+=1
                break
    return correct_num

#visual_featmap_ph_test = tf.placeholder(tf.float32, shape=(test_batch_size, visual_feature_dim))
#sentence_ph_test = tf.placeholder(tf.float32, shape=(test_batch_size, sentence_embedding_size))
def do_eval_slidingclips(sess, vs_eval_op, movie_length_info, iter_step, test_result_output, visual_featmap_ph_test, sentence_ph_test):
    IoU_thresh = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_correct_num_10 = [0.0]*5
    all_correct_num_5 = [0.0]*5
    all_correct_num_1 = [0.0]*5
    all_retrievd = 0.0
    for movie_name in test_set.movie_names:
        movie_length=movie_length_info[movie_name.split(".")[0]]
        print "Test movie: "+movie_name+"....loading movie data"
        movie_clip_featmaps, movie_clip_sentences=test_set.load_movie_slidingclip(movie_name, 16)
        print "sentences: "+ str(len(movie_clip_sentences))
        print "clips: "+ str(len(movie_clip_featmaps))
        sentence_image_mat=np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        sentence_image_reg_mat=np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
        for k in range(len(movie_clip_sentences)):
            sent_vec=movie_clip_sentences[k][1]
            sent_vec=np.reshape(sent_vec,[1,sent_vec.shape[0]])
            for t in range(len(movie_clip_featmaps)):
                featmap = movie_clip_featmaps[t][1]
                visual_clip_name = movie_clip_featmaps[t][0]
                start = float(visual_clip_name.split("_")[1])
                end = float(visual_clip_name.split("_")[2].split("_")[0])
                featmap = np.reshape(featmap, [1, featmap.shape[0]])
                feed_dict = {
                visual_featmap_ph_test: featmap,
                sentence_ph_test: sent_vec
                }
                outputs = sess.run(vs_eval_op,feed_dict=feed_dict)
                sentence_image_mat[k,t] = outputs[0]
                reg_clip_length = (end-start)*(10**outputs[2])
                reg_mid_point = (start+end)/2.0+movie_length*outputs[1]
                reg_end = end+outputs[2]
                reg_start = start+outputs[1]
                
                sentence_image_reg_mat[k,t,0] = reg_start
                sentence_image_reg_mat[k,t,1] = reg_end
        
        iclips = [b[0] for b in movie_clip_featmaps]
        sclips = [b[0] for b in movie_clip_sentences]
        
        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU=IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            print movie_name+" IoU="+str(IoU)+", R@10: "+str(correct_num_10/len(sclips))+"; IoU="+str(IoU)+", R@5: "+str(correct_num_5/len(sclips))+"; IoU="+str(IoU)+", R@1: "+str(correct_num_1/len(sclips))
            all_correct_num_10[k]+=correct_num_10
            all_correct_num_5[k]+=correct_num_5
            all_correct_num_1[k]+=correct_num_1
        all_retrievd+=len(sclips)
    for k in range(len(IoU_thresh)):
        print " IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd)
        test_result_output.write("Step "+str(iter_step)+": IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd)+"\n")