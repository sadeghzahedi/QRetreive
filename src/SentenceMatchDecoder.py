# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
from vocab_utils import Vocab
import namespace_utils

import tensorflow as tf
import SentenceMatchTrainer
from SentenceMatchModelGraph import SentenceMatchModelGraph
from SentenceMatchDataStream import SentenceMatchDataStream

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re


def evaluate(dataStream, valid_graph, sess, outpath=None,
             label_vocab=None, char_vocab=None, word_vocab = None):
    outfile = open (outpath, 'wt')
    for batch_index in range(dataStream.get_num_batch()):
        cur_dev_batch = dataStream.get_batch(batch_index)
        (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch, 
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch, 
                                 sent1_char_length_batch, sent2_char_length_batch,
                                 POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch) = cur_dev_batch
        feed_dict = {
                    valid_graph.get_truth(): label_id_batch, 
                    valid_graph.get_question_lengths(): sent1_length_batch, 
                    valid_graph.get_passage_lengths(): sent2_length_batch, 
                    valid_graph.get_in_question_words(): word_idx_1_batch, 
                    valid_graph.get_in_passage_words(): word_idx_2_batch,
                    valid_graph.get_question_char_lengths (): sent1_char_length_batch,
                    valid_graph.get_passage_char_lengths() : sent2_char_length_batch,
                    valid_graph.get_in_question_chars() : char_matrix_idx_1_batch,
                    valid_graph.get_in_passage_chars (): char_matrix_idx_2_batch
                }
        before_pooling = sess.run([valid_graph.before_pooling_list [0],
          valid_graph.before_pooling_list [1],
          valid_graph.before_pooling_list [2],
          valid_graph.before_pooling_list [3],
          valid_graph.before_pooling_list [4],
          valid_graph.before_pooling_list [5],
          valid_graph.before_pooling_list [6],
          valid_graph.before_pooling_list [7],
          valid_graph.before_pooling_list [8],
          valid_graph.before_pooling_list [9]]
          , feed_dict=feed_dict)
        for i in range (10):
            outfile.write( "SENT1:\t" + str(sent1_batch[i].split ('\t')) + '\n' + "SENT2:\t" + 
              str(sent2_batch [i].split('\t')) + '\n')
            outfile.write(str(before_pooling [i]))
            outfile.write('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default= "../in_path.txt", help='the path to the test file.')
    parser.add_argument('--out_path', type=str, default= "../out_path.txt", help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, default='../data/glove/glove.6B.50d.txt',
                        help='word embedding file for the input file.')
    parser.add_argument('--is_trec', type=bool, default=True, help='prediction or probs')
    parser.add_argument('--index', type=str, default='1', help='prediction or probs') #29 # 'we1-1' #64
    parser.add_argument('--run_id', default='st_b' , help = 'run_id')


    args, unparsed = parser.parse_known_args()

    op = 'wik'
    if args.is_trec == True or args.is_trec == 'True':
        op = 'tre'
    log_dir = '../models' + op
    path_prefix = log_dir + "/SentenceMatch.normal" 

    #model_prefix = args.model_prefix
    in_path = args.in_path
    word_vec_path = args.word_vec_path
    out_json_path = None
    dump_prob_path = None

    # load the configuration file
    print('Loading configurations.')
    FLAGS = namespace_utils.load_namespace(path_prefix + args.run_id + args.index+ ".config.json")
    print(FLAGS)

    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(word_vec_path, fileformat='txt3')
    label_vocab = Vocab(path_prefix + ".label_vocab", fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    print('label_vocab: {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    POS_vocab = None
    NER_vocab = None
    char_vocab = None
    char_vocab = Vocab(path_prefix + ".char_vocab", fileformat='txt2')

    print('Build SentenceMatchDataStream ... ')
    testDataStream = SentenceMatchDataStream(in_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab,
                                                  batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=True,
                                                  max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length,
                                                  is_as=FLAGS.is_answer_selection)

    if FLAGS.wo_char == True: char_vocab = None
    best_path = path_prefix +  args.run_id + args.index + ".best.model"
    print('Decoding on the test set:')
    with tf.Graph().as_default():
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
                    valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab,
                                                          dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                                                          lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim,
                                                          aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False, MP_dim=FLAGS.MP_dim,
                                                          context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num,
                                                          fix_word_vec=FLAGS.fix_word_vec, with_filter_layer=FLAGS.with_filter_layer, with_input_highway=FLAGS.with_highway,
                                                          word_level_MP_dim=FLAGS.word_level_MP_dim,
                                                          with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
                                                          highway_layer_num=FLAGS.highway_layer_num, with_lex_decomposition=FLAGS.with_lex_decomposition,
                                                          lex_decompsition_dim=FLAGS.lex_decompsition_dim,
                                                          with_left_match=(not FLAGS.wo_left_match), with_right_match=(not FLAGS.wo_right_match),
                                                          with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match),
                                                          with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match),
                                                          with_bilinear_att=(FLAGS.attention_type)
                                                          , type1=FLAGS.type1, type2 = FLAGS.type2, type3=FLAGS.type3,
                                                          with_aggregation_attention=not FLAGS.wo_agg_self_att,
                                                          is_answer_selection= FLAGS.is_answer_selection,
                                                          is_shared_attention=FLAGS.is_shared_attention,
                                                          modify_loss=FLAGS.modify_loss, is_aggregation_lstm=FLAGS.is_aggregation_lstm,
                                                          max_window_size=FLAGS.max_window_size
                                                          , prediction_mode=FLAGS.prediction_mode,
                                                          context_lstm_dropout=not FLAGS.wo_lstm_drop_out,
                                                          is_aggregation_siamese=FLAGS.is_aggregation_siamese
                                                          , unstack_cnn=FLAGS.unstack_cnn,with_context_self_attention=FLAGS.with_context_self_attention,
                                                          mean_max=FLAGS.mean_max, clip_attention=FLAGS.clip_attention
                                                          ,with_tanh=FLAGS.tanh, new_list_wise=FLAGS.new_list_wise,
                                                          q_count=1, pos_avg = FLAGS.pos_avg, with_input_embedding=FLAGS.with_input_embedding
                                                          ,with_output_highway = FLAGS.with_output_highway,
                                                          with_matching_layer = FLAGS.with_matching_layer,
                                                          pooling_type=FLAGS.pooling_type, learn_params=FLAGS.learn_params)
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            if 'b_1' in var.name:
              print (var.name)
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        saver.restore(sess, best_path)

        evaluate(dataStream=testDataStream, valid_graph=valid_graph, sess=sess, outpath=args.out_path,
             label_vocab=label_vocab, char_vocab=char_vocab, word_vocab = word_vocab)




