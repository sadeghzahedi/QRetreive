import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import argparse




'''
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.04, help='Dropout ratio.')
    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--char_lstm_dim', type=int, default=20, help='Number of dimension for character-composed embeddings.')
    parser.add_argument('--context_lstm_dim', type=int, default=10, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=10, help='Number of dimension for aggregation layer.')
    parser.add_argument('--MP_dim', type=int, default=10, help='Number of perspectives for matching vectors.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--aggregation_layer_num', type=int, default=2, help='Number of LSTM layers for aggregation layer.')
    parser.add_argument('--context_layer_num', type=int, default=2, help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--highway_layer_num', type=int, default=0, help='Number of highway layers.')
    parser.add_argument('--suffix', type=str, default='normal', required=False, help='Suffix of the model name.')
    parser.add_argument('--with_highway', default=False, help='Utilize highway layers.', action='store_true')
    parser.add_argument('--with_match_highway', default=False, help='Utilize highway layers for matching layer.', action='store_true')
    parser.add_argument('--with_aggregation_highway', default=False, help='Utilize highway layers for aggregation layer.', action='store_true')
    parser.add_argument('--wo_char', default=True, help='Without character-composed embeddings.', action='store_true')
    parser.add_argument('--wo_bilinear_att', default=False, help='Without bilinear attention', action='store_true')
    parser.add_argument('--type1', default='w_mul', help='similrty function 1', action='store_true')
    parser.add_argument('--type2', default= 'w_mul' , help='similrty function 2', action='store_true')
    parser.add_argument('--type3', default= None , help='similrty function 3', action='store_true')
    parser.add_argument('--wo_lstm_drop_out', default=  True , help='with out context lstm drop out', action='store_true')
    parser.add_argument('--wo_agg_self_att', default= False , help='with out aggregation lstm self attention', action='store_true')
    parser.add_argument('--is_shared_attention', default= False , help='are matching attention values shared or not', action='store_true')
    parser.add_argument('--modify_loss', type=float, default=0.1, help='a parameter used for loss.')
    parser.add_argument('--is_aggregation_lstm', default=True, help = 'is aggregation lstm or aggregation cnn' )
    parser.add_argument('--max_window_size', type=int, default=2, help = '[1..max_window_size] convolution')
    parser.add_argument('--is_aggregation_siamese', default=True, help = 'are aggregation wieghts on both sides shared or not' )


'''
#df = pd.DataFrame(columns=['id', 'agg_type' ,'unstack_cnn' ,'context_lstm_dim', 'context_layer_num', 'context_lstm_drop_out', 'agg_(lstm or cnn)_dim' ,'agg_layer_num', 'agg_self_att', 'window_size' ,'mp', 'drop_out', 'type1', 'type2', 'type3', 'lr', 'char(em-ls)', 'shared_attention','shared_aggregation' ,'input_highway', 'match_highway', 'agg_highway' ,'batch_size', 'converged_epoch','loss_type','modify_loss' ,'train_map', 'valid_map', 'test_map', 'test_mrr', 'max_test_map'])

df = pd.DataFrame(columns=['id','context_lstm_dim', 'context_layer_num', 'agg_lstm_dim' ,'agg_layer_num','mp', 'drop_out', 'type', 'lr', 'char(em-ls)' ,'shared_aggregation'  ,'batch_size', 'converged_epoch','loss_type' ,'q_count','pooling_type' , 'train_map', 'valid_map', 'test_map', 'test_mrr', 'max_test_map'])

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default='hinge100', help='do we have cuda visible devices?')
parser.add_argument('--is_trec', default= False, help = 'is trec?')
parser.add_argument ('--just_list', default='False')
parser.add_argument ('--just_hinge', default='False')
parser.add_argument('--list_modified', default='False')
parser.add_argument('--min_value', default=0, type=int)
parser.add_argument('--max_value', default=10000, type=int)

FLAGS, unparsed = parser.parse_known_args()
#print (FLAGS, unparsed)
post_path = ''
w = 'wik'
if FLAGS.is_trec == True or FLAGS.is_trec == 'True':
    w = 'tre'
outcsv = w + FLAGS.run_id
w = outcsv + '.'
w = list (w)
#print (w)
mypath = 'result/' #'result/' #+ st + FLAGS.run_id + "."
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
test_map_list = np.zeros(31)
test_map_cnt = np.zeros(31)
loss_list = np.zeros(31)
dev_map_list = np.zeros(31)
dev_map_cnt = np.zeros(31)


for st in onlyfiles:
#    print (st)
    if st [len (st)-1] == 'S' or  st [len (st)-1] == 'T':
        continue
    st_t = st.split ('.')
    st_t = int (st_t [len (st_t)-1])
    if st_t > FLAGS.max_value or st_t < FLAGS.min_value:
        continue
    rf = open (mypath + st)
    #if (len (st) >=2 and st [1] == '.') : continue
    if (len(st) <= 3) : continue
#    print (st)
    if st[3] == 'l':
        rty = 1
    y = list (st)
    if (y[len(y)-1] == 'S' or y[len(y)-1] == 'v'): #S , .csv
        continue
    fflag = False
    if len(w) > len (y): continue
    for j in range(len(w)):
        if y[j] != w[j]:
            fflag = True
    if fflag == True:
        continue
    #if (st[0] != '3' or st[1] != '8'): continue
    #if (st == '7.19'):
    #    sk = 2

    inf = rf.readline()
    l = inf [10:].split(',')
    if (len (l) <= 2) : continue
    dt = {}
    dt ['unstack_cnn'] = None
    flag1 = True
    for x in l:
        r = x.split('=')
        if (len (r) <=1): break;
        ssind = 0
        if r [0][0] == ' ': ssind = 1
        # if r [0][ssind:] == 'batch_size':
        #     flag1 = False
        # if r [0][ssind:] == 'attention_type' and r[1] != "'dot_product'":
        #     flag1 = False
        # if r [0][ssind:] == 'is_aggregation_lstm' and r[1] == 'False':
        #     flag1 = False
        #print (r[0])
        #print ('T' + r[1])
        dt [r [0][ssind:]] = r[1]
    if flag1 == False:
        continue
    #print (dt)
    ch_inf = None
    #print (st)
    if dt['wo_char'] == 'False':
        ch_inf = (int(dt['char_emb_dim']), int(dt['char_lstm_dim']))
    with_highway = None
    with_aggregation_highway = None
    with_match_highway = None
    if dt['highway_layer_num'] == '1':
       with_highway, with_aggregation_highway, with_match_highway = (dt['with_highway'],
                                                                      dt['with_aggregation_highway'],
                                                                    dt['with_match_highway'])
    #agg_type = 'CNN'
    if dt['is_aggregation_lstm'] == 'True':
        agg_type = 'LSTM'
        max_window_size = None
        agg_layer_num = dt['aggregation_layer_num']
        if dt ['wo_agg_self_att'] == 'True':
            agg_self_att = False
        else:
            agg_self_att = True
    else:
        agg_type = 'CNN'
        max_window_size = dt['max_window_size']
        agg_layer_num = None
        agg_self_att = None

    if dt ['type2'] == 'None':
        dt ['type2'] = dt['type3']
        dt ['type3'] = 'None'
    if dt ['wo_lstm_drop_out'] == 'True':
        context_lstm_drop_out = False
    else:
        context_lstm_drop_out = True

    best_dev_map = 0.0
    tr, dv, te_map, te_mrr = (0.0,0.0,0.0,0.0)
    rf.readline()
    ind = 0
    c_epoch = 0
    i_epoch = 1
    flag_dv = False
    max_test_map = 0
    dv_cnt = 0
    jList = []
    for line in rf:
        if len (line) <= 2: continue
        if ind == 0:
            if line[0] != 't':
                ind += 1
                ls = line.split()
                loss_list[i_epoch] += float(ls [4])
                continue
            else: #train
                kk = line.split()
                if (len(kk) <= 2): break
                tr = float((kk[2])[1:len(kk[2])-2])
                break
        elif ind == 1: #dev
            ind += 1
            kk = line.split()
            if (len (kk) <= 2): break
            dv = float((kk[2])[1:len(kk[2])-2])
            #dv = float((kk[4])[1:len(kk[4]) - 1])
            #dv_cnt += 1
            if dv_cnt %2 == 1:
                continue
            if dv_cnt > 20:
                continue
            dev_map_list[i_epoch] += dv
            if dv < 0.001:
                print ('fuck1')
            if (dv > best_dev_map):
                best_dev_map = dv
                flag_dv = True
            else:
                flag_dv = False
        elif ind == 2:
            ind = 0
            jj = line.split()
            if len(jj) <= 4: break
            #print (jj [2])
            tmp_map = float((jj[2])[1:len(jj[2])-2])
            #tmp_map = float((jj[4])[1:len(jj[4]) - 1])
            jList.append(tmp_map)
            if dv_cnt %2 == 1:
                continue
            if dv_cnt > 20:
                continue
            if tmp_map < 0.001:
                print ('fuck2')
            test_map_list[i_epoch] += tmp_map
            if tmp_map > max_test_map:
                max_test_map = tmp_map
            if flag_dv == True:
                flag_dv = False
                te_map = float((jj[2])[1:len(jj[2])-2])
                te_mrr = float((jj[4])[1:len(jj[4])-1])
                c_epoch = int(i_epoch)
            i_epoch += 1
    dv = best_dev_map
    #['id', 'clstm', 'alstm', 'mp', 'drop', 'type1', 'lr', 'ch(em-ls)', 'bs', 'c_epoch', 'tr', 'dev', 'test']
    # df = pd.DataFrame(
    #     columns=['id', 'agg_type', 'context_lstm_dim', 'context_layer_num', 'context_lstm_drop_out', 'agg_lstm_dim',
    #              'agg_layer_num', 'agg_self_att', 'window_size', 'mp', 'drop_out', 'type1', 'type2', 'type3', 'lr',
    #              'char(em-ls)', 'shared_attention', 'shared_aggregation', 'input_highway', 'match_highway',
    #              'agg_highway', 'batch_size', 'converged_epoch', 'loss_type', 'modify_loss', 'train_map', 'valid_map',
    #              'test_map', 'test_mrr'])

    # df = df.append({'id':st ,'context_self':dt ['with_context_self_attention'] , 'agg_type': agg_type, 'att_type':dt['attention_type'] ,'unstack_cnn': dt['unstack_cnn'], 'context_lstm_dim':dt['context_lstm_dim'],'context_layer_num':dt['context_layer_num'], 'context_lstm_drop_out':context_lstm_drop_out,
    #                 'agg_(lstm or cnn)_dim':dt['aggregation_lstm_dim'], 'agg_layer_num':agg_layer_num, 'agg_self_att':agg_self_att, 'window_size' : max_window_size,
    #            'mp':dt['MP_dim'], 'drop_out':dt['dropout_rate'], 'type1':dt['type1'], 'type2':dt['type2'], 'type3':dt['type3'], 'lr':dt['learning_rate'],
    #            'char(em-ls)': ch_inf, 'shared_attention': dt['is_shared_attention'],'shared_aggregation':dt['is_aggregation_siamese'],'input_highway':with_highway,
    #                 'match_highway': with_match_highway,'agg_highway':with_aggregation_highway ,'batch_size' : dt['batch_size'], 'converged_epoch' : c_epoch,
    #                 'loss_type' : dt ['prediction_mode'],'modify_loss':dt['modify_loss'] ,'train_map':tr, 'valid_map':dv,
    #                 'test_map': te_map,  'test_mrr': te_mrr, 'max_test_map':max_test_map}, ignore_index=True)

    if i_epoch <=10:
        print (i_epoch)
        print ('Xfunck')
    else:
        jList = [float('%.3f' % elem) for elem in jList]
        #print (jList)
    pred_mode = dt ['prediction_mode']
    is_list_wise = False
    if pred_mode == "'list_wise'":
        is_list_wise = True

    if FLAGS.just_list == 'True':
        post_path = 'list'
        if is_list_wise == False: continue
        if FLAGS.list_modified == 'True':
            post_path += 'm'
            if dt ['modify_loss'] == '0':
                continue
        else:
            if dt['modify_loss'] != '0':
                continue
    if FLAGS.just_hinge == 'True':
        post_path = 'hinge'
        if is_list_wise == True: continue



    #if not(dt['context_lstm_dim'] == '150' and dt ['aggregation_lstm_dim'] == '70' and dt['dropout_rate'] == '0.2'):
    #    continue

    st = st.split ('.')
    st = int (st [len (st)-1])

    df = df.append({'id':st  , 'context_lstm_dim':dt['context_lstm_dim'],'context_layer_num':dt['context_layer_num'],
                    'agg_lstm_dim':dt['aggregation_lstm_dim'], 'agg_layer_num':agg_layer_num,
               'mp':dt['MP_dim'], 'drop_out':dt['dropout_rate'], 'type':dt['type1'], 'lr':dt['learning_rate'],
               'char(em-ls)': ch_inf,'shared_aggregation':dt['is_aggregation_siamese'],
                     'batch_size' : dt['batch_size'], 'converged_epoch' : c_epoch,
                    'loss_type' : dt ['prediction_mode'] ,'q_count':dt['question_count_per_batch'],
                    pooling_type = dt ['pooling_type']
                    'train_map':tr, 'valid_map':dv,
                    'test_map': te_map,  'test_mrr': te_mrr, 'max_test_map':max_test_map}, ignore_index=True)
df = df.sort_index(by='valid_map', ascending=False)
print ("mean-test, ", df['max_test_map'].mean(), "mean dev", df['valid_map'].mean()
, "mean mrr", df ['test_mrr'].mean())
dev_map_list = np.divide(dev_map_list, len(df))
test_map_list = np.divide(test_map_list, len(df))
loss_list = np.divide(loss_list, len(df))
dev_ = []
test_ = []
loss_ = []
for i in range(len(dev_map_list)):
    x = dev_map_list[i]
    if x > 0.0001:
        dev_.append (x)

for i in range(len(test_map_list)):
    x = test_map_list[i]
    if x > 0.0001:
        test_.append(x)

for i in range(len(loss_list)):
    x = loss_list[i]
    if x > 0.0001:
        loss_.append(x/1159.0)

dev_ = [ float('%.3f' % elem) for elem in dev_ ]
test_ = [ float('%.3f' % elem) for elem in test_ ]
loss_ = [ float('%.3f' % elem) for elem in loss_ ]
print (len (df))
print (dev_)
print (test_)
print (loss_)

outcsv = outcsv + post_path
df.to_csv(outcsv + '.csv', index=False)

import os, sys, subprocess

def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener ="open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])
#open_file(outcsv + '.csv')




