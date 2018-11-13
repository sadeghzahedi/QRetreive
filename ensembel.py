
import numpy as np
import sys


eps = 1e-8


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default='hinge100', help='do we have cuda visible devices?')
parser.add_argument('--id_list', default= [238, 106], type = list, help = 'is trec?')
parser.add_argument('--is_trec', default= 'True' , help = 'is trec?')
parser.add_argument('--is_server', default= False , type= bool, help = 'is trec?') # irad dare ba server code baiad eslah she
parser.add_argument('--use_csv_res', default= True , type= bool, help = 'is trec?')
parser.add_argument('--top_k', default= 5 , type= int, help = 'is trec?')


FLAGS, unparsed = parser.parse_known_args()

import os


w =  'wik'
if FLAGS.is_trec == 'True':
    w = 'tre'
w = w + FLAGS.run_id
csv_file_path = w + '.csv'
w = 'result/' + w + '.'
import re
if FLAGS.use_csv_res == True:
    csv_file = open(csv_file_path)
    FLAGS.id_list = []
    FLAGS.top_k += 1
    i = FLAGS.top_k + 1
    for line in csv_file:
        i -= 1
        if i == FLAGS.top_k:
            continue
        if i == 0:
            break
        line = re.split(',', line)
        line = line[0]
        line = re.split('\.', line)
        FLAGS.id_list.append(int (line[len(line) - 1]))


file_name_list = []
for x in FLAGS.id_list:
    y = w + str(x) + 'S'
    file_name_list.append(y)

in_files = []

for x in file_name_list:
    in_files.append(open(x))

import re

list_of_whole_list = []



def MAP_MRR(logit, gold, candidate_answer_length, sent1s, sent2s):
    c_1_j = 0.0 #map
    c_2_j = 0.0 #mrr
    visited = 0
    output_sentences = []
    for i in range(len(candidate_answer_length)):
        prob = logit[visited: visited + candidate_answer_length[i]]
        label = gold[visited: visited + candidate_answer_length[i]]
        question = sent1s[visited: visited + candidate_answer_length[i]]
        answers = sent2s[visited: visited + candidate_answer_length[i]]
        visited += candidate_answer_length[i]
        rank_index = np.argsort(prob).tolist()
        rank_index = list(reversed(rank_index))
        score = 0.0
        count = 0.0
        for i in range(1, len(prob) + 1):
            if label[rank_index[i - 1]] > eps:
                count += 1
                score += count / i
        for i in range(1, len(prob) + 1):
            if label[rank_index[i - 1]] > eps:
                c_2_j += 1 / float(i)
                break
        c_1_j += score / count
        output_sentences.append((question[0]) + "\n")
        for jj in range(len(answers)):
            output_sentences.append(str(label[rank_index[jj]]) + " " + str(prob[rank_index[jj]]) + "- " +
                                   answers[rank_index[jj]] + "\n")
        output_sentences.append("AP: {} \n\n".format(score / count))

    my_map = c_1_j/len(candidate_answer_length)
    my_mrr = c_2_j/len(candidate_answer_length)
    return (my_map,my_mrr, output_sentences)





def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

first_time = True
answer_len = []
#all_touples_for_sort = []
#in_files = [open('result/wiklast_run1-1.106S'), open('result/wikglove5-0.238S')]
for f in in_files:
    whole_list = []
    flag = True
    for line in f:
        if flag == True:
            question = line
            flag = False
            touple_list = []
        else:
            if len (line.strip()) == 0: # '\n'
                #if (len (touple_list) == 0): continue
                flag = True
                if first_time == True:
                    answer_len.append(len(touple_list))
                # sort every thing based on answers
                touple_list = sorted(touple_list, key=lambda tup: tup[1])

                pred_list = []
                for k in touple_list:
                    pred_list.append(k[3])
                pred_list = np.array(pred_list)
                pred_list = softmax(pred_list)
                tmp_tuple_list = []
                for k in range(len(touple_list)):
                    tmp_tuple_list.append((touple_list[k][0], touple_list[k][1],
                                           touple_list[k][2], pred_list[k]))
                touple_list = tmp_tuple_list
                whole_list.extend(touple_list)
                continue
            else:
                line = re.split (' ', line)
                if line [0] == "AP:":
                    continue
                else:
                    print (line [0])
                    label = float (line[0])
                    y = (line[1])[0:len(line[1])-1]
                    pred = float (y)
                    st = ''
                    for zz in range(2, len (line)):
                        st = st + line [zz] + ' '
                    answer = st
                    touple_list.append((question, answer, label, pred))

    list_of_whole_list.append(whole_list)
    pairs_cnt = len (whole_list)
    first_time = False
    f.close ()

sent1s = []
sent2s = []
logits = []
gold = []

for i in range(pairs_cnt):
    pred = 0
    for j in range(len (list_of_whole_list)):
        pred += list_of_whole_list [j][i][3]
    pred = pred / len (list_of_whole_list)
    question = list_of_whole_list [0][i][0]
    answer = list_of_whole_list [0][i][1]
    label = list_of_whole_list [0][i][2]
    sent1s.append(question)
    sent2s.append(answer)
    logits.append(pred)
    gold.append(label)

(my_map , my_mrr, output_sentences) = MAP_MRR(logits,gold,answer_len,sent1s, sent2s)

print (my_map, my_mrr)
st_cuda = ''

if FLAGS.is_server == True:
    st_cuda = str(os.environ['CUDA_VISIBLE_DEVICES']) + '.'
if FLAGS.is_trec == 'True':
    ssst = 'tre' + FLAGS.run_id
else:
    ssst = 'wik' + FLAGS.run_id

output_sentence_file = open('../result/' + ssst + '.' + st_cuda + str (len (FLAGS.id_list)) + "E", 'wt')


for zj in output_sentences:
    if sys.version_info[0] < 3:
        output_sentence_file.write(zj)#.encode('utf-8'))
    else:
        output_sentence_file.write(zj)

output_sentence_file.close()




