# QRetreive


show sorted results:

run: python show_res.py --run_id 'run_id0'

result: a csv file is created in the current repository which contains all results of run_id.

results are sorted based on valid_map. to sort based on test_map or max_test_map modify line 284 in show_res.py 


Ensemble learning:

Run this after running show_res.py because this file, uses the csv file created by show_res to determine the top results and ensemble them.

parameters:

--top_k : ensembel top k result (k is integer)

run: python ensemble.py --run_id 'run_id0'

result file: 'run_id-0E' in result directory



parameters for fixing pretrain:

--learn_params in src directory.
set it to False if you want to fix pretrain parameters. in this case all parameters considered fixed expect finall highway layer.




Clone this repository into your desired directory with command: git clone https://github.com/sadeghzahedi/QRetreive


data:

data/wikiqa/train.txt test.txt dev.txt  ->  pretrain data (Qura)

data/trecqa/train.txt test.txt dev.txt  -> train data (Semeval)

srcPreTrain directory -> original code(pretrain)

src -> main code (train)


running:

run srcPreTrain/SentenceMatchTrainer.py to generate pretraine checkpoint in modelswik directory. (dont change run_id, because it will change the pretrain path)

run src/SentenceMatchTrainer.py with your desired run_id to see your state of the art result :)


some points:

your parameters must be the same in original and main code (you have to change them into your desired parameters, word embbeding is 50d for faster testing on my PC):

parser.add_argument('--word_vec_path', type=str, default='../data/glove/glove.6B.50d.txt', help='Path the to pre-trained word

parser.add_argument('--MP_dim', type=int, default=10, help='Number of perspectives for matching vectors.')

parser.add_argument('--context_lstm_dim', type=int, default=10, help='Number of dimension for context representation layer.')

parser.add_argument('--aggregation_lstm_dim', type=int, default=10, help='Number of dimension for aggregation layer.')


pretrain Parameters(in src directory):

parser.add_argument('--has_pretrain',default=True, help='is trec or wiki?')

parser.add_argument('--pretrain_path',default='../modelswik/SentenceMatch.normalst_b1.best.model', help='is trec or wiki?')


VERY IMPORTANT: AFTER EVERY CHANGE TO CODE, PUSH THE CODE INTO THE REPOSITORY as follows:

git add src

git add srcPreTrain

git commit -m 'new'

git push


after my changes to repositry you can easily have them with command: git pull








  





