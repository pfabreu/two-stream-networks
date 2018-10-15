#! python2
from utils import *
import numpy as np
import pdb
import copy
import json
import pickle

def my_expnum(a):
    a = '%.0e' %a
    return a[:3] + a[-1]

def flatten_list(t_list):
    if type(t_list) != type([]):
        return [t_list]
    result = []
    for i in  t_list:
        result += flatten_list(i)
    return result

# save and load obj in all format, pkl and np method is supported
def save_obj(obj, path, type='pkl'):
    try:
        if type == 'pkl':
            with open(path + '.pkl', 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        elif type == 'np':
            np.save('%s' %path, obj) 
        else:
            print ('type %s not supported' %type)
    except IOError,e:
        print e


def load_obj(path, type='pkl' ):
    try:
        if type == 'pkl':
            with open(path + '.pkl', 'rb') as f:
                return pickle.load(f)
        elif type == 'np':
            return np.load('%s.npy' %path)
        else:
            print ('type %s not supported' %type)
    except IOError, e:
        print e
        # pdb.set_trace()
        return None

def print_len_dir(path):
    import os
    print len(os.listdir(path))

def write_file(write_list, write_name, mode='wb'):
    #
    #
    from collections import Iterable
    # isinstance('abc', Iterable)
    # 2-level supprted
    with open(write_name, mode) as f:
        if isinstance(write_list, Iterable) and \
         not isinstance(write_list, type('')) and \
          not isinstance(write_list, type(u'')):
            for i in write_list:
                if isinstance(i, Iterable) and \
                 not isinstance(i, type('')) and \
                  not isinstance(i, type(u'')):
                    for t_i in i:
                        f.write(str(t_i) + ' ')
                    f.write('\n')
                else:
                    f.write(str(i) + '\n')
        else:
            f.write(str(write_list) + '\n')

def path_check(path, delete = False, ask = False):
    import os
    if os.path.exists(path):
        def del_cmd():
            del_cmd = os.system('rm -rf %s/*' %path)
        if delete :
            del_cmd()
        if ask:
            print '%s exists before training, delete and continue? y/n' %path       
            if raw_input() == 'y':
                del_cmd()
        else:
            return None
    else:
        os.makedirs(path)
        #os.system('mkdir -p %s' %path)    

check_path = path_check

def vid_id(d):
    return d['video'].split('_')[1]

def int2name(t):
    t = int(t)
    if t < 10:
        return '0000%d'%t
    elif t < 100:
        return '000%d'%t
    elif t < 1000:
        return '00%d'%t
    elif t < 10000:
        return '0%d'%t
    else:
        print 'such a big t?'
        pdb.set_trace()

def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def iou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union
  
def rank(pred, gt):
      return pred.index(tuple(gt)) + 1
  
def eval_predictions(segments, data, print_result = True):
    
    average_ranks = []
    average_iou = []
    
    for s, d in zip(segments, data):
      pred = s[0]
      ious = [iou(pred, t) for t in d['times']]
      average_iou.append(np.mean(np.sort(ious)[-3:]))
      ranks = [rank(s, t) for t in d['times']]
      average_ranks.append(np.mean(np.sort(ranks)[:3]))
      
    rank1 = np.sum(np.array(average_ranks) <= 1)/float(len(average_ranks))
    rank5 = np.sum(np.array(average_ranks) <= 5)/float(len(average_ranks))
    miou = np.mean(average_iou)
    if print_result:
        print "Average rank@1: %f" %rank1
        print "Average rank@5: %f" %rank5
        print "Average iou: %f" %miou
    return rank1, rank5, miou

def train_prior():
    #start = time.time()
    train_data = read_json('data/train_data.json')
    #train_data = read_json('data/test_data.json')#######
    #val_data = read_json('data/val_data.json')
    prior = {}
    for d in train_data:
        times = [t for t in d['times']]
        for time in times:
            time = tuple(time)
            if time not in prior.keys():
                prior[time] = 0
            prior[time] += 1
    prior[(0,5)] = 1
    return prior
    
def mix_prior(dist, prior_dict, w=2.0):
    mode = [(start, end) for start in range(6) for end in range(start, 6)]
    prob_dict = copy.deepcopy(prior_dict)
    
    post_prob = np.exp(1 - dist/dist.max()/w)
    
    for j in range(21):
        prob_dict[mode[j]] *= post_prob[j]
    return prob_dict
    
def eval_dist(dist, test_caps):
    mode = [(start, end) for start in range(6) for end in range(start, 6)]
    l = dist.shape[0]
    rgb_prop = range(l)
    for i in range(l):
        i_dict = {}
        for j in range(21):
            i_dict[mode[j]] = dist[i][j]
        rgb_prop[i] = sorted(i_dict, key=i_dict.get)
        
    return eval_predictions(rgb_prop, test_caps)
    
def word2vec(filename = '/DB/rhome/bhuang/Link to DATA3 workspace/video_cap_retrieval/glove.6B/glove.6B.300d.txt', dim_emb = 300):
    word_emb = {}
    with open(filename, 'rb') as f:
        for line in f:
            line = line.split()
            word_emb[line[0]] = np.array(line[1:], dtype = 'float')      
    return word_emb
    
def word2vec_idx(filename = '/DB/rhome/bhuang/Link to DATA3 workspace/video_cap_retrieval/glove.6B/glove.6B.300d.txt', dim_emb = 300):
    word_vec = []
    ixtoword = []
    wordtoix = {}
    
    with open(filename, 'rb') as f:
        for i, line in enumerate(f):
            line = line.split()
            cur_word = line[0]
            cur_vec = np.array(line[1:], dtype = 'float32')
            ixtoword.append(cur_word)
            wordtoix[cur_word] = i
            word_vec.append(cur_vec)
            word_vec = np.array(word2vec)            
            
    return word_vec, ixtoword, wordtoix
    
def most_common(time_list):
    a = {}
    for i in time_list:
        i = tuple(i)
        if i in a:
            a[i] += 1
        else:
            a[i] = 1
    a = sorted(a, key=a.get, reverse = True)
    return a[0]
