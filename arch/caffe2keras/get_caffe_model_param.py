import caffe
import pickle
import json
import numpy as np
from utils import *

stream = 'flow'
model_path = '../models/kinetics_caffe/inception_v3_kinetics_' + stream + '_pretrained/inception_v3_' + stream + '_deploy.prototxt'
param_path = '../models/kinetics_caffe/inception_v3_kinetics_' + stream + '_pretrained/inception_v3_' + stream + '_kinetics.caffemodel'
caffe.set_mode_cpu
net = caffe.Net(model_path, param_path, caffe.TEST)

save_name = '../models/kinetics_keras/tsn_%s_params' % stream
order_name = '%s_names' % save_name
# bn_order_name = 'bn_%s'%order_name
dict_name = net.params
# bn_names_order = [t for t in net.params.keys() if 'batchnorm' in t]
param_dict = {}
for key, value in net.params.items():
    param_dict[key] = [t.data for t in value]

output = open(save_name + '.pkl', 'wb')
pickle.dump(param_dict, output)
output.close()

output = open(order_name + '.pkl', 'wb')
pickle.dump(net.params.keys(), output)
output.close()
# save_obj(bn_names_order, bn_order_name)
