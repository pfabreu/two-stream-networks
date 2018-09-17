import pickle

lossList = []
NHU = [32, 64, 128, 256, 512]
for i in NHU:
    with open('context_results/timeWindow_3/results_3Neighbors_window3_' + str(i) + 'HU', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
        valloss = content['val_loss']
        lossList.append(min(valloss))
