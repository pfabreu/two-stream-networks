import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def cleanupList(in_list):
    out_list = []
    m = in_list.tolist()
    if type(m[0]) is float:
        return m
    for el in m:
        el = el.split('[')[1]
        el = el.split(']')[0]
        print el
        el = float(el)
        out_list.append(el)
    return out_list


def main():
    command_line = False

    if command_line != True:
        # Timestamp of when the model began training
        timestamp = "1805291945"
        set_type = "val"
        stream = "flow"
        filename = "../loss_acc_plots/" + stream + "_" + set_type + "_plot_resnet50_" + timestamp + ".csv"
    else:
        parser = ArgumentParser()
        parser.add_argument("-f", "--file", dest="filename", help="plot file", metavar="FILE")
        filename = parser.parse_args()

    # Plot most recent csv
    my_csv = pd.read_csv(filename, header=None)
    nb_chunks = len(my_csv[0].tolist())
    # print my_csv[0].tolist()
    # print nb_chunks
    nb_epochs = len(my_csv[4].tolist())
    plt.figure(1)
    plt.subplot(211)
    jointacc = plt.plot(range(nb_chunks), cleanupList(my_csv[0]))
    poseacc = plt.plot(range(nb_chunks), cleanupList(my_csv[1]))
    objloss = plt.plot(range(nb_chunks), cleanupList(my_csv[2]))
    humanacc = plt.plot(range(nb_chunks), cleanupList(my_csv[3]))
    plt.title('Accuracy')
    plt.gca().legend(('Average Accuracy', 'Pose Accuracy', 'Object-Human Accuracy', 'Human-Human Accuracy'))

    plt.subplot(212)
    plt.plot(range(nb_epochs), cleanupList(my_csv[4]))
    if set_type == "val":
        plt.plot(range(nb_epochs), cleanupList(my_csv[5]))
        plt.plot(range(nb_epochs), cleanupList(my_csv[6]))
        plt.plot(range(nb_epochs), cleanupList(my_csv[7]))
        plt.gca().legend(('Joint Loss', 'Pose Loss', 'Object-Human Loss', 'Human-Human Loss'))

    plt.title('Loss')
    plt.show()


if __name__ == '__main__':
    main()
