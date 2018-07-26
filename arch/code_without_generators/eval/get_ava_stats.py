import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-white')
# Show histograms of class distribution for each split


def coincidence_matrix(file):
    cm = np.load(file)
    cm = cm.astype(int)
    print(cm.shape)
    return cm


# Show coheremce matrices one at a time
show_individual_mats = False
if show_individual_mats:
    cm = coincidence_matrix("TrainLabelsConMatrix.npy")
    g = sns.heatmap(cm, xticklabels=True, yticklabels=True, annot=True, fmt="d", cmap=sns.cubehelix_palette(8, start=.5, rot=-.75), linewidths=0.5, linecolor='gray', cbar=True)
    plt.title("Co-incidence Matrix (Train)")

    i = 0
    for ytick_label, xtick_label in zip(g.axes.get_yticklabels(), g.axes.get_xticklabels()):
        if i <= 10:
            ytick_label.set_color("r")
            xtick_label.set_color("r")

        elif i <= 22:
            ytick_label.set_color("b")
            xtick_label.set_color("b")
        else:
            ytick_label.set_color("g")
            xtick_label.set_color("g")
        i += 1
    plt.show()

    cm = coincidence_matrix("ValidationLabelsConMatrix.npy")
    g = sns.heatmap(cm, xticklabels=True, yticklabels=True, annot=True, fmt="d", cmap=sns.cubehelix_palette(8, start=.5, rot=-.75), linewidths=0.5, linecolor='gray', cbar=True)
    plt.title("Co-incidence Matrix (Validation)")

    i = 0
    for ytick_label, xtick_label in zip(g.axes.get_yticklabels(), g.axes.get_xticklabels()):
        if i <= 10:
            ytick_label.set_color("r")
            xtick_label.set_color("r")

        elif i <= 22:
            ytick_label.set_color("b")
            xtick_label.set_color("b")
        else:
            ytick_label.set_color("g")
            xtick_label.set_color("g")
        i += 1
    plt.show()

    cm = coincidence_matrix("TestLabelsConMatrix.npy")
    g = sns.heatmap(cm, xticklabels=True, yticklabels=True, annot=True, fmt="d", cmap=sns.cubehelix_palette(8, start=.5, rot=-.75), linewidths=0.5, linecolor='gray', cbar=True)
    plt.title("Co-incidence Matrix (Test)")

    i = 0
    for ytick_label, xtick_label in zip(g.axes.get_yticklabels(), g.axes.get_xticklabels()):
        if i <= 10:
            ytick_label.set_color("r")
            xtick_label.set_color("r")

        elif i <= 22:
            ytick_label.set_color("b")
            xtick_label.set_color("b")
        else:
            ytick_label.set_color("g")
            xtick_label.set_color("g")
        i += 1
    plt.show()

# Show Co-incidence matrices together
for i in range(1, 7):
    plt.subplot(2, 3, i)
    if i < 4:
        if i == 1:
            cm = coincidence_matrix("TrainLabelsConMatrix.npy")
            plt.title("Co-incidence Matrix (Train)")
        elif i == 2:
            cm = coincidence_matrix("ValidationLabelsConMatrix.npy")
            plt.title("Co-incidence Matrix (Validation)")
        elif i == 3:
            cm = coincidence_matrix("TestLabelsConMatrix.npy")
            plt.title("Co-incidence Matrix (Test)")
        g = sns.heatmap(cm, xticklabels=True, yticklabels=True, annot=False, fmt="d", cmap=sns.cubehelix_palette(8, start=.5, rot=-.75), linewidths=0.5, linecolor='gray', cbar=True)
        t = 0
        for ytick_label, xtick_label in zip(g.axes.get_yticklabels(), g.axes.get_xticklabels()):
            if t <= 10:
                ytick_label.set_color("r")
                xtick_label.set_color("r")

            elif t <= 22:
                ytick_label.set_color("b")
                xtick_label.set_color("b")
            else:
                ytick_label.set_color("g")
                xtick_label.set_color("g")
            t += 1
    else:
        if i == 4:
            classes = []
            with open("../../../data/AVA/files/AVA_Train_Custom_Corrected.csv") as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    classes.append(int(row[6]))
            ax = sns.distplot(classes, color="y", bins=range(1, 31))
            plt.xlim(0, 30)
            plt.ylim(0, 0.2)
            #plt.yticks(ax.get_yticks(), ax.get_yticks() * 100.0)
            plt.title("Class distribution histogram (Train)(%)")
            plt.grid(True)
            t = 0
            plt.xticks(range(0, 31))
            for xtick_label in ax.axes.get_xticklabels():
                if t <= 10:
                    xtick_label.set_color("r")
                elif t <= 22:
                    xtick_label.set_color("b")
                else:
                    xtick_label.set_color("g")
                t += 1
            print("done histo")
        elif i == 5:
            classes = []
            with open("../../../data/AVA/files/AVA_Val_Custom_Corrected.csv") as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    classes.append(int(row[6]))
            ax = sns.distplot(classes, color="m")
            plt.xlim(0, 30)
            plt.ylim(0, 0.2)
            #plt.yticks(ax.get_yticks(), ax.get_yticks() * 100.0)
            plt.title("Class distribution histogram (Val)(%)")
            plt.grid(True)
            t = 0
            plt.xticks(range(0, 31))
            for xtick_label in ax.axes.get_xticklabels():
                if t <= 10:
                    xtick_label.set_color("r")
                elif t <= 22:
                    xtick_label.set_color("b")
                else:
                    xtick_label.set_color("g")
                t += 1
            print("done histo")
        elif i == 6:
            classes = []
            with open("../../../data/AVA/files/AVA_Test_Custom_Corrected.csv") as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    classes.append(int(row[6]))
            ax = sns.distplot(classes, color="b")
            plt.xlim(0, 30)
            plt.ylim(0, 0.2)
            #plt.yticks(ax.get_yticks(), ax.get_yticks() * 100.0)
            plt.title("Class distribution histogram (Test)(%)")
            plt.grid(True)
            t = 0
            plt.xticks(range(0, 31))
            for xtick_label in ax.axes.get_xticklabels():
                if t <= 10:
                    xtick_label.set_color("r")
                elif t <= 22:
                    xtick_label.set_color("b")
                else:
                    xtick_label.set_color("g")
                t += 1
            print("done histo")
            print("done histo")

plt.show()
