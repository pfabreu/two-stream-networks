import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-white')
# Show histograms of class distribution for each split

# Show coincidence matrices one at a time
for i in range(1, 7):
    plt.subplot(2, 3, i)

    if i < 4:
        if i == 1:
            types = [0, 0, 0]
            names = ["pose", "obj", "human"]
            with open("files/AVA_Train_Custom_Corrected.csv") as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    tp = int(row[6])
                    if tp <= 10:
                        types[0] += 1
                    elif tp <= 22:
                        types[1] += 1
                    else:
                        types[2] += 1
            types = [100.0 * float(i) / sum(types) for i in types]
            ax = sns.barplot(x=names, y=types, palette=['red', 'green', 'blue'])
            for p in ax.patches:
                ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='gray', rotation=0, xytext=(0, 4),
                            textcoords='offset points')
            #plt.yticks(ax.get_yticks(), ax.get_yticks() * 100.0)
            plt.title("Type (Train)(%)")
            plt.grid(True)
            t = 0
            plt.xticks(range(0, 3))
            for xtick_label in ax.axes.get_xticklabels():
                if t == 0:
                    xtick_label.set_color("r")
                elif t == 1:
                    xtick_label.set_color("g")
                else:
                    xtick_label.set_color("b")
                t += 1
        if i == 2:
            types = [0, 0, 0]
            names = ["pose", "obj", "human"]
            with open("files/AVA_Val_Custom_Corrected.csv") as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    tp = int(row[6])
                    if tp <= 10:
                        types[0] += 1
                    elif tp <= 22:
                        types[1] += 1
                    else:
                        types[2] += 1
            types = [100.0 * float(i) / sum(types) for i in types]
            ax = sns.barplot(x=names, y=types, palette=['red', 'green', 'blue'])
            for p in ax.patches:
                ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='gray', rotation=0, xytext=(0, 4),
                            textcoords='offset points')
            #plt.yticks(ax.get_yticks(), ax.get_yticks() * 100.0)
            plt.title("Type (Val)(%)")
            plt.grid(True)
            t = 0
            plt.xticks(range(0, 3))
            for xtick_label in ax.axes.get_xticklabels():
                if t == 0:
                    xtick_label.set_color("r")
                elif t == 1:
                    xtick_label.set_color("g")
                else:
                    xtick_label.set_color("b")
                t += 1
        if i == 3:
            types = [0, 0, 0]
            names = ["pose", "obj", "human"]
            with open("files/AVA_Test_Custom_Corrected.csv") as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    tp = int(row[6])
                    if tp <= 10:
                        types[0] += 1
                    elif tp <= 22:
                        types[1] += 1
                    else:
                        types[2] += 1
            types = [100.0 * float(i) / sum(types) for i in types]
            ax = sns.barplot(x=names, y=types, palette=['red', 'green', 'blue'])
            for p in ax.patches:
                ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='gray', rotation=0, xytext=(0, 4),
                            textcoords='offset points')
            #plt.yticks(ax.get_yticks(), ax.get_yticks() * 100.0)
            plt.title("Type (Test)(%)")
            plt.grid(True)
            t = 0
            plt.xticks(range(0, 3))
            for xtick_label in ax.axes.get_xticklabels():
                if t == 0:
                    xtick_label.set_color("r")
                elif t == 1:
                    xtick_label.set_color("g")
                else:
                    xtick_label.set_color("b")
                t += 1
    else:
        if i == 4:
            classes = []
            with open("files/AVA_Train_Custom_Corrected.csv") as csvDataFile:
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
        elif i == 5:
            classes = []
            with open("files/AVA_Val_Custom_Corrected.csv") as csvDataFile:
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
        elif i == 6:
            classes = []
            with open("files/AVA_Test_Custom_Corrected.csv") as csvDataFile:
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

plt.show()
