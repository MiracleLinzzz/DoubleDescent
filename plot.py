from cv2 import TermCriteria_COUNT
from matplotlib import markers
import numpy as np
import os
import matplotlib.pyplot as plt
path = os.getcwd()
w1 = [1, 2, 4, 6, 10, 14, 22, 30, 46, 62, 94, 126, 180]
w2 = [3, 16]
w3 = [38, 54, 78, 110, 308]
wnew1 = sorted(w1 + w2 + w3)
wnew2 = [1, 2, 3, 4, 10, 14, 19, 22, 30, 38, 46, 62, 78, 94, 126, 180, 308]
wnew3 = [1, 2, 3, 4, 6, 10, 14, 16, 22, 30, 38, 46, 54, 62, 78, 94, 110, 126, 180, 308]
pciture_path = "Plot_Picture"
subpath1 = "Plot1_100epochs_WidthNet_1_20_width_1000_training_10000_testing"
subpath2 = "Plot2_1000epochs_WidthNet_1_20_width_1000_training_600_testing"
subpath3 = "Plot3_1000epochs_WidthNet_1_25_1000_training_1000_testing_0.2_noise"
subpath4 = "Plot4_1000epochs_WidthNet2_1_15_1000_training_1000_testing_0.2_noise"
subpath5 = "Plot5_1000epochs_WidthNet3_1_10_1000_training_1000_testing_0.2_noise"
subpath8 = "Plot8_1000epochs_WidthNet2_1_10_1500_training_1500_testing_0.2_noise"
subpath9 = "Plot9_1000epochs_WidthNet2_widths_10000_training_10000_testing_0.2_noise"
subpath10 = "Plot10_1000epochs_WidthNet2_widths2_10000_training_10000_testing_0.2_noise"
subpath11 = "Plot11_1000epochs_WidthNet2_widths3_10000_training_10000_testing_0.2_noise"
subpath12 = "Plot12_1000epochs_WidthNet1_widths4_10000_training_10000_testing_0.2_noise"
subpath13 = "Plot13_1000epochs_WidthNet2_widths_10000_training_10000_testing_0_noise"
subpath14 = "Plot14_5000epochs_WidthNet2_512_10000_training_10000_testing_0.2_noise"
subpath15 = "Plot15_1000epochs_WidthNet2_widths_10000_training_10000_testing_0.2_noise_fixlr"
subpath16 = "Plot16_1000epochs_WidthNet2_widths_10000_training_10000_testing_0.2_noise_SGLD"
path1 = os.path.join(path, subpath1, "cnn_width_test.npy")
path2 = os.path.join(path, subpath2, "cnn_width_test.npy")
path3 = os.path.join(path, subpath3, "cnn_width_test.npy")
path4 = os.path.join(path, subpath4, "cnn_width_test.npy")
path5 = os.path.join(path, subpath5, "cnn_width_test.npy")
path8 = os.path.join(path, subpath8, "cnn_width_test.npy")
path9 = os.path.join(path, subpath9, "cnn_width2_test.npy")
path9n = os.path.join(path, subpath9, "cnn_width_test.npy")
path10 = os.path.join(path, subpath10, "cnn_width2_test.npy")
path11 = os.path.join(path, subpath11, "cnn_width2_test.npy")
path12 = os.path.join(path, subpath12, "cnn_width1_test.npy")
path13 = os.path.join(path, subpath13, "cnn_width2_test.npy")
path15 = os.path.join(path, subpath15, "cnn_width2_test.npy")
path16 = os.path.join(path, subpath16, "cnn_width2_test.npy")
path3_1 = os.path.join(path, subpath3, "cnn_width_train.npy")
path4_1 = os.path.join(path, subpath4, "cnn_width_train.npy")
path5_1 = os.path.join(path, subpath5, "cnn_width_train.npy")
path8_1 = os.path.join(path, subpath8, "cnn_width_train.npy")
path9_1 = os.path.join(path, subpath9, "cnn_width2_train.npy")
path9n_1 = os.path.join(path, subpath9, "cnn_width_train.npy")
path10_1 = os.path.join(path, subpath10, "cnn_width2_train.npy")
path11_1 = os.path.join(path, subpath11, "cnn_width2_train.npy")
path12_1 = os.path.join(path, subpath12, "cnn_width1_train.npy")
path13_1 = os.path.join(path, subpath13, "cnn_width2_train.npy")
path15_1 = os.path.join(path, subpath15, "cnn_width2_train.npy")
path16_1 = os.path.join(path, subpath16, "cnn_width2_train.npy")
def plot_epoch_error(widths, subpath, save = False):
    for width in widths:
        filename = "test_error_wdith2_" + str(width) + "_.npy"
        filepath = os.path.join(path, subpath, filename)
        data = np.load(filepath, allow_pickle = True)
        epochs = [i + 1 for i in range(len(data))]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, data)
        plt.xlabel("Epoch")
        plt.ylabel("Testing Error")
        plt.title("The testing error when width = " +  str(width))
        if save and width in [1, 10, 62, 308, 512]:
            plt.savefig(os.path.join(path, "Plot_Picture", "w" + str(width) + "epoch_double_descent.png"), format = "png", dpi = 400)
        plt.show()
def plot_width_error(filepath1, filepath2, widths, save = False):
    data1 = np.load(filepath1, allow_pickle = True)
    data2 = np.load(filepath2, allow_pickle = True)
    plt.figure(figsize = (10, 6))
    plt.plot(widths, data1, marker = 'o', ms = 3)
    plt.plot(widths, data2, marker = 'o', ms = 3)
    plt.xlabel("Width")
    plt.ylabel("Testing/Training Error")
    plt.title("The testing and training error when epoch = 1000 with 20% noise")
    plt.legend(["testing - 20% label noise", "training - 20% label noise"])
    if save:
        plt.savefig(os.path.join(path, "Plot_Picture", "width_double_descent.png"), format = "png", dpi = 400)
    plt.show()
def plot_width_error_2(filepath1, filepath2, filepath3, filepath4, widths, widths2, save = False):
    data1 = np.load(filepath1, allow_pickle = True)
    data2 = np.load(filepath2, allow_pickle = True)
    data3 = np.load(filepath3, allow_pickle = True)
    data4 = np.load(filepath4, allow_pickle = True)
    plt.figure(figsize=(10, 6))
    plt.plot(widths, data1, marker = 'o', ms = 3)
    plt.plot(widths, data2, marker = 'o', ms = 3)
    plt.plot(widths2, data3, marker = 'o', ms = 3)
    plt.plot(widths2, data4, marker = 'o', ms = 3)
    plt.xlabel("Width")
    plt.ylabel("Testing/Training Error")
    plt.title("The testing and training error when epoch = 1000 with decaying and fixed learning rate")
    plt.legend(["testing - 20% label noise + decaying lr", "training - 20% label noise + decaying lr", "testing - 20% label noise + fixed lr", "training - 20% label noise + fixed lr"])
    if save:
        plt.savefig(os.path.join(path, "Plot_Picture", "lr_double_descent.png"), format = "png", dpi = 400)
    plt.show()
def plot_subepoch_width1_error(widths, subpath, newepoch):
    new_test_error = []
    for width in widths:
        filename = "test_error_wdith2_" + str(width) + "_.npy"
        filepath = os.path.join(path, subpath, filename)
        data = np.load(filepath, allow_pickle = True)
        new_test_error.append(data[newepoch - 1])
    plt.plot(widths, new_test_error)
    plt.show()
def plot_subepoch_width2_error(widths, subpath, newepochs, save = False):
    plt.figure(figsize = (10, 6))
    for newepoch in newepochs:
        new_test_error = []
        for width in widths:
            filename = "test_error_wdith2_" + str(width) + "_.npy"
            filepath = os.path.join(path, subpath, filename)
            data = np.load(filepath, allow_pickle = True)
            new_test_error.append(data[newepoch - 1])
        plt.plot(widths, new_test_error, marker = 'o', ms = 3)
    plt.xlabel("Width")
    plt.ylabel("Testing Error")
    plt.title("The testing error v.s. width at different epoches")
    plt.legend(["epoch = " + str(newepoch) for newepoch in newepochs] + ["best"])
    if save:
        plt.savefig(os.path.join(path, "Plot_Picture", "subepoch_double_descent.png"), format = "png", dpi = 400)
    plt.show()

def plot_bestepoch_width_error(widths, subpath):
    new_test_error = []
    for width in widths:
        filename = "test_error_wdith_" + str(width) + "_.npy"
        filepath = os.path.join(path, subpath, filename)
        data = np.load(filepath, allow_pickle = True)
        new_test_error.append(np.min(data))
    plt.plot(widths, new_test_error)
    plt.show()

def plot_combine_data(file1, file2, file3, file4, width1, width2):
    new_w = width1 + width2
    data1 = np.load(file1, allow_pickle = True)
    data2 = np.load(file2, allow_pickle = True)
    data3 = np.load(file3, allow_pickle = True)
    data4 = np.load(file4, allow_pickle = True)
    new_data1 = data1.tolist() + data2.tolist()
    new_data2 = data3.tolist() + data4.tolist()
    plt.plot(new_w, new_data1)
    plt.plot(new_w, new_data2)
    plt.show()

plot_epoch_error(wnew1, subpath9, save = True)
#plot_width_error_2(path9n, path9n_1, path15, path15_1, wnew1, wnew1, save = True)
#plot_width_error(path9n, path9n_1, wnew1, save = True)
#plot_subepoch_width2_error(wnew1, subpath9, [150, 250, 500, 1000], save = True)
#plot_combine_data(path9, path10, path9_1, path10_1, w1, w2)