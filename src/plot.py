import matplotlib.pyplot as plt
import pickle
import numpy as np


def plotAccCurve(h_list, name_list, modelName, path, num_epochs=1):


    plt.figure(figsize=(8, 6), dpi = 120)
    plt.title("Training & Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Perplexity")
    # pdb.set_trace()
    start = 10
    for i in range(len(h_list)):
        plt.plot(range(start, num_epochs + start), h_list[i], label=modelName + "-" + name_list[i])
    # plt.ylim((0,1.))
    plt.xticks(np.arange(start, num_epochs + 1 + start, (num_epochs + 9) // 10))
    plt.legend()
    plt.savefig(path)
    plt.show()

if __name__ == '__main__':
    path = "result"
    # files = os.listdir(path)
    files = [
        'Naive LSTM-2019-6-14-16-28-6-train.txt',
         'Nn_LSTM-2019-6-14-16-24-6-train.txt',
         'Optimized LSTM+Decoder+Log_softmax+Log_softmax-2019-6-15-16-52-25-train.txt',
         # 'Optimized LSTM+Decoder+Tanh-2019-6-15-16-10-7-train.txt',
         'Optimized LSTM+Decoder-2019-6-15-14-17-36-train.txt',
         'Optimized LSTM-2019-6-15-9-10-39-train.txt',
         'Optimized_LSTM+Decoder+Log_softmax-2019-6-15-16-15-23-train.txt',
         # 'Optimized_LSTM+Decoder+Relu-2019-6-15-14-56-18-train.txt'
     ]
    print(files)
    h_list = []
    name_list = []
    for file in files:
        if file.endswith("-train.txt"):
            name = path + "/" + file.replace("-train.txt", "")
            print(name)
            fp = open(name + "-train.txt", 'rb')
            train = pickle.load(fp)
            fp.close()
            fp = open(name + "-valid.txt", 'rb')
            valid = pickle.load(fp)
            fp.close()
            import math

            print(math.log(valid[-1]) / 73760)
            name = file.split('-')[0]
            # h_list.append(train)
            # name_list.append(name + "-train")
            h_list.append(valid[10:])
            name_list.append(name + "-valid")
            # plotAccCurve([train, valid],
            #              [name + "-train", name + "-valid"],
            #              'RNN', path + "/" + name + "-accuracy-curve.png", num_epochs=30)
    plotAccCurve(h_list, name_list, 'RNN', path + "/train-accuracy-curves.png", num_epochs=20)