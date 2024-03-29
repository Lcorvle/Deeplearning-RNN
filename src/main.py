# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pdb

import data
from model import LMModel
import copy
import os
import os.path as osp

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', default=True, action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=4, help='GPU device id used')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='Learning rate of the model')
parser.add_argument('--version', type=str, default="default", help='version of the run')

args = parser.parse_args()
if args.version == "defalut":
    version = "Naive LSTM+Decoder"
else:
    version = args.version
time_stamp = time.localtime(time.time())
version = version + "-" + str(time_stamp.tm_year) + '-' +  str(time_stamp.tm_mon)\
          + '-' +  str(time_stamp.tm_mday) + '-' +  str(time_stamp.tm_hour)\
          + '-' +  str(time_stamp.tm_min) + '-' +  str(time_stamp.tm_sec)


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train

if args.cuda:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size, 'valid': eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)

# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (build your language model here)
n_input = 256
n_hid = 512
n_layers = 2
model = LMModel(data_loader.nvoc,
                n_input,
                n_hid,
                n_layers)

if args.cuda:
    model.cuda()

opt = optim.Adam(model.parameters(),
                 lr=args.learning_rate,
                 weight_decay=0.00001)

criterion = nn.CrossEntropyLoss()


########################################
# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate():
    t1 = time.time()
    model.eval()
    data_loader.set_valid()

    error = 0
    accuracy = 0
    number = 0
    step = 0
    step_per_report = 100

    while True:
        input, target, flag = data_loader.get_batch()
        if args.cuda:
            input = input.cuda()
            target = target.cuda()
        decoded, hidden = model(input)
        _error = criterion(decoded, target)
        perplexity = torch.exp(_error)
        _, index = torch.max(decoded, dim=1)
        _accuracy = torch.sum(index == target)

        error += _error.item() * decoded.size(0)
        number += decoded.size(0)
        accuracy += _accuracy.item()

        if step % step_per_report == 0:
            delta_t = time.time() - t1
            print("Valid Step " + str(step) + ":",
                  "Time=" + str(delta_t),
                  "Loss=" + str(_error.item()),
                  "Accuracy=" + str(_accuracy.item() / decoded.size(0)),
                  "Perplexity=" + str(perplexity.item()))
        step += 1
        if flag:
            break
    delta_t = time.time() - t1
    error /= number
    accuracy /= number
    perplexity = math.exp(error)
    print("Valid Step " + str(step) + ":",
          "Time=" + str(delta_t),
          "Loss=" + str(error),
          "Accuracy=" + str(accuracy),
          "Perplexity=" + str(perplexity))
    return perplexity


########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function
def train():
    t1 = time.time()
    model.train()
    data_loader.set_train()

    error = 0
    accuracy = 0
    number = 0
    step = 0
    step_per_report = 100

    while True:
        opt.zero_grad()
        input, target, flag = data_loader.get_batch()
        if args.cuda:
            input = input.cuda()
            target = target.cuda()
        decoded, hidden = model(input)
        # pdb.set_trace()
        _error = criterion(decoded, target)
        perplexity = torch.exp(_error)
        _, index = torch.max(decoded, dim=1)
        _accuracy = torch.sum(index == target)

        error += _error.item() * decoded.size(0)
        number += decoded.size(0)
        accuracy += _accuracy.item()
        _error.backward()
        opt.step()
        if step % step_per_report == 0:
            delta_t = time.time() - t1
            print("Train Step " + str(step) + ":",
                  "Time=" + str(delta_t),
                  "Loss=" + str(_error.item()),
                  "Accuracy=" + str(_accuracy.item() / decoded.size(0)),
                  "Perplexity=" + str(perplexity.item()))
        step += 1
        if flag:
            break
    delta_t = time.time() - t1
    error /= number
    accuracy /= number
    perplexity = math.exp(error)
    print("Train Step " + str(step) + ":",
          "Time=" + str(delta_t),
          "Loss=" + str(error),
          "Accuracy=" + str(accuracy),
          "Perplexity=" + str(perplexity))
    return perplexity


########################################


# Loop over epochs.
def loop_over_epochs():
    best_valid_perplexity = 1e10
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = -1
    perplexitys = {
        'train': [],
        'valid': []
    }
    t1 = time.time()
    epoch_per_report = 1

    for epoch in range(1, args.epochs + 1):
        if epoch % epoch_per_report == 0:
            print("Epoch " + str(epoch))
        perplexitys['train'].append(train())
        valid_perplexity = evaluate()

        if valid_perplexity < best_valid_perplexity:
            best_valid_perplexity = valid_perplexity
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        perplexitys['valid'].append(valid_perplexity)
        delta_t = time.time() - t1
        if epoch % epoch_per_report == 0:
            print("Epoch " + str(epoch),
                  "Time:" + str(delta_t // 3600) + 'h' + str(delta_t // 60 % 60) + 'm' + str(delta_t % 60) + 's',
                  "Best epoch:" + str(best_epoch),
                  "Best valid perplexity:" + str(best_valid_perplexity))
    delta_t = time.time() - t1
    print("Time:" + str(delta_t // 3600) + 'h' + str(delta_t // 60 % 60) + 'm' + str(delta_t % 60) + 's',
          "Best epoch:" + str(best_epoch),
          "Best valid perplexity:" + str(best_valid_perplexity))
    model.load_state_dict(best_model_state_dict)
    return model, perplexitys, best_epoch, best_valid_perplexity


model, perplexitys, best_epoch, best_valid_perplexity = loop_over_epochs()
torch.save(model.state_dict(), './result/' + version + "-model.pt")

with open('./result/' + version + '-train.txt', 'wb') as f:
    pickle.dump(perplexitys["train"], f)

with open('./result/' + version + '-valid.txt', 'wb') as f:
    pickle.dump(perplexitys["valid"], f)

with open('./result/' + version + '.sh', 'w') as f:
    f.write("python main.py --version={0} --epochs={1} --train_batch_size={2} --eval_batch_size={3} --max_sql={4} "
            "--seed={5} {6}--gpu_id={7} --learning_rate={8}".format(
        '\"' + str(version) + '\"', #version
        str(args.epochs), # epochs
        str(args.train_batch_size), # train_batch_size
        str(args.eval_batch_size), # eval_batch_size
        str(args.max_sql), # max_sql
        str(args.seed), # seed
        "--cuda= " if args.cuda else "", # cuda
        str(args.gpu_id), # gpu_id
        str(args.learning_rate))) # learning_rate