import os
import torch
from random import shuffle

class Corpus(object):
    def __init__(self, path, batch_size, max_sql):
        self.word_vocab = []
        self.word_id = {}

        self.trainCnt, self.trainWord = self.tokenize(os.path.join(path, 'train.txt'))
        self.validCnt, self.validWord = self.tokenize(os.path.join(path, 'valid.txt'))
        self.nvoc = len(self.word_vocab)
        self.dset_flag = "train"
        
        ## max_sql means the maximum sequence length
        self.max_sql = max_sql
        self.batch_size = batch_size
        print("size of train set: ",self.trainCnt)
        print("size of valid set: ",self.validCnt)
        self.train_batch_num = self.trainCnt // self.batch_size["train"]
        self.valid_batch_num = self.validCnt // self.batch_size["valid"]
        self.train = None
        self.valid = None

    def set_train(self):
        self.dset_flag = "train"
        self.train_si = 0
        self.train = self.getData(self.trainWord,
                                  self.trainCnt,
                                  self.batch_size['train'],
                                  self.train_batch_num, True)

    def set_valid(self):
        self.dset_flag = "valid"
        self.valid_si = 0
        self.valid = self.getData(self.validWord,
                                  self.validCnt,
                                  self.batch_size['valid'],
                                  self.valid_batch_num, False)

    def getData(self, wordList, wordCnt, bh_size, bh_num, isShuf):
        file_tokens = torch.LongTensor(wordCnt)
        wId = list(range(len(wordList)))
        if isShuf:
            shuffle(wId)

        token_id = 0
        for i in range(len(wordList)):
            for j in range(len(wordList[wId[i]])):
                file_tokens[token_id] = wordList[wId[i]][j]
                token_id += 1

        file_tokens = file_tokens.narrow(0, 0, bh_size * bh_num)
        file_tokens = file_tokens.view(bh_size, -1).t().contiguous()
        return file_tokens

    def tokenize(self, file_name):
        file_lines = open(file_name, 'r').readlines()
        num_of_words = 0
        wordList = []
        for line in file_lines:
            words = line.split() + ['<eos>']
            num_of_words += len(words)
            for word in words:
                if word not in self.word_id:
                    self.word_id[word] = len(self.word_vocab)
                    self.word_vocab.append(word)
            wordList.append([self.word_id[w] for w in words])
        return num_of_words, wordList

    def get_batch(self):
        ## train_si and valid_si indicates the index of the start point of the current mini-batch
        if self.dset_flag == "train":
            start_index = self.train_si
            seq_len = min(self.max_sql, self.train.size(0)-self.train_si-1)
            data_loader = self.train
            self.train_si = self.train_si + seq_len
        else:
            start_index = self.valid_si
            seq_len = min(self.max_sql, self.valid.size(0)-self.valid_si-1)
            data_loader = self.valid
            self.valid_si = self.valid_si + seq_len
        data = data_loader[start_index:start_index+seq_len, :]
        target = data_loader[start_index+1:start_index+seq_len+1, :].view(-1)
        # tId = torch.unsqueeze(tId, dim=1)
        # target = torch.LongTensor(tId.size(0), self.nvoc).scatter_(dim = 1, index = tId, value = 1)

        ## end_flag indicates whether a epoch (train or valid epoch) has been ended
        if self.dset_flag == "train" and self.train_si+1 == self.train.size(0):
            end_flag = True
            self.train_si = 0
        elif self.dset_flag == "valid" and self.valid_si+1 == self.valid.size(0):
            end_flag = True
            self.valid_si = 0
        else:
            end_flag = False
        return data, target, end_flag
