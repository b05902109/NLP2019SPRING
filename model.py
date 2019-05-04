import torch 
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
import torchvision.transforms as transforms
import numpy as np

class DatasetWord2Vec():
    def __init__(self, args, enDict, zhDict, dataTrain, 
                enWord2VecDict=None, zhWord2VecDict=None, 
                useEn=False, useZh=False
                ):
        self.seed = args.seed
        self.padding_len = args.padding_len
        self.embedding_dim = args.embedding_dim
        self.enDict = enDict
        self.zhDict = zhDict
        self.dataTrain = dataTrain
        self.enWord2VecDict = enWord2VecDict
        self.zhWord2VecDict = zhWord2VecDict
        self._total_data = len(dataTrain)
        self.useEn = useEn
        self.useZh = useZh
        self.enSOSVec, self.enEOSVec, self.enPADVec, self.enUNKVec = self.getSpecailVec(self.enWord2VecDict)
        self.zhSOSVec, self.zhEOSVec, self.zhPADVec, self.zhUNKVec = self.getSpecailVec(self.enWord2VecDict)
    def __len__(self):
        return self._total_data
    def __getitem__(self, idex):
        enItem1 = [self.enSOSVec]
        enItem2 = [self.enSOSVec]
        zhItem1 = [self.zhSOSVec]
        zhItem2 = [self.zhSOSVec]
        if self.useEn:
            #print(self.enDict[self.dataTrain[]])
            for i in range(self.padding_len - 2):
                try:
                    enItem1.append(self.enWord2VecDict[self.enDict[ self.dataTrain[idex][0] ][i]])
                except IndexError:
                    enItem1.append(self.enPADVec)
                except KeyError:
                    enItem1.append(self.enUNKVec)
                try:
                    enItem2.append(self.enWord2VecDict[self.enDict[ self.dataTrain[idex][1] ][i]])
                except IndexError:
                    enItem2.append(self.enPADVec)
                except KeyError:
                    enItem2.append(self.enUNKVec)
            enItem1.append(self.enEOSVec)
            enItem2.append(self.enEOSVec)
        if self.useZh:
            for i in range(self.padding_len - 2):
                try:
                    zhItem1.append(self.zhWord2VecDict[self.zhDict[ self.dataTrain[idex][0] ][i]])
                except IndexError:
                    zhItem1.append(self.zhPADVec)
                except KeyError:
                    zhItem1.append(self.zhUNKVec)
                try:
                    zhItem2.append(self.zhWord2VecDict[self.zhDict[ self.dataTrain[idex][1] ][i]])
                except IndexError:
                    zhItem2.append(self.zhPADVec)
                except KeyError:
                    zhItem2.append(self.zhUNKVec)
            zhItem1.append(self.zhEOSVec)
            zhItem2.append(self.zhEOSVec)
        enItem1 = torch.tensor(enItem1, dtype=torch.float)
        enItem2 = torch.tensor(enItem2, dtype=torch.float)
        zhItem1 = torch.tensor(zhItem1, dtype=torch.float)
        zhItem2 = torch.tensor(zhItem2, dtype=torch.float)
        if self.useEn and self.useZh:
            return enItem1, enItem2, zhItem1, zhItem2, torch.tensor(self.dataTrain[idex][2], dtype=torch.long)
        elif self.useEn:
            return enItem1, enItem2, torch.tensor(self.dataTrain[idex][2], dtype=torch.long)
        else:
            return zhItem1, zhItem2, torch.tensor(self.dataTrain[idex][2], dtype=torch.long)
    def getSpecailVec(self, word2VecDict):
        np.random.seed(self.seed)
        SOSVec = np.random.rand(self.embedding_dim)
        EOSVec = np.random.rand(self.embedding_dim)
        PADVec = np.random.rand(self.embedding_dim)
        UNKVec = np.random.rand(self.embedding_dim)
        return SOSVec, EOSVec, PADVec, UNKVec

class DatasetWordDict():
    def __init__(self, args, enDict, zhDict, dataTrain, 
                enWordDict=None, zhWordDict=None, 
                useEn=False, useZh=False
                ):
        self.padding_len = args.padding_len
        self.enDict = enDict
        self.zhDict = zhDict
        self.enWordDict = enWordDict
        self.zhWordDict = zhWordDict
        self.dataTrain = dataTrain
        self._total_data = len(dataTrain)
        self.useEn = useEn
        self.useZh = useZh
    def __len__(self):
        return self._total_data
    def __getitem__(self, idex):
        enItem1 = [self.enWordDict['<SOS>']]
        enItem2 = [self.enWordDict['<SOS>']]
        zhItem1 = [self.zhWordDict['<SOS>']]
        zhItem2 = [self.zhWordDict['<SOS>']]
        if self.useEn:
            for i in range(self.padding_len - 2):
                try:
                    enItem1.append(self.enWordDict[self.enDict[ self.dataTrain[idex][0] ][i]])
                except IndexError:
                    enItem1.append(self.enWordDict['<PAD>'])
                except KeyError:
                    enItem1.append(self.enWordDict['<UNK>'])
                try:
                    enItem2.append(self.enWordDict[self.enDict[ self.dataTrain[idex][1] ][i]])
                except IndexError:
                    enItem2.append(self.enWordDict['<PAD>'])
                except KeyError:
                    enItem2.append(self.enWordDict['<UNK>'])
            enItem1.append(self.enWordDict['<EOS>'])
            enItem2.append(self.enWordDict['<EOS>'])
            enItem1 = torch.tensor(enItem1, dtype=torch.long)
            enItem2 = torch.tensor(enItem2, dtype=torch.long)
        if self.useZh:
            for i in range(self.padding_len - 2):
                try:
                    zhItem1.append(self.zhWordDict[self.zhDict[ self.dataTrain[idex][0] ][i]])
                except IndexError:
                    zhItem1.append(self.zhWordDict['<PAD>'])
                except KeyError:
                    zhItem1.append(self.zhWordDict['<UNK>'])
                try:
                    zhItem2.append(self.zhWordDict[self.zhDict[ self.dataTrain[idex][1] ][i]])
                except IndexError:
                    zhItem2.append(self.zhWordDict['<PAD>'])
                except KeyError:
                    zhItem2.append(self.zhWordDict['<UNK>'])
            zhItem1.append(self.zhWordDict['<EOS>'])
            zhItem2.append(self.zhWordDict['<EOS>'])
            zhItem1 = torch.tensor(zhItem1, dtype=torch.long)
            zhItem2 = torch.tensor(zhItem2, dtype=torch.long)
        if self.useEn and self.useZh:
            return enItem1, enItem2, zhItem1, zhItem2, torch.tensor(self.dataTrain[idex][2], dtype=torch.long)
        elif self.useEn:
            return enItem1, enItem2, torch.tensor(self.dataTrain[idex][2], dtype=torch.long)
        else:
            return zhItem1, zhItem2, torch.tensor(self.dataTrain[idex][2], dtype=torch.long)

class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
    def forward(self,x):
        return x*torch.sigmoid(x)

class bidirRNN(torch.nn.Module):
    def __init__(self, args):
        super(bidirRNN, self).__init__()
        self.padding_len = args.padding_len
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.hidden_size
        self.bi_Gru = torch.nn.GRU(
                    input_size=args.embedding_dim,
                    hidden_size=args.hidden_size,
                    batch_first=True,
                    dropout=0.2,
                    bidirectional=args.bidirectional)
        self.linear1 = nn.Linear(args.hidden_size*2*2, args.hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.activation1 = Swish()
        self.batchnorm = nn.BatchNorm1d(args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.activation2 = Swish()
        self.linear3 = nn.Linear(64, 3)

    def forward(self, x1, x2):
        '''
        input:  [batch, padding_len(fixed), embedding_dim]
        output: [batch, hidden_size]
        '''
        batchSize = x1.shape[0]
        out1, _ = self.bi_Gru(x1)
        out2, _ = self.bi_Gru(x2)
        out1 = out1[:, -1, :].view(batchSize,self.hidden_size*2)
        out2 = out2[:, -1, :].view(batchSize,self.hidden_size*2)
        out = torch.cat([out1, out2], 1)
        out = self.linear1(out)
        out = self.dropout1(out)
        out = self.activation1(out)
        out = self.batchnorm(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        out = self.activation2(out)
        out = self.linear3(out)
        out = functional.softmax(out,dim=1)
        return out

class enRNN_WordDict(bidirRNN):
    def __init__(self, args, wordDictLen):
        super(enRNN_WordDict, self).__init__(args)
        self.embeddings = nn.Embedding(wordDictLen, args.embedding_dim)
        self.bi_rnn = bidirRNN(args)
    def forward(self, sentence1, sentence2):
        '''
        input:  [batchSize, padding_len(not fixed), 1]
        output: [batchSize, 3(...?)]
        '''
        # [batchSize, padding_len, embedding_dim] <- [batchSize, padding_len, 1]
        embeddings1 = self.embeddings(sentence1)
        embeddings2 = self.embeddings(sentence2)
        # [batchSize, 3] <- [batchSize, padding_len, embedding_dim]
        out = self.bi_rnn(embeddings1, embeddings2)
        return out


class enRNN_Word2Vec(bidirRNN):
    def __init__(self, args):
        super(enRNN_Word2Vec, self).__init__(args)
        self.bi_rnn = bidirRNN(args)
    def forward(self, sentence1, sentence2):
        '''
        input:  [batchSize, padding_len(not fixed), 1]
        output: [batchSize, 3(...?)]
        '''
        # [batchSize, 3] <- [batchSize, padding_len, embedding_dim]
        out = self.bi_rnn(sentence1, sentence2)
        return out

if __name__ == '__main__':
    print('model.py')
