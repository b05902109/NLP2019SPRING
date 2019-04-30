import torch 
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
import torchvision.transforms as transforms

class DatasetWord2Vec():
    def __init__(self, enDict, zhDict, dataTrain, 
                enWord2VecDict=None, zhWord2VecDict=None, 
                useEn=False, useZh=False
                ):
        self.enDict = enDict
        self.zhDict = zhDict
        self.dataTrain = dataTrain
        self.enWord2VecDict = enWord2VecDict
        self.zhWord2VecDict = zhWord2VecDict
        self._total_data = len(dataTrain)
    def __len__(self):
        return self._total_data

    def __getitem__(self, i):
        if useEn and useZh:
           return [self.enWord2VecDict[word] for word in self.enDict[self.dataTrain[i][0]]], \
                [self.enWord2VecDict[word] for word in self.enDict[self.dataTrain[i][1]]], \
                [self.zhWord2VecDict[word] for word in self.zhDict[self.dataTrain[i][0]]], \
                [self.zhWord2VecDict[word] for word in self.zhDict[self.dataTrain[i][1]]], \
                self.dataTrain[i][2]
        elif useEn:
            return [self.enWord2VecDict[word] for word in self.enDict[self.dataTrain[i][0]]], \
                [self.enWord2VecDict[word] for word in self.enDict[self.dataTrain[i][1]]], \
                self.dataTrain[i][2]
        else:
            return [self.zhWord2VecDict[word] for word in self.zhDict[self.dataTrain[i][0]]], \
                [self.zhWord2VecDict[word] for word in self.zhDict[self.dataTrain[i][1]]], \
                self.dataTrain[i][2]

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
                    enItem1.append(self.enWordDict[self.enDict[ self.dataTrain[i][0] ][i]])
                except IndexError:
                    enItem1.append(self.enWordDict['<PAD>'])
                except KeyError:
                    enItem1.append(self.enWordDict['<UNK>'])
                try:
                    enItem2.append(self.enWordDict[self.enDict[ self.dataTrain[i][1] ][i]])
                except IndexError:
                    enItem2.append(self.enWordDict['<PAD>'])
                except KeyError:
                    enItem2.append(self.enWordDict['<UNK>'])
            enItem1.append(self.enWordDict['<EOS>'])
            enItem2.append(self.enWordDict['<EOS>'])
        if self.useZh:
            for i in range(self.padding_len - 2):
                try:
                    zhItem1.append(self.zhWordDict[self.zhDict[ self.dataTrain[i][0] ][i]])
                except IndexError:
                    zhItem1.append(self.zhWordDict['<PAD>'])
                except KeyError:
                    zhItem1.append(self.zhWordDict['<UNK>'])
                try:
                    zhItem2.append(self.zhWordDict[self.zhDict[ self.dataTrain[i][1] ][i]])
                except IndexError:
                    zhItem2.append(self.zhWordDict['<PAD>'])
                except KeyError:
                    zhItem2.append(self.zhWordDict['<UNK>'])
            zhItem1.append(self.zhWordDict['<EOS>'])
            zhItem2.append(self.zhWordDict['<EOS>'])
        enItem1 = torch.tensor(enItem1, dtype=torch.long)
        enItem2 = torch.tensor(enItem2, dtype=torch.long)
        zhItem1 = torch.tensor(zhItem1, dtype=torch.long)
        zhItem2 = torch.tensor(zhItem2, dtype=torch.long)
        if self.useEn and self.useZh:
            return enItem1, enItem2, zhItem1, zhItem2, torch.tensor(self.dataTrain[idex][2], dtype=torch.long)
        elif self.useEn:
            return enItem1, enItem2, torch.tensor(self.dataTrain[idex][2], dtype=torch.long)
        else:
            return zhItem1, zhItem2, torch.tensor(self.dataTrain[idex][2], dtype=torch.long)

class BaseRNN(torch.nn.Module):
    def __init__(self, args):
        super(BaseRNN, self).__init__()
        self.batchSize = args.batch_size
        self.padding_len = args.padding_len
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.hidden_size
        self.bi_Gru = torch.nn.GRU(
                    input_size=args.embedding_dim,
                    hidden_size=args.hidden_size,
                    batch_first=True,
                    dropout=args.dropout,
                    bidirectional=args.bidirectional)

    def forward(self, x):
        '''
        input:  [batch, padding_len(fixed), embedding_dim]
        output: [batch, hidden_size]
        '''
        out, _ = self.bi_Gru(x)
        out = out[:, -1, :].view(self.batchSize,self.hidden_size*2)
        return out

class enRNN_WordDict(BaseRNN):
    def __init__(self, args, enWordDict=None):
        super(enRNN_WordDict, self).__init__(args)
        self.embeddings = nn.Embedding(len(enWordDict), args.embedding_dim)
        self.bi_gru = BaseRNN(args)
        self.linear1 = nn.Linear(args.hidden_size*2*2, args.hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.RReLU()
        self.batchnorm = nn.BatchNorm1d(self.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, 3)
    def forward(self, sentence1, sentence2, sentenceLen):
        '''
        input:  [batchSize, padding_len(not fixed), 1]
        output: [batchSize, 3(...?)]
        '''
        # [batchSize, padding_len, embedding_dim] <- [batchSize, padding_len, 1]
        embeddings1 = self.embeddings(sentence1)
        embeddings2 = self.embeddings(sentence2)
        
        # [, padding_len(fixed), embedding_dim] <- [, padding_len(not fixed), embedding_dim]
        #s1_packed = nn.utils.rnn.pack_padded_sequence(embeddings1, sentenceLen, batch_first=False)
        #s2_packed = nn.utils.rnn.pack_padded_sequence(embeddings2, sentenceLen, batch_first=False)
        # [batch, hidden_size] <- [batch, padding_len(fixed), embedding_dim]
        #s1_output_reshape = self.bi_gru(s1_packed)
        #s2_output_reshape = self.bi_gru(s2_packed)

        # [batchSize, hidden_size*2] <- [batchSize, padding_len, embedding_dim]
        s1_output_reshape = self.bi_gru(embeddings1)
        s2_output_reshape = self.bi_gru(embeddings2)

        s1s2_cat = torch.cat([s1_output_reshape, s2_output_reshape], 1)
        out200 = self.linear1(s1s2_cat)
        out200 = self.dropout(out200)
        out200 = self.activation(out200)
        out200 = self.batchnorm(out200)
        out3 = self.linear2(out200)
        out3 = functional.softmax(out3,dim=1)
        return out3

if __name__ == '__main__':
    print('model.py')