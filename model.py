import torch 
import torch.nn as nn
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
                , self.dataTrain[i][2]
        elif useEn:
            return [self.enWord2VecDict[word] for word in self.enDict[self.dataTrain[i][0]]], \
                [self.enWord2VecDict[word] for word in self.enDict[self.dataTrain[i][1]]], \
                , self.dataTrain[i][2]
        else:
            return [self.zhWord2VecDict[word] for word in self.zhDict[self.dataTrain[i][0]]], \
                [self.zhWord2VecDict[word] for word in self.zhDict[self.dataTrain[i][1]]], \
                , self.dataTrain[i][2]

class DatasetWordDict():
    def __init__(self, enDict, zhDict, dataTrain, 
                enWordDict=None, zhWordDict=None, 
                useEn=False, useZh=False
                ):
        self.enDict = enDict
        self.zhDict = zhDict
        self.enWordDict = enWordDict
        self.zhWordDict = zhWordDict
        self.dataTrain = dataTrain
        self._total_data = len(dataTrain)
    def __len__(self):
        return self._total_data

    def __getitem__(self, i):
        if useEn and useZh:
           return [self.enWordDict[word] for word in self.enDict[self.dataTrain[i][0]]], \
                [self.enWordDict[word] for word in self.enDict[self.dataTrain[i][1]]], \
                [self.zhWordDict[word] for word in self.zhDict[self.dataTrain[i][0]]], \
                [self.zhWordDict[word] for word in self.zhDict[self.dataTrain[i][1]]], \
                , self.dataTrain[i][2]
        elif useEn:
            return [self.enWordDict[word] for word in self.enDict[self.dataTrain[i][0]]], \
                [self.enWordDict[word] for word in self.enDict[self.dataTrain[i][1]]], \
                , self.dataTrain[i][2]
        else:
            return [self.zhWordDict[word] for word in self.zhDict[self.dataTrain[i][0]]], \
                [self.zhWordDict[word] for word in self.zhDict[self.dataTrain[i][1]]], \
                , self.dataTrain[i][2]

class Net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, padding_len, 
                    dropout=0.2, useEn=False, useZh=False, 
                    enWordDict=enWordDict, zhWordDict=zhWordDict):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.padding_len = padding_len
        
        self.embeddings = nn.Embedding(n_vocab, embedding_dim)
        self.bi_gru = nn.GRU(embedding_dim, hidden_dim, num_layers=1, dropout=dropout, bidirectional=True)
        self.linear1 = nn.Linear(hidden_dim, 32)
        self.linear2 = nn.Linear(32, 3)
    def forward(self, sentence1, sentence1Len, sentence2, sentence2Len):   # [batch_size, len]
        batch_size = sentence1.shape[0]
        
        embeddings = self.embeddings(sentence1)                         # [batch_size, len, embedding_dim]
        s1_packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, sentence1Len, batch_first=True)
        s1_output, _ = self.bi_gru(s1_packed)
        s1_output_reshape = s1_output.view(batch_size, -1)    # [batch_size, 3] < [batch_size, len*embedding_dim]
        
        embeddings = self.embeddings(sentence2)                         # [batch_size, len, embedding_dim]
        s2_packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, sentence1Len, batch_first=True)
        s2_output, _ = self.bi_gru(s2_packed)
        s2_output_reshape = s2_output.view(batch_size, -1)    # [batch_size, 3] < [batch_size, len*embedding_dim]

        s1s2_cat = torch.cat([s1_output_reshape, s2_output_reshape], 1)
        output32 = self.linear1(s1s2_cat)
        output3 = self.linear2(output32)
        return output

def collate_fn(batch):
    sentence1, sentence2, target = zip(*batch)
    sentence1, sentence1Len = _padding(sentence1)
    sentence2, sentence2Len = _padding(sentence2)
    return (sentence1, sentence1Len), (sentence2, sentence2Len), target

    video = torch.tensor(video, dtype=torch.float)
    correct_caption, correct_length = _padding(correct_caption)
    correct_caption, correct_length, correct_indices = \
            _sort_and_get_indices(correct_caption, correct_length)
    wrong_caption, wrong_length = _padding(wrong_caption)
    wrong_caption, wrong_length, wrong_indices = \
            _sort_and_get_indices(wrong_caption, wrong_length)
    return video, (correct_caption, correct_length, correct_indices),\
            (wrong_caption, wrong_length, wrong_indices)