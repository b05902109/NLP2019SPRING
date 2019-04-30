import os, csv, jieba, torch, numpy, random
from gensim.models import word2vec, Word2Vec

def setRandomSeed(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

def loadData(args, target='trainData'):
    zhDict = {}
    enDict = {}
    data = []
    path = ''
    relationDict = {'agreed': 0, 'disagreed': 1,'unrelated':2}

    if target == 'trainData':
        path = os.path.join(args.pathPrefix, args.dataFolder, args.trainData)
    elif target == 'testData':
        path = os.path.join(args.pathPrefix, args.dataFolder, args.testData)

    print('==== load data from: ' + path + ' ====')
    with open(path, encoding='utf8') as csvfile:
        rows = csv.reader(csvfile)
        for index, row in enumerate(rows):
            if index == 0:
                continue
            
            if row[1] not in zhDict:
                zhDict[row[1]] = list(jieba.cut(row[3], cut_all=False))
                enDict[row[1]] = row[5].replace('!', ' !').replace('?', ' ?').replace('.', '  .').replace('\'', '  \' ').replace('\"', ' \' ').replace('-', ' ').split()
            
            if row[2] not in zhDict:
                zhDict[row[2]] = list(jieba.cut(row[4], cut_all=False))
                enDict[row[2]] = row[6].replace('!', ' !').replace('?', ' ?').replace('.', '  .').replace('\'', '  \' ').replace('\"', ' \' ').replace('-', ' ').split()
            
            if target == 'trainData':
                data.append([row[1], row[2], relationDict[row[7]]])
            elif target == 'testData':
                data.append([row[1], row[2]])
            #if index == 8000:
            #	break
    print('==== load data from: ' + path + ' success. ====')
    return data, zhDict, enDict

def getEnWord2VecDict(args, enDict, toSave=False):
    print("==== getEnWord2VecDict start ====")
    enWord2VecDictPath = os.path.join(args.pathPrefix, args.pretrainFolder, args.enWord2VecDict%(args.embedding_Dim))
    enWord2VecDict = None
    if toSave:
        enWord2VecPretrainDataPath = os.path.join(args.pathPrefix, args.pretrainFolder, args.enWord2VecPretrainData)
        sentences = word2vec.Text8Corpus(enWord2VecPretrainDataPath)
        enWord2VecDict = Word2Vec(sentences, size=args.embedDim, workers = 16)
        enWord2VecDict.save(enWord2VecDictPath)
        print("==== new enWord2Vec Dict saved ====")
    else:
        enWord2VecDict = Word2Vec.load(enWord2VecDictPath)
        print("==== old enWord2Vec Dict loaded ====")
    return enWord2VecDict

def getZhWord2VecDict(args, zhDict, toSave=False):
    print("==== getZhWord2VecDict start ====")
    zhWord2VecDictPath = os.path.join(args.pathPrefix, args.pretrainFolder, args.zhWord2VecDict%(args.embedding_Dim))
    zhWord2VecDict = None
    if toSave:
        data = []
        for key,values in  zhDict.items():
            for word in values:
                data.append(word)
        zhWord2VecDict = Word2Vec(data, size=args.embedDim, workers=16)
        zhWord2VecDict.save(zhWord2VecDictPath)
        print("==== new zhWord2Vec Dict saved ====")
    else:
        zhWord2VecDict = Word2Vec.load(zhWord2VecDictPath)
        print("==== old zhWord2Vec Dict loaded ====")
    return zhWord2VecDict

def getWordDict(myDict):
    wordDict = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
    for key,values in myDict.items():
        for word in values:
            if word not in wordDict:
                wordDict[word] = len(wordDict)
    return wordDict