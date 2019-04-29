import os, csv, jieba
from gensim.models import word2vec, Word2Vec

pathPrefix = '../NLP_data/'

def loadData(target='trainData'):
	zhDict = {}
	enDict = {}
	data = []
	path = ''
	relationDict = {'agreed': 0, 'disagreed': 1,'unrelated':2}

	if target == 'trainData':
		path = pathPrefix + './data/train.csv'
	elif target == 'testData':
		path = pathPrefix + './data/test.csv'

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
			if index == 5:
				break
	print('==== load data from: ' + path + 'success. ====')
	return data, zhDict, enDict

def getEnWord2VecDict(enDict, toSave=False, embedDim=200):
	print("==== getEnWord2VecDict start ====")
	enWord2VecDict = None
	if toSave:
		sentences = word2vec.Text8Corpus(pathPrefix + './pretrain/text8')
		enWord2VecDict = Word2Vec(sentences, size=embedDim, workers = 16)
		enWord2VecDict.save(pathPrefix + './pretrain/enWord2VecModel_dim=%d.model'%(embedDim))
		print("==== new enWord2Vec Dict saved ====")
	else:
		enWord2VecDict = Word2Vec.load(pathPrefix + './pretrain/enWord2VecModel_dim=%d.model'%(embedDim))
		print("==== old enWord2Vec Dict loaded ====")
	return enWord2VecDict

def getZhWord2VecDict(zhDict, toSave=False, embedDim=200):
	print("==== getZhWord2VecDict start ====")
	zhWord2VecDict = None
	if toSave:
		data = []
		for key,values in  zhDict.items():
			for word in values:
				data.append(word)
		zhWord2VecDict = Word2Vec(data, size=embedDim, workers = 16)
		zhWord2VecDict.save(pathPrefix + './pretrain/zhWord2VecModel_dim=%d.model'%(embedDim))
		print("==== new zhWord2Vec Dict saved ====")
	else:
		zhWord2VecDict = Word2Vec.load(pathPrefix + './pretrain/zhWord2VecModel_dim=%d.model'%(embedDim))
		print("==== old zhWord2Vec Dict loaded ====")
	return zhWord2VecDict

def getWordDict(myDict):
	wordDict = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
	for key,values in myDict.items():
		for word in values:
			if word not in wordDict:
				wordDict[word] = len(wordDict)
	return wordDict