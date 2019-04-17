import os
import csv
import jieba
from gensim.models import word2vec

def loadData(target='trainData'):
	zhDict = {}
	enDict = {}
	data = []
	path = ''
	relationDict = {"agreed": 0, "disagreed": 1,"unrelated":2}

	if target == "trainData":
		path = "./data/train.csv"
	elif target == 'testData':
		path = "./data/test.csv"

	print("load data from: " + path)
	with open(path, encoding="utf8") as csvfile:
		rows = csv.reader(csvfile)
		for index, row in enumerate(rows):
			if index == 0:
				continue
			
			if row[1] not in zhDict:
				zhDict[row[1]] = list(jieba.cut(row[3], cut_all=False))
				enDict[row[1]] = row[5].replace("!", " !").replace("?", " ?").replace(".", "  .").replace("\"", "  \" ").replace("'", " ' ").split()
			
			if row[2] not in zhDict:
				zhDict[row[2]] = list(jieba.cut(row[4], cut_all=False))
				enDict[row[2]] = row[6].replace("!", " !").replace("?", " ?").replace(".", "  .").replace("\"", "  \" ").replace("'", " ' ").split()
			
			if target == "trainData":
				data.append([row[1], row[2], relationDict[row[7]]])
			elif target == "testData":
				data.append([row[1], row[2]])
			if index == 5:
				break
	return data, zhDict, enDict

def enWord2Vec(enDict, toSave=False, embedDim=200):
	if toSave:
		sentences = word2vec.Text8Corpus("./pretrain/text8")
		enWord2VecModel = word2vec.Word2Vec(sentences, size=embedDim)
		enWord2VecModel.save("./pretrain/enWord2VecModel_dim=%d.model"%(embedDim))
	else:
		enWord2VecModel.load("./pretrain/enWord2VecModel_dim=%d.model"%(embedDim))
	
	for key,values in  engdict.items():
		vec = [model[word] for word in values]
		enDict[key] = vec
	return engdict

def zhWord2Vec(data):
	return