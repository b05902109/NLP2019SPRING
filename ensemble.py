import pandas as pd
import numpy as np
import os
from args import *


if __name__ == '__main__':
	args = get_args()
	w2i = {'agreed':0, 'disagreed':1, 'unrelated':2}
	i2w = {0:'agreed', 1:'disagreed', 2:'unrelated'}
	filenameList = ['ans_Zh_wordDict_embid100_hid50_li32.csv',
					'ans_Zh_wordDict_embid100_hid50_li64.csv',
					'ans_Zh_wordDict_embid200_hid100_li32.csv',
					'ans_Zh_wordDict_embid200_hid100_li64.csv']
	answerDict = {}
	for filename in filenameList:
		print(filename, 'start')
		filenamePath = os.path.join(args.pathPrefix, 'predict', filename)
		answer = pd.read_csv(filenamePath)
		for index in range(len(answer)):
			if answer['id'][index] not in answerDict:
				answerDict[answer['id'][index]] = np.zeros(3)
			answerDict[answer['id'][index]][w2i[answer['Category'][index]]] += 1
			#if index == 3:
			#	print(answerDict)
			#	exit(1)
		print(filename, 'finish')

	voteDict = {}
	for idx, vote in answerDict.items():
		voteStr = '%d/%d/%d'%(vote[0], vote[1], vote[2])
		if voteStr not in voteDict:
			voteDict[voteStr] = 0
		voteDict[voteStr] += 1
	print(voteDict)
	
	with open(os.path.join('submission_0.csv'), 'w') as f:
		f.write('id,Category\n')
		for idx, vote in answerDict.items():
			f.write('%d,%s\n' % (idx, i2w[np.argmax(vote)]))

'''
{'0/0/4': 48788, '1/0/3': 10302, '2/0/2': 6771, '4/0/0': 8153, '3/0/1': 5941, '0/1/3': 156, '2/1/1': 5, '3/1/0': 3, '1/1/2': 7}
NLP_data/model/BidirRNN_WordDict_model_e29_loss0.712_valid0.751.h5
NLP_data/model/BidirRNN_WordDict_model_e29_loss0.725_valid0.761.h5
NLP_data/model/BidirRNN_WordDict_model_e29_loss0.732_valid0.765.h5
'''