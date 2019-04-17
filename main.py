from model import  *
from util import *

if __name__ == '__main__':
	data, zhDict, enDict = loadData('trainData')
	#print(data)
	#print(zhDict)
	print(enDict)
	enDict = enWord2Vec(enDict, toSave=True)
	print(enDict)