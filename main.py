from model import  *
from util import *
import torch.optim as optim

if __name__ == '__main__':
    dataTrain, zhDictTrain, enDictTrain = loadData('trainData')
    dataTest, zhDictTest, enDictTest = loadData('testData')
    zhDict = dict(zhDictTrain)
    zhDict.update(zhDictTest)
    enDict = dict(enDictTrain)
    enDict.update(enDictTest)
    #enWord2VecDict = getEnWord2VecDict(enDict, toSave=False)
    #zhWord2VecDict = getZhWord2VecDict(zhDict, toSave=False)
    enWordDict = getWordDict(enDict)
    zhWordDict = getWordDict(zhDict)
    
    batch_size = 256
    dataTrainLen = len(dataTrain)
    trainDataset = DatasetWordDict(
                enDict, zhDict, dataTrain[dataTrainLen/10:]
                enWordDict=enWordDict, zhWordDict=zhWordDict, 
                useEn=True
            )
    trainLoader = torch.utils.data.DataLoader(
                dataset=trainDataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )

    validDataset = DatasetWordDict(
                enDict, zhDict, dataTrain[:dataTrainLen/10]
                enWordDict=enWordDict, zhWordDict=zhWordDict, 
                useEn=True
            )
    validLoader = torch.utils.data.DataLoader(
                dataset=validDataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )

    myModel = Net(
            embedding_dim=200, hidden_dim=200, padding_len=30, 
            enWordDict=enWordDict, zhWordDict=zhWordDict
        )

    lossFunction = nn.CrossEntropyLoss(weight=torch.tensor([1/15, 1/5, 1/16]))
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
        # training
        totalLoss, totalStep = 0.0, 0.0
        for index, (sentence1, sentence2, target) in enumerate(trainloader):
            optimizer.zero_grad()
            predict = myModel() #????
            loss = lossFunction(predict, target)
            loss.backward()
            optimizer.step()
            totalStep += 1
            totalValidLoss += loss
        # valid
        totalValidLoss, totalValidStep = 0.0, 0.0
        with torch.no_grad():
            myModel.eval()
            for index, (sentence1, sentence2, target) in enumerate(validLoader):
                optimizer.zero_grad()
                predict = myModel() #????
                loss = lossFunction(predict, target)
                totalValidStep += 1
                totalValidLoss += loss
            myModel.train()
        progress_msg = 'epoch:%3d, loss:%.3f, valid:%.3f'
                    %(epoch, totalLoss/totalStep, totalValidLoss/totalValidStep)
        print(progress_msg)
        # save
        modelSavePath = os.path.join(pathPrefix, './model/models_e%d.pt' % (epoch))
        myModel.save(modelSavePath)

# scp main.py model.py util.py b05902109@cml11.csie.ntu.edu.tw:/tmp2/b05902109/NLP2019SPRING/