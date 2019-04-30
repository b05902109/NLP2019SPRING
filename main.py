from model import  *
from util import *
from args import *
import torch.optim as optim

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = get_args()
    setRandomSeed(args.seed)
    dataTrain, zhDictTrain, enDictTrain = loadData(args, 'trainData')
    dataTest, zhDictTest, enDictTest = loadData(args, 'testData')
    zhDict = dict(zhDictTrain)
    zhDict.update(zhDictTest)
    enDict = dict(enDictTrain)
    enDict.update(enDictTest)
    #enWord2VecDict = getEnWord2VecDict(enDict, toSave=False)
    #zhWord2VecDict = getZhWord2VecDict(zhDict, toSave=False)
    enWordDict = getWordDict(enDict)
    zhWordDict = getWordDict(zhDict)
    
    dataTrainLen = len(dataTrain)
    print(dataTrainLen)
    trainDataset = DatasetWordDict(
                args, enDict, zhDict, dataTrain[dataTrainLen//10:],
                enWordDict=enWordDict, zhWordDict=zhWordDict, 
                useEn=True,
            )
    validDataset = DatasetWordDict(
                args, enDict, zhDict, dataTrain[:dataTrainLen//10],
                enWordDict=enWordDict, zhWordDict=zhWordDict, 
                useEn=True,
            )

    trainLoader = torch.utils.data.DataLoader(
                dataset=trainDataset,
                batch_size=args.batch_size,
                shuffle=True,
            )
    validLoader = torch.utils.data.DataLoader(
                dataset=validDataset,
                batch_size=args.batch_size,
                shuffle=False,
            )

    myModel = enRNN_WordDict(args, enWordDict=enWordDict).cuda()

    lossFunction = nn.CrossEntropyLoss(weight=torch.tensor([1/15, 1/5, 1/16]).cuda())
    optimizer = optim.Adam(myModel.parameters(), lr=args.learning_rate)
    sentenceLen = torch.tensor([args.padding_len]*args.batch_size)
    for epoch in range(30):
        # training
        totalLoss, totalStep = 0.0, 0.0
        for index, (sentence1, sentence2, target) in enumerate(trainLoader):
            #print(sentence1.shape)
            #exit(1)
            optimizer.zero_grad()
            predict = myModel.forward(sentence1.cuda(), sentence2.cuda(), sentenceLen.cuda())
            loss = lossFunction(predict, target.cuda())
            loss.backward()
            optimizer.step()
            totalStep += 1
            totalLoss += loss
        # valid
        totalValidLoss, totalValidStep = 0.0, 0.0
        with torch.no_grad():
            myModel.eval()
            for index, (sentence1, sentence2, target) in enumerate(validLoader):
                optimizer.zero_grad()
                predict = myModel.forward(sentence1.cuda(), sentence2.cuda(), sentenceLen.cuda())
                loss = lossFunction(predict, target.cuda())
                totalValidStep += 1
                totalValidLoss += loss
            myModel.train()
        totalLoss = totalLoss/totalStep
        totalValidLoss = totalValidLoss/totalValidStep
        progress_msg = 'epoch:%3d, loss:%2.3f, valid:%2.3f'%(epoch, totalLoss, totalValidLoss)
        print(progress_msg)
        # save
        modelSavePath = os.path.join(args.pathPrefix, args.modelFolder, 
                    'model_e=%2d_loss=%2.3f_valid=%2.3f'%(epoch, totalLoss, totalValidLoss))
        torch.save(myModel.state_dict(), modelSavePath)

# scp main.py model.py util.py b05902109@cml11.csie.ntu.edu.tw:/tmp2/b05902109/NLP2019SPRING/