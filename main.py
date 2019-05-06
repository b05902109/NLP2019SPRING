from model import  *
from util import *
from args import *
import random
import torch.optim as optim
from torch.autograd import Variable

def train(args, myModel, dataTrain, enDict, zhDict, enWord2VecDict, zhWord2VecDict, enWordDict, zhWordDict):
    if args.use_word2Vec:
        modelName = 'BidirRNN_Word2Vec_'
    else:
        modelName = 'BidirRNN_WordDict_'
    
    modelSavePathPrefix = os.path.join(args.pathPrefix, args.modelFolder)
    print('modelSavePath: ', modelSavePathPrefix, '/'+ modelName + 'model_eXXX_lossXXX_validXXX.h5')
    print('==== train start ====')

    dataTrainLen = len(dataTrain)
    print(dataTrainLen)
    random.shuffle(dataTrain)

    if args.use_word2Vec:
        trainDataset = DatasetWord2Vec(
                    args, enDict, zhDict, dataTrain[dataTrainLen//10:],
                    enWord2VecDict=enWord2VecDict, 
                    zhWord2VecDict=zhWord2VecDict, 
                    useEn=args.use_En, useZh=(not args.use_En)
                )
        validDataset = DatasetWord2Vec(
                    args, enDict, zhDict, dataTrain[:dataTrainLen//10],
                    enWord2VecDict=enWord2VecDict, 
                    zhWord2VecDict=zhWord2VecDict, 
                    useEn=args.use_En, useZh=(not args.use_En)
                )
    else:
        trainDataset = DatasetWordDict(
                    args, enDict, zhDict, dataTrain[dataTrainLen//10:],
                    enWordDict=enWordDict, 
                    zhWordDict=zhWordDict, 
                    useEn=args.use_En, useZh=(not args.use_En)
                )
        validDataset = DatasetWordDict(
                    args, enDict, zhDict, dataTrain[:dataTrainLen//10],
                    enWordDict=enWordDict, 
                    zhWordDict=zhWordDict, 
                    useEn=args.use_En, useZh=(not args.use_En)
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

    if args.use_cuda:
        lossFunction = nn.CrossEntropyLoss(weight=torch.tensor([1/15, 1/5, 1/16]).cuda())
    else:
        lossFunction = nn.CrossEntropyLoss(weight=torch.tensor([1/15, 1/5, 1/16]))

    optimizer = optim.Adam(myModel.parameters(), lr=args.learning_rate)
    setRandomSeed(args.seed)
    for epoch in range(args.epoches):
        # training
        totalLoss, totalStep = 0.0, 0.0
        for index, (sentence1, sentence2, target) in enumerate(trainLoader):
            #print(sentence1.shape)
            #exit(1)
            optimizer.zero_grad()
            if args.use_cuda:
                predict = myModel.forward(sentence1.cuda(), sentence2.cuda())
                loss = lossFunction(predict, target.cuda())
            else:
                predict = myModel.forward(sentence1, sentence2)
                loss = lossFunction(predict, Variable(target))
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
                if args.use_cuda:
                    predict = myModel.forward(sentence1.cuda(), sentence2.cuda())
                    loss = lossFunction(predict, target.cuda())
                else:
                    predict = myModel.forward(sentence1, sentence2)
                    loss = lossFunction(predict, Variable(target))
                totalValidStep += 1
                totalValidLoss += loss
            myModel.train()
        totalLoss = totalLoss/totalStep
        totalValidLoss = totalValidLoss/totalValidStep
        progress_msg = 'epoch:%3d, loss:%2.3f, valid:%2.3f'%(epoch, totalLoss, totalValidLoss)
        print(progress_msg)
        # save
        modelSavePath = os.path.join(modelSavePathPrefix, modelName +
                    'model_e%02d_loss%01.3f_valid%01.3f.h5'%(epoch, totalLoss, totalValidLoss))
        # torch.save(myModel.state_dict(), modelSavePath)
        state_dict = {name:value.cpu() for name, value in myModel.state_dict().items()}
        status = {'state_dict':state_dict,}
        with open(modelSavePath, 'wb') as f_model:
            torch.save(status, f_model)

    modelName = 'model_'
    modelName += 'En_' if args.use_En else 'Zh_'
    modelName += 'word2Vec_' if args.use_word2Vec else 'wordDict_'
    modelName += 'embed%d_hid%d_li%d.h5'%(args.embedding_dim, args.hidden_size, args.li)
    modelSavePath = os.path.join(modelSavePathPrefix, modelName)
    state_dict = {name:value.cpu() for name, value in myModel.state_dict().items()}
    status = {'state_dict':state_dict,}
    with open(modelSavePath, 'wb') as f_model:
        torch.save(status, f_model)
    print('==== train finish ====')

def predict(args, myModel, dataTest, enDict, zhDict, enWord2VecDict, zhWord2VecDict, enWordDict, zhWordDict):
    print('predictModelPath: ', args.predictModelPath)
    print('predictAnswerPath: ', args.predictAnswerPath)
    print('==== predict start ====')
    status = torch.load(args.predictModelPath)
    myModel.load_state_dict(status['state_dict'])
    myModel.eval()
    if args.use_word2Vec:
        testDataset = DatasetWord2Vec(
                    args, enDict, zhDict, dataTest,
                    enWord2VecDict=enWord2VecDict, 
                    zhWord2VecDict=zhWord2VecDict, 
                    useEn=args.use_En, useZh=(not args.use_En)
                )
    else:
        testDataset = DatasetWordDict(
                    args, enDict, zhDict, dataTest,
                    enWordDict=enWordDict, 
                    zhWordDict=zhWordDict, 
                    useEn=args.use_En, useZh=(not args.use_En)
                )

    testLoader = torch.utils.data.DataLoader(
                dataset=testDataset,
                batch_size=args.batch_size,
                shuffle=False,
            )
    predictAnswers = None
    predictIds = None
    with torch.no_grad():
        for index, (sentence1, sentence2, target) in enumerate(testLoader):
            if args.use_cuda:
                predict = myModel.forward(sentence1.cuda(), sentence2.cuda())
            else:
                predict = myModel.forward(sentence1, sentence2)
            predict = torch.max(predict, 1)[1]
            if predictAnswers is None:
                predictAnswers = predict
                predictIds = target
            else:
                predictAnswers = torch.cat([predictAnswers, predict])
                predictIds = torch.cat([predictIds, target])

    relationDict = {0:'agreed', 1:'disagreed', 2:'unrelated'}
    predictAnswers = predictAnswers.cpu().numpy().tolist()
    predictIds = predictIds.cpu().numpy().tolist()
    with open(args.predictAnswerPath, 'w') as f:
        f.write('id,Category\n')
        for i in range(len(predictAnswers)):
            f.write('%d,%s\n' % (predictIds[i], relationDict[predictAnswers[i]]))
    print('==== predict finish ====')

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print('use_cuda: ', args.use_cuda)
    print('use_word2Vec: ', args.use_word2Vec)
    dataTrain, dataTest, enDict, zhDict, enWord2VecDict, zhWord2VecDict, enWordDict, zhWordDict = None, None, None, None, None, None, None, None
    dataTrain, zhDictTrain, enDictTrain = loadData(args, 'trainData')
    dataTest, zhDictTest, enDictTest = loadData(args, 'testData')
    zhDict = dict(zhDictTrain)
    zhDict.update(zhDictTest)
    enDict = dict(enDictTrain)
    enDict.update(enDictTest)
    enWord2VecDict, zhWord2VecDict, enWordDict, zhWordDict = None, None, None, None
    if args.use_word2Vec:
        enWord2VecDict = getEnWord2VecDict(args, enDict)
        zhWord2VecDict = getZhWord2VecDict(args, zhDict)
    else:
        enWordDict = getWordDict(enDict)
        zhWordDict = getWordDict(zhDict)
    if args.use_cuda:
        if args.use_word2Vec:
            myModel = RNN_Word2Vec(args).cuda()
        else:
            myModel = RNN_WordDict(args, (len(enWordDict) if args.use_En else len(zhWordDict))).cuda()
    else:
        if args.use_word2Vec:
            myModel = RNN_Word2Vec(args)
        else:
            myModel = RNN_WordDict(args, (len(enWordDict) if args.use_En else len(zhWordDict)))
    if args.usage.lower() == 'train':
        train(args, myModel, dataTrain, enDict, zhDict, enWord2VecDict, zhWord2VecDict, enWordDict, zhWordDict)
    else:
        predict(args, myModel, dataTest, enDict, zhDict, enWord2VecDict, zhWord2VecDict, enWordDict, zhWordDict)
    

# scp main.py model.py util.py args.py b05902109@cml10.csie.ntu.edu.tw:/tmp2/b05902109/NLP2019SPRING/
'''
python3 main.py --use_word2Vec=True --use_En=True --cuda_device=1 --usage=train --predictModelPath=../NLP_data/model --embedding_dim=100 --hidden_size=50
python3 main.py --use_word2Vec=False --use_En=True --cuda_device=3 --usage=train --predictModelPath=../NLP_data/model --embedding_dim=100 --hidden_size=50
python3 main.py --use_word2Vec=False
epoch: 29, loss:0.721, valid:0.769
python3 main.py --use_word2Vec=False --use_En=False
epoch: 29, loss:0.715, valid:0.754
'''