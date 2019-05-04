import argparse
import model
'''
pathPrefix
├ data         (data Folder)
├ pretrain     (pretrain Folder)
├ model     (model Folder)
└ predict     (predict Folder)

'''
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda',
            type=str2bool, nargs='?',
            const=True, default=True,
            help='Use cuda or not.')
    parser.add_argument('--use_word2Vec',
            type=str2bool, nargs='?',
            const=True, default=True,
            help='Use word2vec or wordDict.')
    parser.add_argument('--use_En',
            type=str2bool, nargs='?',
            const=True, default=True,
            help='Use En or Zh.')
    parser.add_argument('--use_smallData',
            type=str2bool, nargs='?',
            const=False, default=False,
            help='Use small data(8003) or all data(320552).')
    parser.add_argument('--cuda_device',
            default='1',
            help='Use which gpu device.')
    parser.add_argument('--usage',
            default='train',
            help='what to do, train or predict.')
    # About path
    parser.add_argument('--pathPrefix',
            default='../NLP_data',
            help='The path prefix. Inside is four directories.')
    parser.add_argument('--dataFolder',
            default='data',
            help='The data folder.')
    parser.add_argument('--pretrainFolder',
            default='pretrain',
            help='The pretrain folder.')
    parser.add_argument('--modelFolder',
            default='model',
            help='The model folder.')
    parser.add_argument('--predictFolder',
            default='predict',
            help='The predict folder.')
    # About data folder
    parser.add_argument('--trainData',
            default='train.csv',
            help='The csv file of train data.')
    parser.add_argument('--testData',
            default='test.csv',
            help='The csv file of test data.')
    # About pretrain folder
    parser.add_argument('--enWord2VecPretrainData',
            default='text8',
            help='The pretrain dataset for English word2vec.')
    parser.add_argument('--enWord2VecDict',
            default='enWord2VecModel_dim%d.model',
            help='The English word2vec dictionary.')
    parser.add_argument('--zhWord2VecDict',
            default='zhWord2VecModel_dim%d.model',
            help='The Chinese word2vec dictionary.')
    # About model Folder
    parser.add_argument('--predictModelPath',
            help='The model file name for predict.')
    # About predict folder
    parser.add_argument('--predictAnswerPath',
            default='../NLP_data/predict/ans.csv',
            help='The predict file name.')
    # About model argument
    parser.add_argument('--padding_len',
            default=16,
            type=int,
            help='The padding length of sentence, word as unit.')
    parser.add_argument('--embedding_dim',
            default=200,
            type=int,
            help='The word2vec embedding dimension of words. Same as input_size.')
    parser.add_argument('--hidden_size',
            default=100,
            type=int,
            help='The hidden size of model.')
    parser.add_argument('--bidirectional',
            default=True,
            help='Gru is bidirectional or not.')
    # About training argument
    parser.add_argument('--validation',
            default=True,
            help='To split validation or not.')
    parser.add_argument('--seed',
            default=7122,
            type=int,
            help='The random seed.')
    parser.add_argument('-b', '--batch_size',
            default=64,
            type=int,
            help='The batch size for training.')
    parser.add_argument('-e', '--epoches',
            type=int,
            default=30)
    parser.add_argument('-lr', '--learning_rate',
            type=float,
            default=1e-4)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args(False)
