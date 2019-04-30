import argparse
import model
'''
pathPrefix
├ data         (data Folder)
├ pretrain     (pretrain Folder)
├ model     (model Folder)
└ predict     (predict Folder)

'''
def get_args(train=True):
    parser = argparse.ArgumentParser()
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
            default='enWord2VecModel_dim=%d.model',
            help='The English word2vec dictionary.')
    parser.add_argument('--zhWord2VecDict',
            default='zhWord2VecModel_dim=%d.model',
            help='The Chinese word2vec dictionary.')
    # About model Folder
    parser.add_argument('--modelName',
            default='modelRNN_paddingLen=%3d_embedDim=%3d_hiddenSize=%3d_b=%3d_e=%2d_lr=%.3d.model',
            help='The save name of the model.')

    if train:
        # About model argument
        parser.add_argument('--padding_len',
                default=30,
                type=int,
                help='The padding length of sentence, word as unit.')
        parser.add_argument('--embedding_dim',
                default=200,
                type=int,
                help='The word2vec embedding dimension of words. Same as input_size.')
        parser.add_argument('--hidden_size',
                default=200,
                type=int,
                help='The hidden size of model.')
        parser.add_argument('--dropout',
                default=0.0,
                type=float,
                help='The dropout rate for RNN')
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
                default=32,
                type=int,
                help='The batch size for training.')
        parser.add_argument('-e', '--epoches',
                type=int,
                default=10)
        parser.add_argument('-lr', '--learning_rate',
                type=float,
                default=0.0005)
        parser.add_argument('--save_intervals',
                default=10,
                type=int,
                help='The epoch intervals to save models')
    else:
        # About predict folder
        parser.add_argument('--predictFileName',
                default='ans.csv',
                help='The predict file name.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args(False)