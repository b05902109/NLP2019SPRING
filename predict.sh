python3 main.py --usage=predict --cuda_device=1 --use_word2Vec=false --use_En=false --embedding_dim=200 --hidden_size=100 --li=32 --predictModelPath=../NLP_data/model/model_Zh_wordDict_embed200_hid100_li32.h5 --predictAnswerPath=../NLP_data/predict/ans_Zh_wordDict_embid200_hid100_li32.csv
python3 main.py --usage=predict --cuda_device=1 --use_word2Vec=false --use_En=false --embedding_dim=200 --hidden_size=100 --li=64 --predictModelPath=../NLP_data/model/model_Zh_wordDict_embed200_hid100_li64.h5 --predictAnswerPath=../NLP_data/predict/ans_Zh_wordDict_embid200_hid100_li64.csv
python3 main.py --usage=predict --cuda_device=1 --use_word2Vec=false --use_En=false --embedding_dim=100 --hidden_size=50 --li=32 --predictModelPath=../NLP_data/model/model_Zh_wordDict_embed100_hid50_li32.h5 --predictAnswerPath=../NLP_data/predict/ans_Zh_wordDict_embid100_hid50_li32.csv
python3 main.py --usage=predict --cuda_device=1 --use_word2Vec=false --use_En=false --embedding_dim=100 --hidden_size=50 --li=64 --predictModelPath=../NLP_data/model/model_Zh_wordDict_embed100_hid50_li64.h5 --predictAnswerPath=../NLP_data/predict/ans_Zh_wordDict_embid100_hid50_li64.csv
python3 ensemble.py