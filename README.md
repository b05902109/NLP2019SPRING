# Natural Language Inference for Fake News Detection

## Introduction

- Project 1 of Natural Language Processing (NLP) course in National Taiwan Normal University
- Kaggle Competition: [link](https://www.kaggle.com/c/fake-news-pair-classification-challenge)
- Topic summary
  - Use two headlines of two news to predict whether they are related or not.
  - Headlines have two language kinds in each, Zh and En.
  - There are three classes to predict, *Agreed*, *Disagreed*, *Unrelated*.

## Installation

- python3 package install

```bash
pip3 install jieba argparse numpy gensim
```

- About install **pytorch**: go to the official website to download.
- If you want to use google pretrain English word2vec, please download the **tex8** file in their official website and put it into the **pretrain** folder. Notice that the name and the place is restricted.

## Reproduce My Work

- Download and put the **train.csv** & **test.csv** file , create a **data** folder and put into it.

- Create a **pretrain** folder.

- It will be like this

  ```
  .
  ├ data
  │	├ train.csv
  │	└ test.csv
  ├ model
  │	├ model1.h5
  │	└ model2.h5
  ├ pretrain
  │	└ .
  ├ predict
  │	└ .
  ├ script & file
  └ readme.md
  ```

  

- Run the following command to train model.

  - I have already upload the finished models in my github, so you can skip this process if you want.

  ```bash
  bash train.sh
  ```

- Run the following command to predict answers.

  - There are five csv files, and my best submission will be **submission_0.csv**

  ```bash
  bash predict.sh
  ```

## Run main.py in train and predict

- There are some parameters you have to set, and some have defaults. You can see them in the **args.py**.

  - **--usage**: train (default) or valid
  - **--use_cuda**: true (default) or false
  - **--cuda_device**: 1 as default, change to the number you want.
  - **--use_word2Vec**: true (default) or false, if true, use **word2vec** function in gensim, otherwise, **wordDict** means we use torch **nn.Embedding** to train an end-to-end word2vec in the model.
  - **--use_En**: true (default) or false, use En or Zh to train.
  - **--embedding_dim** , **--hidden_size** , **--li** are some model parameters, you have to set again when you run predict, the same as you run train.
  - **--batch_size** , **--epoches** , **--learning_rate** are some training parameters, see the default values in **args.py**.
  - **--modelSavePath** and **--predictAnswerPath** have to set when you run the usage of prediction.

- For example, to train a Zh WordDict model and predict, you have to run the following to command.

  ```bash
  python3 main.py --usage=train --cuda_device=1 \
  				--use_word2Vec=false --use_En=false \
  				--embedding_dim=200 --hidden_size=100 --li=32
  python3 main.py --usage=predict --cuda_device=1 \
  				--use_word2Vec=false --use_En=false \
  				--embedding_dim=200 --hidden_size=100 --li=32 \
  				--modelSavePath=model/model_Zh_wordDict_embed200_hid100_li32.h5 \
  				--predictAnswerPath=predict/ans_Zh_wordDict_embid200_hid100_li32.csv
  ```

  - Model will be named as **model_Zh_wordDict_embed200_hid100_li32.h5** in the **model** folder.
  - Predict will be named as **ans_Zh_wordDict_embid200_hid100_li32.csv** in the **predict** folder.
  - Notice that model parameters have to be same as you set in train when you use predict.

## More Experiment and Model Details

- See the pdf file **report_team_4.pdf**.