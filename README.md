Readme for RNN time prediction in tf.keras
==================================
- Final update: 2018 Oct 
- All right reserved @  Chanhee Jeong and Jaewook Kang 2018


## 요구사항 (코드 최종 작성 후 삭제 예정)
찬희님 RNN time prediction example code 작성을 요청드립니다. :-) 요구사항은 아래와 같습니다
- ~~tf.keras를 이용해서 실행가능한 .py 코드 작성~~
- ~~주피터 노트북이나 colab이 아닌 파이참에서 돌아가는 파일 필요~~
- ~~AWS에서 GPU로 동작하는 코드 (tf.device()필요?)~~
- ~~적절하게 readme.md 수정~~
- ~~아래에서 구분하는 6가지 파일 + alpha로 코드를 구분해서 작성~~
- ~~실행되는 파일은 `eval.py` 와 `trainer.py`.~~
- ~~간단한 데이터셋을 다운로드하는 코드가 data_loader.py에 포함되야함 (아니면 데이터셋 다운로드 방법을 readme.md에 표시)~~
- ~~어플리케이션은 [구글의 예제](https://colab.research.google.com/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb#scrollTo=qXDHTaZZGN4L)
를 그대로 따라서 해도 무방하고 `주식차트 예측` 같은 것이 가능하면 선호함~~
- 기타 질문은 언제든지 슬랙으로
- ~~코드 리뷰는 깃헙의 풀리퀘스트~~
- 작업은 develop에서 `features/branch_name`으로 브랜치 따서 작업 요망

- 프로젝트 형식 참고
    - https://github.com/jwkanggist/tf-code-pattern-lenet5
    - 참고문서: [Jaewook Kang, Tensorflow Practical Project Configuration](https://docs.google.com/presentation/d/1zyubZQKQ3tQvQppp_7ljPnWXwCNmf3UDMQhP2GBn7ng/edit#slide=id.p1)

- deadline: 10/11 목요일 11:00 (협의가능)

## About
> 아래 적절하게 수정

The aim of this repository is to introduce an exemplary TF project in practice.
We show the example with respect to [S&P500](https://github.com/CNuge/kaggle-code/raw/master/stock_data/) prediction by simpleRNN.
This project is composed of several `.py` files,
each of which undertakes single role and responsibility 
according to the OOP philosophy.

> 아래 적절하게 수정, 파일은 추가가 가능하나 아래 요소들은 꼭 필요함
```bash
- data_loader.py    : Preparing and feeding the dataset in batchwise by using tf.data
- model_builder.py  : Building a model in tensorflow computational graph.
- model_config.py   : Specifying a configulation for the model 
- trainer.py        : Training the model by importing the dataloader and the model_builer
- train_config.py   : Including a configulation for the training
- eval.py           : Evaluating the model with respect to test dataset by loading a ckpt

```


## How to Run
Training
```bash
python ./tf_module/trainer.py
```

Inference
```bash
python ./tfmodule/eval.py
```

## Components
> 아래를 적절하게 수정

```bash
./tfmodule/
├── data
│   └──  all_stocks_5yr.csv
├── data_loader.py
├── eval.py
├── model_builder.py
├── model_config.py
├── train_config.py
└── trainer.py
```

### Compiler/Interface Dependencies
> 아래를 적절하게 수정
- Tensorflow >=1.9
- Python2 <= 2.7.12
- Python3 <= 3.6.0
- pandas >= 0.23.0
- numpy >= 1.14.5
- matplotlib >= 3.0.0


### Related Materials
- [Kaggle S&P500 Stock Data](https://www.kaggle.com/camnugent/sandp500/)

# Feedback 
- Issues: report issues, bugs, and request new features
- Pull request
- Email: jwkang10@gmail.com; chjeong530@gmail.com

# License
- Apach License 2.0


# Authors information 
- Jaewook Kang Ph.D.
    - Personal website: [link](https://sites.google.com/site/jwkang10/)
    - Facebook : [link](https://www.facebook.com/jwkkang)
    - Linkedin : [link](https://www.linkedin.com/in/jaewook-kang-3a4217b9/)

- Chanhee Jeong M.S.
    - Facebook : [link](https://www.facebook.com/Cris.Jeong)
    - Linkedin : [link](https://www.linkedin.com/in/chanhee-jeong-711842107/)
