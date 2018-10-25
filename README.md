Readme for S&P500 stock prediction in tf.keras
==================================
- Final update: 2018 Oct 
- All right reserved @  Chanhee Jeong and Jaewook Kang 2018


## About
![alt text](https://github.com/jwkanggist/tf-keras-rnn-time-pred/blob/master/images/compare.png)


This repository is to provide a toy stock price prediction in `tf.keras`.
For the dataset, we have used the 5 years (`1255 days`) [S&P500 data](https://github.com/CNuge/kaggle-code/raw/master/stock_data/),
where 
- `1004` days data is used for the model training, and 
- the remaining `251` data is for the model testing.

We consider single layer RNN-like model for the problem.
Various options are available to compose the layer from  `tf.keras`:
- Simple RNN model: `tf.keras.layers.SimpleRNN`
- LSTM model: `tf.keras.layers.LSTM` 
- GRU model:  `tf.keras.layers.GRU`

where the model shape in `the training` as 
Input X: 
- A sequence of the `close value` in past 3-days 
- X: [X_0:X_1003] has its shape as [1004 x 3 ] where the input of each  cell  is X_t = [1 x 3]

Output Y: 
- A prediction of the `close value` of the next day
- Y has its shape as [1004 x 1] where the output of each  cell is Y_t = [1]

Hidden state of each cell H:
- H_t denote the hidden state of each RNN cell which has its shape [64 x 1]


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
