# TEJPredict

Using TEJ dataset to predict Net profit margin growth rate by RNN and Attention analysis

# set up

### Necessary model
```
pip install torch
pip install numpy
pip install sklearn
pip install pandas
```
#### If you want to see the complete code and test the script, you can run the code below

##### training's scrpit ( TEJPredict/trainModel )
```
python3  rnn_attention_model.py
python3 rnn_model.py
```

##### predict's scrpit ( TEJPredict/predict )
```
python3 RNN_attention_predict.py
python3 RNN_predict.py
```

##### classifier's scrpit ( TEJPredict/classifier )
```
python3 k-means.py
```

### Experimental details

| * | RNN | RNN + Attention |
|---------|---------|---------|
| Convergence speed | slow|   fast |





#### RNN (220000 epochs)
![RNN(220000 epochs)](https://github.com/ImChouOWO/TEJPredict/blob/main/img/RNN_img/RNN_losses_220000.png)


#### RNN + Attention (60000 epochs)
![RNN+Attention(220000 epochs)](https://github.com/ImChouOWO/TEJPredict/blob/main/img/RNN_attention_img/RNN_attention_losses_80000.png)


#### classified

#### top 10 in terms of frequency in dataset of top 20% 
![top](https://github.com/ImChouOWO/TEJPredict/blob/main/classifier/outputData/output/img/Top%2010%20company.png)

#### bottom 10 in terms of frequency in dataset of bottom 20% 
![top](https://github.com/ImChouOWO/TEJPredict/blob/main/classifier/outputData/output/img/Bottom%2010%20company.png)



