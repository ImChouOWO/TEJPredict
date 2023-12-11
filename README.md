﻿# TEJPredict

Using TEJ data set to predict Net profit margin growth rate by RNN add Attention analysis

# set up

### Necessary model
```
pip install torch
pip install numpy
pip install sklearn
pip install pandas
```
#### If you want to see the complete code and test the script, you can run the code below
```
cd predict
python3 RNN_attention_predict.py
python3 RNN_predict.py
```

### Experimental details

| * | RNN | RNN + Attention |
|---------|---------|---------|
| Convergence speed | slow|   fast |





#### RNN (220000 epochs)
![RNN(220000 epochs)](https://github.com/ImChouOWO/TEJPredict/blob/main/img/RNN_img/RNN_losses_220000.png)


#### RNN + Attention (60000 epochs)
![RNN+Attention(220000 epochs)](https://github.com/ImChouOWO/TEJPredict/blob/main/img/RNN_attention_img/RNN_attention_losses_80000.png)


#### TO DO List
- [ ] Classifier  

