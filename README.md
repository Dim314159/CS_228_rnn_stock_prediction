# CS_228 currency pair Trend prediction



Project for CS 228, Spring 2021


Deep Learning for Currency Pair Prediction

Amirsadra Mohseni(amohs002@ucr.edu)
Dmitri Koltsov(dkolt002@ucr.edu)
Linxuan Liu(lliu163@ucr.edu)








INTRODUCTION

A currency pair is the quotation of two different currencies, with the value of one currency being quoted against the other. We take 6 major currency pairs including EUR/USD, GBP/USD, USD/JPY, EUR/GBP, EUR/JPY, GBP/JPY as our data set.
For our project, we will focus on the currency pair data retrieved in the interval of 1 minute as our data sets. The data we use is according to Forex historyi. We will use the year of 2017 data to predict the year of 2018, the year of 2018 data to predict the year of 2019, the year of 2019 data to predict the year of 2020.
We are going to train our models on one year and use this model to predict next year. Then we are going to keep training model on the next year to predict next coming year and so on.


COLLECTING AND ORGANIZING DATA

We downloaded data for each currency pair. Each data file contains timestamp and ratio for given currency pair.
We combined data matching timestamps. So, at the end we have 4 data sets for each year 2017, 2018, 2019 and 2020. Each data set has following structure:
- rows: currency pairs ratio for given timestamp (every minute)
- columns: all 6 currency pairs.
 

MODELS WE USE

We used pytorchii to implement our models.
Our models are:
- plain RNN
- LSTM
- GRU


PREDICTING STRATEGY


As you can see within long term trends there are smaller trends. And even if long term trend goes up smaller trends may go up and down within this main trend.
We assume that long term trends (months) depend on ‘out of market’ situation. Economy, politics, natural disasters… And short term trends depend on ‘inside the market’ situation. People start to buy particular stocks more and price goes high, people start to sell particular stocks more and price goes down.
This assumption is not far from the truth, though maybe oversimplified. (But after all we are not in stock marketing class)
Our predicting strategy based on this assumption. We will analyze market behavior and try to predict short term trends. In stock marked such strategy called “shaving” or “scalping”.
This is not what professional brokers use, because it doesn’t bring a lot of profit, but beginner individual traders use it quite a lot.


PREDICTING TACTICS

We choose short time period for which we are going to predict trend ‘prediction length’. We also choose our target difference of price ‘price jump’.
For each timestamp we look forward for ‘prediction length’ and check if price goes up or down for the ‘price jump’ within this period.
We create target variable ‘y’ for our data sets based on these observations.
If price goes down for ‘price jump’, means we have to ‘sell’. If price goes up, means we have to ‘buy’. If price stays within ‘price jump’ limits for ‘prediction length’ we ‘hold’.
Our target variable ‘y’ will look like:
- ‘sell’ [1,0,0]
- ‘hold’ [0,1,0]
- ‘buy’  [0,0,1]
In order to evaluate the model we could not simply compare  ‘ŷ’ and ‘y’ because we don’t care if model correctly predict ‘hold’. This prediction is not giving us any profit or loss.
We wrote function that compare ‘ŷ’ and ‘y’ and count ‘wrong’ if we predicted ‘buy’ when we have to ‘sell’ and vice versa, and count ‘right’ if we predicted ‘buy’ when we have to ‘buy’ and we predicted ‘sell’ when we have to ‘sell’. 


PROCESS

For training and evaluating we use following hyper-parameters:
sequence length – it’s how deep back we are going to do back propagation
prediction length – time period for which we are going to predict trends 
price jump – difference in the price between ‘buy’ and ‘sell’ points 
hidden size – the dimension of the hidden layer
number of layers – the amount of hidden layers we use 
learning rate
epochs
batch_size

After creating ‘y’ variable as mentioned above we can start learning and evaluating process.

RNN.
First we used RNN (many to many) with following parameters:
sequence length = 1000
prediction length = 10000
price jump = 0.01
hidden size = 42
number of layers = 3
learning rate = 1e-4
epochs = 2
batch_size = 1

For the beginning we choose small price jump just to check how model works.
We trained the model on 3 years: 2017, 2018, 2019 and tested it on 2020. Results:

right = 112749
wrong = 75682

We decided to use different tactics of evaluating. We trained the model on 2017 and tested on 2018, then we trained the model on 2018 and tested on 2019, then we trained the model on 2019 and tested on 2020.
We got results much worse. ‘Right’ and ‘wrong’ was approximately even. Sometimes ‘right’ was higher, sometimes ‘wrong’ was higher.

So we decided to change some of hyper-parameters. We increased ‘sequence length’, ‘prediction length’ and ‘price jump’ twice:
sequence length = 2000
prediction length = 20000
price jump = 0.02
We thought that making ‘price jump’ more prominent will help model better recognize difference. And also increasing ‘sequence length’ will make model more sensitive to past events.
With new parameters we had results:
training on 2017 evaluating on 2018
training on 2018 evaluating on 2019
training on 2019, evaluating on 2020
right 93046
wrong 69601
right 30443
wrong 9821
right 107923
wrong 69434

So, we decided that results are acceptable and we kept those hyper-parameters.

We also tried RNN many to one:

17-18
18-19
19-20
right 72185
wrong 91795
right 106753
wrong 90078
right 117130
wrong 87774

‘Many to many’ produced better results. Probably because fully ‘many to one’ training would take much more time and we had to use randomly taken sequences and overall we trained ‘many to one’ on much less iterations than ‘many to many’.


LSTM.
LSTMiii (Long Short-Term Memory) is an improvement of RNN. Make long story short it deals better with past events, avoiding vanishing gradient descent.
So, we decided to use LSTM to improve our prediction.
We used same hyper-parameters we used for last good working RNN:
sequence length = 2000
prediction length = 20000
price jump = 0.02
hidden size = 42
number of layers = 3
learning rate = 1e-4
epochs = 2
batch_size = 1
Results were surprisingly bad. They were approximately equal or even not in our favor:

right 71828
wrong 75003

We concluded that LSTM takes into account past events better than RNN. And maybe that ‘sequence length’ = 2000 was to high for RNN because of vanishing gradient (we used sigmoid as non-linearity), and RNN simply could not look so deep back, and truly used much less ‘sequence length’ than we gave to it.
We decided to reduce ‘sequence length’ for LSTM to 512. And results became better:
17-18
18-19
19-20
right 86002
wrong 75113
right 17973
wrong 12146
right 79703
wrong 66151

Which made us thing that we are maybe on right track.

After experimenting with different ‘sequence length’ we came to conclusion that ‘sequence length’ = 256 or 512 gives approximately equal and best results among other ‘sequence length’ choices.


GRU
GRU is doing approximately the same as LSTM, just uses 2 gates instead of 3, so it computationally more efficient. It trains faster. But is it better or worse than LSTM is still subject of debates.
Anyway we trained and used GRU on our data.
We used different ‘sequence length’ = 128, 256, 512 and 1024. Best results we get with ‘sequence length’ = 256:
17-18
18-19
19-20
right 98438
wrong 76710
right 28777
wrong 14488
right 106069
wrong 80696


CONCLUSION

We used RNN, RNN many to one, LSTM and GRU. And between these 3 models GRU and RNN produced better and more stable results.
We can use our model to decide weather to buy or sell EUR/USD currency pair. 
Example of using the model:
    1. we have to use 2 hyper-parameters: ‘price jump’ and ‘prediction length’
    2. at any timestamp the model predicts what to do. For example model says ‘buy’. We buy fixed amount of units of EUR/USD pair
    3. Now we wait for one of 3 events:
        1. price goes up for ‘price jump’
        2. price goes down for ‘price jump’
        3. neither one of previous 2 happened during ‘prediction length’
    4. Which ever happens first – we sell.
So if we guess right we win fixed price, if we guess wrong we lose same fixed price. If price will not go up or down we would stay approximately the same1.


TESTING

There is a possibility that our model just coincidentally right for given data. As we mentioned for different cases we used different hyper-parameters. And we pick them after experimenting.
For testing purposes we picked the hyper-parameters which give us generally better performance among all models. And without tuning and changing we used it on different currency pairs.
For testing we used EUR/GBP and GBP/USD. Bellow you can see results.

EUR/GBP

RNN
many to many
RNN
many to one
LSTM
GRU
17-18
right 24637
wrong 13863
right 21143
wrong 14907
right 15740
wrong 18992
right 27372
wrong 14093
18-19
right 61299
wrong 48633
right 52642
wrong 43277
right 58515
wrong 52813
right 63851
wrong 54709
19-20
right 52509
wrong 51153
right 54451
wrong 47379
right 54451
wrong 47379
right 66360
wrong 52358

 GBP/USD

RNN
many to many
RNN
many to one
LSTM
GRU
17-18
right 167648
wrong 136252
right 72185
wrong 91795
right 86890
wrong 98574
right 166273
wrong 133532
18-19
right 138612
wrong 111615
right 106753
wrong 90078
right 77266
wrong 90303
right 123373
wrong 108532
19-20
right 173447
wrong 130161
right 117130
wrong 87774
right 117130
wrong 87774
right 144063
wrong 117598

As we can see results are less stable but for RNN many to many and GRU quite good.
Models were tuned for different currency pair. And each currency pair has their own characteristics. The situation is not like 1 dollar = 1 euro = 1 pound etc. That’s why ‘price jump’ should be chosen individually. Also different currencies may have different dynamic of changing according to their countries economy structure. Some economies may react faster some may have “amortization” mechanisms so their currencies react slower. So we may need to choose different ‘sequence length’ and ‘prediction length’.
But we think testing results show us that even not tuned, general model can still perform well. Which supports our assumption that short term trends can be predicted based on price history. 
