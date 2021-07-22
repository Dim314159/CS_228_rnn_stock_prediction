# CS_228 currency pair Trend prediction
# Deep Learning for Currency Pair Prediction

**Dmitri Koltsov**^1 **, Amirsadra Mohseni**^2 **, Linxuan Liu**^3
University of California, Riverside
{^1 dkolt 002 ,^2 amohs 002 ,^3 lliu 163 }@ucr.edu

## 1 Introduction

### Background

Acurrencypairisthequotationoftwodifferentcurrencies,withthevalueofonecurrencybeingquotedagainstthe
other.Inthisreport,weutilizesixmajorcurrencypairsincludingEUR/USD,GBP/USD,USD/JPY,EUR/GBP,EUR/JPY,
GBP/JPYasourdatasets,focusingonthecurrencypairdataretrievedin1-minuteintervals.Theoutputofourmodelsis
whetheracuriouscurrencytradershouldbuy,sell,orholdatthisparticularmomentintime.Wewouldliketoremindour
readersthatthisprojectisonlymeanttoserveasaproofofconceptandisnotmeanttobedeployedinareal-life
scenario involving valuable assets.

Ourmodelsusethedatafromtheyear 2017 topredicttheyear2018,theyear 2018 topredict2019,and 2019 to
predict 2020 usingthreemachinelearningmethodsoptimizedfortimeseriespredictionincludingavanillaRecurrent
NeuralNetwork(RNN),Long-Short-Term-Memory(LSTM)[1],andGatedRecurrentUnits(GRU).Thecodeweusedto
producetheseresultsisfreelyavailable,inthePythonlanguage,here:[ **GITHUBLINK** ]Allmodelsareimplementedusing
the PyTorch [2] library.

### Related Work

Galeshchuket.al[3],giveabriefoverviewofpredictionproblemsandsomemethodstosolvethatsuchaseconometric
models,timeseriesmodels,ANN,andDNN.Nguyen[4]mentionsseveralregressionmodelscanbeusefulindeep
learningandcontributetoourproject.Vukovic[5]usesnaturalnetworkmodelsforpredictiononUSD/EURcurrency
pairs.Healsoprovidessomeadvancedideasaboutdatacollectionanddeeplearningmodelsthatyieldgoodresultsin
prediction.

## 2 Materials and Methods

### Data Acquisition

Inthisreport,weusetheForexhistorycurrencypairdataretrievedat1-minuteintervalsobtainedfromHistData[6]asour
dataset.Wehypothesizedthatusingonlytheyear 2020 topredictthecurrencytrendswillprovetoosensitiveandthe
resultsmaynotbesignificantenough.Therefore,wetrainourmodelsonthedatafrom 2017 topredict2018,then 2018
topredict 2019 and,finally, 2019 topredict2020.Eachdatafilecontainsthetimestampandratioforagivencurrency
pair.Wecombineddatamatchingtimestamps.Ultimately,wehave 4 datasetsforeachyear2017,2018,2019,and2020.
Each dataset has the following structure:

- Columns represent all six currency pairs: EUR/USD,GBP/USD, USD/JPY, EUR/GBP, EUR/JPY, GBP/JPY
- Each row represents the currency pairs’ ratio fora given timestamp separated by every 1 minute


### Preliminaries

Fig 1 ShowsthecurrencypairEUR/USDovertheyear 2017 andfig 2 showsthepriceofEURversusUSDfromJanuary
through the middle of February of the same year.

Fig 3 showsthesamepricepairforthefirst 10 daysof2017.Observe
thattherearesmallertrendswithinthelong-termtrendsandthateven
thougha long-termtrendmayincrease,smallertrendsmayfluctuate
within this main trend. We assume that long-term trends (months)
dependon“outofmarket”situationssuchaseconomy,politics,natural
disasters, etc.In contrast, short-termtrendsdependon “insidethe
market”situations.Whentradersbuyparticularstocksmore,theprice
rises,andwhentheysellthosestocks,thepricefalls.Thisassumption
is not far fromthetruth,thoughoversimplifiedforthisproject.Our
predictingstrategyisbasedonthissimplifyingassumption.Weanalyze
market behavior andtry topredict short-term trends. Inthe stock
market,sucha strategyis called“shaving”or“scalping.”Thisisnotwhatprofessionalbrokersuse,however,itis
employed extensively by beginner individual traders.

### Predicting Strategy

Twooftherequiredhyper-parameterstotrainourmodelsaretheperiodforwhichwearegoingtopredictatrend,called
“predictionlength,”andourtargetdifferenceofprice,called“pricejump.”Foreachtimestamp,welookinthefuturefor
thedurationof“predictionlength”andcheckifthepricegoesupordownforthe“pricejump”withinthisperiod.We
createatargetvariable“y”forourdatasetsbasedupontheseobservations.Ifthepricegoesbelowthethresholdof
“pricejump,”wepredict“sell.”Ifthepriceincreases,wepredict“buy.”Ifthepricestayswithinthe“pricejump”limitsfor
“prediction length,” we predict “hold.”

Accordingly, our one-hot-encoded target variable “y”is illustrated as follows:
“sell”: [1, 0, 0] “hold”: [0, 1, 0] “buy”:[0, 0, 1]

Toevaluatethemodel,wecouldnotsimplycompare“ŷ,”ourprediction,and“y,”thetrueeventbecauseweremain
indifferentifthemodelcorrectlypredicts“hold”sincethispredictiondoesnotyieldanyprofitorloss.Thesolutionthenis
tocompare“ŷ”and“y”andcount“wrong”ifwepredict“buy”whenwehaveto“sell”orviceversa.Conversely,we
count “right” if we predict “buy” or “sell” correctly.

Example of using the model:

1. Define 2 hyper-parametersexperimentally:"pricejump"and"predictionlength."Thesearedifferentdepending
    on the currency pairs


2. Atanytimestamp,themodelpredictswhattodo.Forexample,themodelsays"buy."Webuyafixedamountof
    units of EUR/USD pairs
3. Now we anticipate one of the 3 following events:
    a. Price goes up for "price jump"
    b. Price goes down for "price jump"
    c. Neither of the previous two happen during "predictionlength"
4. Whichever happens first – we sell.

Ifweguessright,weprofitbya **fixedprice** ,ifweguesswrong,welosethesame **fixedprice** .Ifthepricedoesnotgoup
or down, we would sustain, approximately, no financialgain or loss^1.

## 3 Experiments

For training and evaluating we use the following hyper-parameters:

```
● Sequence length – How deep back we are going to performback-propagation
● Prediction length – The period for which we are goingto predict trends
● Price jump – The price difference between "buy" and"sell" points
● Hidden size – The dimension of the hidden layer
● Number of hidden layers
● Learning rate
● Epochs
● Batch size
```
After creating the "y" variable as mentioned before,we may start the learning and evaluation processes.

### RNN

First we designed an RNN (many to many) with followingparameters:
● Sequence length = 1000
● Prediction length = 10000
● Price jump = 0.
● Hidden size = 42
● Number of layers = 3
● Learning rate = 1e-
● Epochs = 2
● Batch size = 1

Initially,wechooseasmallpricejumpjusttocheckhowthemodelworks.Wetrainedthemodelonthedatasetsfor
three years: 2017, 2018, and 2019 and tested it on2020. Results:

```
● Right: 112749
● Wrong: 75682
```
Wedecidedtouseadifferentapproach.Wetrainedthemodelon 2017 andtestedon 2018 andperformedthesame
routinetotrainthemodelonthedatasetforoneyearandtestonthenext.Theresultsweremuchworse."Right"and
"wrong" were approximately even. Sometimes "right"was higher, and sometimes "wrong" was higher.

(^1) It is true that whenever we buy or sell, we may haveto pay some fees. There are also margins etc. butas we mentioned before, we would like to disregardthese costs and
instead explore theoretical possibilities of our machinelearning models.


Sowedecidedtochangesomeofthehyperparameters.Weincreased"sequencelength","predictionlength"and"price
jump" by a factor of 2:

```
● sequence length = 2000
● prediction length = 20000
● price jump = 0.
```
Wehypothesizedthatmaking"pricejump"moreprominentwillenableourmodeltodiscriminatemoreconfidentlyand
increasing"sequencelength"willmakethemodelmoresensitivetopastevents.Thesenewparametersimprovedour
results:

```
Training on:
Testing on:
```
#### 2017

#### 2018

#### 2018

#### 2019

#### 2019

#### 2020

```
Right 78409 19248 82129
```
```
Wrong 50739 15141 61335
```
We considered these experiments sufficient and movedon to the next model using the same hyperparameters.

### LSTM

LSTMisanimprovementofRNN.Briefly,LSTMdealswithpasteventsbetterthananRNN,avoidingtheproblemof
vanishing gradient descent. Hence,to improveour predictionand exploreother architectures as an educational
endeavor,weoptedtomodelanLSTM.Weusedthesamehyper-parametersweusedforthefinalwell-performingRNN.
The "rights" and "wrongs" were approximately equaland in some cases not in our favor:

```
● Right: 71828
● Wrong: 75003
```
WeconcludedthatLSTMtakespasteventsintoaccountbetterthanRNNandperhapsthata"sequencelength"of 2000
wastoohighforRNNbecauseofthevanishinggradientproblem(weusedsigmoidasnon-linearity).Itmayhavebeen
thecasethattheRNNcouldnot"remember"sofarback,andtrulyusedasmaller"sequencelength"thanwespecified.
We decided to reduce the "sequence length" of ourLSTM to 512 and the results became slightly better:

```
Training on:
Testing on:
```
#### 2017

#### 2018

#### 2018

#### 2019

#### 2019

#### 2020

```
Right 86002 17973 79703
```
```
Wrong 75113 12146 66151
```
These results led us to believe that we may be onthe right track.

Afterexperimentingwithdifferent"sequencelengths"weconcludedthata"sequencelength"of 256 or 512 givesus
approximately equal and best results among other "sequencelength" choices.


### GRU

GRUismuchsimilarinfunctionalityasanLSTM,onlythatituses 2 gatesinsteadof 3 andthereforeiscomputationally
more efficient and trains faster. However, whetherit is better or worse than LSTM is still a subjectof debate.

We used different "sequence length" = 128, 256, 512and 1024. We get the best results with "sequencelength" = 256:

```
Training on:
Testing on:
```
#### 2017

#### 2018

#### 2018

#### 2019

#### 2019

#### 2020

```
Right 98438 28777 106069
```
```
Wrong 76710 14488 80696
```
## 4 Conclusions

Weused 2 typesofRNN:"manytomany"and"manytoone.""Manytomany"producedbetterresults.Wepresumethis
isbecauseweusedrandomlysampledsequences(sincefullytraining"manytoone"takesmuchmoretime).Inother
words, we trained "many to one" on much fewer samplesthan "many to many."

WeusedRNNmanytomany,RNNmanytoone,LSTM,andGRU,andbetweenthese,GRUandRNNproducedbetter
and more stable results. We can use our model to decidewhether to buy or sell EUR/USD currency pairs.

### Testing

Thereisa possibilitythat ourmodeliscoincidentally rightforthegivendata.Aswementioned,weuseddifferent
hyper-parameters in different cases and picked themexperimentally.
Fortestingpurposes,wepickedthehyper-parameterswhichgivesusbetterperformanceingeneralamongallmodels,
andusedthemondifferentcurrencypairswithoutanychangesortuning.WeusedEUR/GBPandGBP/USDandyou
can see the results below:

#### EUR/GBP

#### RNN

```
many-to-many
```
#### RNN

```
many-to-one
```
#### LSTM GRU

```
17-18 Right 24637
Wrong 13863
```
```
Right 21143
Wrong 14907
```
```
Right 15740
Wrong 18992
```
```
Right 27372
Wrong 14093
```
```
18-19 Right 61299
Wrong 48633
```
```
Right 52642
Wrong 43277
```
```
Right 58515
Wrong 52813
```
```
Right 63851
Wrong 54709
```
```
19-20 Right 52509
Wrong 51153
```
```
Right 54451
Wrong 47379
```
```
Right 54451
Wrong 47379
```
```
Right 66360
Wrong 52358
```

#### GBP/USD

#### RNN

```
many-to-many
```
#### RNN

```
many-to-one
```
#### LSTM GRU

```
17-18 Right 167648
Wrong 136252
```
```
Right 72185
Wrong 91795
```
```
Right 86890
Wrong 98574
```
```
Right 166273
Wrong 133532
```
```
18-19 Right 138612
Wrong 111615
```
```
Right 138612
Wrong 90078
```
```
Right 106753
Wrong 90303
```
```
Right 123373
Wrong 108532
```
```
19-20 Right 173447
Wrong 130161
```
```
Right 117130
Wrong 87774
```
```
Right 118130
Wrong 86774
```
```
Right 144063
Wrong 117598
```
As we can see, the results are less stable for allbut RNN (many to many) and GRU.

Modelsweretunedfordifferentcurrencypairsandeachcurrencypairhasitscharacteristics.Thesituationisnotlike 1
dollar= 1 euro= 1 poundandsoon.Thisiswhy"pricejump"shouldbechosenindividually.Furthermore,different
currenciesmayhavedifferentdynamicsthataffecttheirpricesaccordingtotheircountries"economicstructures.Some
economiesmayreactfaster,somemayhave"amortization"mechanismssotheircurrenciesreactslower.Thus,wemay
need to choose different "sequence lengths" and "predictionlengths" for different pairs.

Ultimately, ourtestresultsshowthatRNN, LSTM,andGRUmodelsthatarenot exhaustivelytunedforparticular
currencypaircanstillprovidepositiveresults.Thissupportsourassumptionthatshort-termtrendscanbepredicted
based on price history.
Instructions of using the model would be:

1. Gover more years of data.
2. Choose a currency pair.
3. Tune hyper-parameters for this currency pair uningtraining and validation sets.
4. Test the model on a test set.
5. Use the model in real time to predict this currencypair behavior on the market (make some profit).
Again,inthisprojectwearenotcreatingoptimizedmodelforuseinthemarket,weareexploringpossibilityforrecurrent
neural networks predict trends based on price history.

### Supplementary Experiments

Stockmarketbehavesdifferentlyrightafteropeningandrightbeforeclosing(weekends,holidays). Wemodifiedourdata
and included feature indicated periods before openingand closing.
We also experimented with different prediction lengths.
Prediction length = 1024

```
Training on:
Testing on:
```
#### 2017

#### 2018

#### 2018

#### 2019

#### 2019

#### 2020

```
Right 12 0 186
```
```
Wrong 0 0 12
```
Witha shortpredictionperiodthemodelbecamemuchmorecautious,predicting holdmoreoften,butalsomore
accurate.


Prediction length = 4096

```
Training on:
Testing on:
```
#### 2017

#### 2018

#### 2018

#### 2019

#### 2019

#### 2020

```
Right 13886 487 16779
```
```
Wrong 7409 56 9942
```
Naturally by increasing prediction length more chancesfor price change it’s value by ‘price jump’.

Wenoticedthatduringthetrainingprocesslossstoppedreducingatsomepoint.Sowetriedtostoptrainingwhenloss
stops reducing. Results became much worse. We concluded that with more data despite training loss stays
approximately the same model still keeps trainingand adjusting and validation accuracy improves.

## 5 References

[1] "Long short-term memory." _Wikipedia_ , en.wikipedia.org/wiki/Long_short-term_memory.Accessed 30 May 2021.

[2] _PyTorch_. pytorch.org

[3]Galeshchuk, Svitlana, and Sumitra Mukherjee. "Deeplearning for predictions in emerging currency markets."
_International Conference on Agents and ArtificialIntelligence_. Vol. 2. SCITEPRESS, 2017.

[4]Nguyen, Andrew. "Exchange Rate Prediction: MachineLearning with 5 Regression Models." _towards datascience_ ,
Medium, 20 May 2020, towardsdatascience.com/
exchange-rate-prediction-machine-learning-with-5-regression-models-d7a3192531d.Accessed 27 Apr. 2021.

[5]Vyklyuk, Yaroslav, Darko Vukovic, and Ana Jovanovic."Forex prediction with neural network: USD/EUR currencypair."
_Актуальні проблеми економіки_ 10 (2013): 261-273.

[6] _HistData_. https://www.histdata.com/. Accessed27 Apr. 2021.


