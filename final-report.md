---
title: "Final Project"
author: Dave Miller
date: April 28, 2021
output: pdf_document
export_on_save:
    pandadoc: true
---

# CSCI 347 Final Project
* Dave Miller
* April 28, 2021

## Introduction

### The problem
The problem I am trying to solve is the task given by (https://numer.ai/). This consists of using clear, obfuscated stock market data to predict a discrete, real-valued target between 0 and 1 that gives the Numerai hedge fund an indication of how to trade the next week. Numerai works by providing the high quality data and crowd sourcing the predictions to a bunch of data scientists in order to make what they call a 'meta-model'. Thus, the task at hand is to test out some different machine learning models and evaluate them according to some well known regression metrics (MSE, MAE) as well as some of the metrics they use on the site (correlation, feature neutral correlation, sharpe). Some of these I have never heard of and had to look up how they are calculated.

### The data 
The data is regularized to values between 0 and 1 and consists of 310 discrete, numeric features, an id, and a time era corresponding to when the data came from. There are ~500,000 training samples and ~1.7 million tournament samples, with a portion of those being used for validation (they have a target attribute). There are no missing values. Since the features have names like 'charisma' and 'intelligence', I am unable to do any sort of feature engineering and I decided that any feature extraction / dimesionality reduction methods would be too computationally expensive. Therefore, pre-processing is minimal and somewhat unnecessary. 

## Methods

### Metrics
For this project, I had to go into depth with quite a few metrics that were completely new to me since they are used to score the model on the site. Here, I will give an outline of each of them.

##### Mean Squared Error (MSE)
$$
MSE(prediction, target) = \frac{1}{n} \sum (prediction - target)^2
$$

##### $R^2$ Score
This is a known as the coefficient of determination, and is an indication of how well data fits a regression model. An $R^2$ of 1.0 shows that the variance explained by the model is equal to the total variance of the data. The formula itself is rather dense, so I will show the simplified one here.

$$
R^2 = 1 - \frac{Unexplained    variance}{Total  variance}
$$


##### Spearman Rank Correlation
$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}
$$
Rank correlation is a bit different from the regular correlation that we learned in class, but the way I understand it is that it is a measure of the strength of the association between two or more ranked / ordered variables. In this context, the ordered variables are the time eras given in the data that correspond to what week that each sample was pulled from the stock market. The equation is shown above, and the python code shown below.
```python
def corr(X, y):
    ranked_preds = X.rank(pct=True, method='first')
    return np.corrcoef(ranked_preds, y)[0,1]
```

##### Sharpe Ratio
Sharpe ratio is a finance term that is defined as the amount of return an investor gets per unit increase in risk. A higher sharpe ratio amounts to better return on investment for the same amount of risk. In this context, it is calculated as follows.
```python
sharpe_ratio = correlations.mean() / correlations.std(ddof=0)
```

### PCA
I was thinking that I would not use any dimensionality reduction methods, but ended up trying PCA with ```n_components='mle'```, which uses Minka's MLE algorithm to guess the dimension. It decreased the dimensionality of the dataset by 1 while preserving 99.99% of the variance, so all of the following use this PCA transformed dataset.

### Linear Regressor
First, I wanted to train a regular old linear regressor as a sort of benchmark to improve upon with more powerful ML models. It trained somewhat quickly, but the very small $R^2$ values are not promising. However, it did have a rather high test set correlation.

LR | Train | Test
--- | --- | ---
MSE | 0.049 | 0.050
$R^2$ | 0.003 | -0.001
Corr | -0.0003 | 0.015

### Random Forest
I have heard a lot of good things about random forest models as great performers in general across all tasks. I ended up using ```n_esimators=10``` since that took around 15 minutes to fit and I don't think my laptop could handle any more.

RF | Train | Test
--- | --- | ---
MSE | 0.009 | 0.055
$R^2$ | 0.802 | -0.106
Corr | 0.0047 | 0.0046

### Neural Network
Since the stock market is such a complex, chaotic system and this data is resistant to feature engineering, I thought it would be a good idea to just throw my data at a large neural network and see what happens. I decided to continue using sklearn's implementations since they are easy to understand and use. Here are the results from three different architectures using sklearn's ```MLPRegressor```.

* NN 1 - 1 hidden layer w/ 100 nodes
This one ran for 21 iterations before being stopped by the algorithm for not improving in 10 generations. The performance is not bad, but not anything special.

NN1 | Train | Test
--- | --- | ---
MSE | 0.045 | 0.053
$R^2$ | 0.091 | -0.069
Corr | 0.0019 | 0.0072

* NN2 - 2 hidden layers w/ 50 nodes each
I was hoping this one might be able to capture some non-linear relationships that the last one missed using more hidden layers. Unfortunately, it also stopped, this time after 13 iterations. The performance is similar to the last one, if not worse.

NN2 | Train | Test
--- | --- | ---
MSE | 0.047 | 0.051
$R^2$ | 0.054 | -0.030
Corr | 0.0015 | 0.0068

* NN3 - 1 hidden layer w/ 200 nodes
On a hunch I decided to try one more architecture, reasoning that the network was losing too much information trying to encode the 309 dimensional input array into 50-100 nodes. This one performed the best, stopping after 28 iterations, but not by much.

NN2 | Train | Test
--- | --- | ---
MSE | 0.041 | 0.057
$R^2$ | 0.174 | -0.142
Corr | 0.0027 | 0.0065

## Results
Overall, no one method stood out and they each had weaknesses and strengths. Validation MSEs were all around 0.5, and correlations around 0.007 with the exception of the linear regressor. However, the linear regressor did perform better than I expected when compared to the other models. All of the $R^2$ values showed that it is an extremely difficult task to capture all of the variance in the data with a regression model. This is expected, as this data is highly variant and covariant. I think that the random forest and neural nets could be improved substantially with more computing power. Also, feature engineering using the era attribute could also improve these models since this is, in a way, time series data. For comparison, the [numerai](https://numer.ai) leaderboards show correlations around 0.05, and feature neutral correlations around 0.01. 

## Conclusion
I am mostly pleased with how this experiment turned out, but I know I could do better. Over the last week, I spent a lot of time trying to get a good development environment set up but ran into a lot of barriers because of the size of my dataset. I tried AWS and Google Collab, but they kept kicking me out of the cloud for using too much memory. I spent quite a few hours trying to get [Rapidsai](https://rapids.ai/), a GPU machine learning library to install on my machine. This one I was particularily excited about since they promise speedups of 10x - 50x, but in the end I ran out of time and had to run everything on my laptop. In the future, I think that a good development environment, lots of compute, and potentially a custom loss function to learn for correlation could yield great results. Also, since the targets are discrete, various classification methods could be used to replace or augment the models I used here.
