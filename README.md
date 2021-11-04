## Data Mining
An exploration of various data mining methods

### Projects

#### Project 1 - Exploring a dataset
I used the forest fires dataset from the UCI Machine Learning Repository for basic data mining tasks. This is a regression task to predict forest fires based on 10 numerical attributes and 2 categorical ones. I wrote functions to compute mean, covariance, standard deviation, and correlation for use on this dataset as well as functions to range normalize, z-score normalize, label encode the categorical attributes, and get the covariance matrix. 

#### Project 2 - Exploring a graph dataset
For this project I used a Wikipedia network dataset consisting of articles as nodes and the links between them as edges. I wrote functions to compute degree, cluster coefficient, betweeness centrality, and average shortest path length. Using these functions and some others from networkx, I explored the data.

#### Project 3 - Clustering
In this project, I used a leaf dataset with 15 numerical attributes and a class label. Thus, the task was to write the two most common clustering methods: k-means and DBSCAN.

#### Final Project - Numerai
For the final project, we got to choose any data mining task that was interesting and explore it thoroughly. I chose the task given by the website (https://numer.ai/), which crowdsources stock market predictions to data scientists using obfuscated data, calling it 'the hardest data science problem in the world'. In order to complete this task, I researched numerous metrics to optimize my models on, including MSE, R^2 score, spearman rank correlation, and sharpe ratio. I tried PCA to reduce the dimensionality for the sake of training, as it is an enormous data set. I ended up making 3 different models and comparing the results: a linear regressor, random forest model, and 3 different simple neural network architectures.
