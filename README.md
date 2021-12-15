# K nearest neighbours implementation for GTZAN dataset music genre classification

This is a group project for CPS803 (Machine Learning). 3 models were implemented including Logistic Regression, Neural Networks, and K Nearest Neighbours. I implemented the entirety of the KNN model.


# Problem Statement: 
The problem faced is to classify GTZAN dataset song features into 1 of the 10 music genres. For example, given a list of 59 features from an unclassified song, applying machine learning models is essential to accurately sort the song into its proper genre. This process consists of preprocessing the dataset and then splitting the already feature-selected dataset into training, testing, and validation datasets. Next, machine learning techniques will be applied to evaluate and train the applied models to classify each features list with high accuracy. Extra evaluation techniques such as error, accuracy, precision, f1, recall scores and confusion matrices, along with various hyperparameter finetuneing techniques will be used on the data to analyze and upgrade the performance of the models in detail. 

# Dataset: 
We will be using the GTZAN music classification dataset from Kaggle. It consists of a collection of 10 genres with 100 audio files each (with a total of 1000 audio files). The 10 genres are; Blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock. All tracks are 30 second 22050 Hz Mono 16-bit audio files in .wav format. The  dataset also contains audio files split into 3 second intervals to generate more data for training. Additionally, a visual representation of audio files represented as Mel Spectrogram and 2 CSV files containing features of the audio files are also present in our dataset. 

# Preprocessing: 
Since this particular GTZAN dataset contains all the features required for the classification, we will not be extracting any features directly from audio files. The dataset has 60 features from the audio signals present in the audio recordings of the dataset.
Since most models do not work with semantic training labels (e.g. “blues”) , we will transform them using one-hot encoding. The method can help remove implicit assumptions about each class in relation to other classes, such as two nearby values being more similar. We apply a min-max scalar to standardize the data values of all numerical values from the range of 0-1. We will split the data into a ratio of 70:20:10 % for training, testing, and validation sets, respectively. We use the validation set to tune the hyperparameters during our analysis and the test set to evaluate the final performance of the model. 

# Model Evaluation and Finetuning: 

The metrics we used implement functions that assess model prediction error and success, along with creating visualizations to get a better understanding of model performance through training and finetuning. The following performance metrics are used to evaluate and compare model performance: 

- Accuracy score
- Error score
- Precision and Recall
- F1 score 
- Confusion matrix 

To determine the overfitting or underfitting within the models, accuracy and error scores on the training, testing, and validating data sets are very important. The error scores are essential in showing bias and variance errors and to make decisions when dealing with bias variance trade offs. The confusion matrix is used to predict the accuracy of the model identification of  the data and its correct genre. Graphs are plotted to provide a visualization of accuracy and error trends for each model. By finding the precision score, we can clearly see the quantitative existence of true positive and false positive predictions; for recall score, we can ascertain true positive and false negative predictions. Next, with the f1 scores, it would take both precision and recall scores into the equation and provide a mean where model performance gets better as the score is closer to 1, and worse when the score is near 0. 

As seen below, the accuracy and error scores for the baseline model were low. Specifically, the testing dataset has an error rate of 3% while the validation dataset has 6%. The difference between the scores shows that the model is overfitting and high in variance (and low in bias). This means that there will likely be significant changes to the model’s predictions with changes to the training dataset. 

![image](https://user-images.githubusercontent.com/59906096/146260746-a37e7b46-7dc0-4196-ae69-9f069a3b8b24.png)

![image](https://user-images.githubusercontent.com/59906096/146260772-c1e31cb0-60b4-4376-85d3-cd127966c133.png)

In order to reduce the high variance, we decided to change the K value to its optimal value, as a way of fine tuning the hyperparameters. The grid search function was used to determine the best metric, K value (number of neighbours), and weights that resulted in the highest accuracy of the model. The results of the grid search are shown below, along with a much higher accuracy score as well improving from 76% to 93%:

![image](https://user-images.githubusercontent.com/59906096/146260822-9b6f0510-9bff-40ba-8f9d-b324870d70e6.png)

# Future Changes: 
Going forward, we would appreciate the opportunity to integrate our work into an actual application. There are many uses for models that deal with music genre classification. In particular, we would like to create an application that has the ability to recommend music to users based on the genres they already listen to frequently. For this goal, a model that can efficiently categorize a piece of music based on its characteristics is extremely useful. Moreover, if our project were to be continued, we would try out more complex models such as CNN (convolutional neural networks) and SVM (support vector machine). We would also do some of our own in-depth exploratory analysis of different music samples instead of just relying on datasets created by others. This would help us do our own feature extraction and decide what key features of the data to operate on. It would also allow us to include a large number of genres, such as the currently very popular Korean pop genre. Finally, we would consider doing feature refining in order to maximize the efficiency and accuracy of the models we use.
