# Sentimental_analysis using Spark
A typical large-scale data analytics task is sentiment analysis using classification, i.e., the computational 
determination of the attitudes of people towards a certain topic or item, achieved by supervised learning from data sets 
with given sentiment annotations.
Download the following data archive: 
Data set imdb_labelled.txt in this archive contains a number of single-sentence movie reviews, each labeled with “0” 
(negative sentiment) or “1” (positive sentiment).

The program builds a Linear Support Vector Machine (SVM) model using any 60% of the labeled sentences in 
imdb_labelled.txt as training data, using Spark MLlib and RDDs. 
Afterwards, the learned model is used to predict and print the labels (sentiments) of a few test movie reviews.
Accuracy testing:
 One way of estimating the accuracy of a classifier is by computing the Area Under the ROC Curve (AUROC)
The second part of the code prints the achieved AUROC. 40% of the data (i.e., the full data set without the 
training data) is used as test data for computing the AUROC
