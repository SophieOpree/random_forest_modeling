# random_forest_modeling

## Short description
Prediction of cross-buying behavior using bank customer data and applying random forest models.

## What is this project about?
This project used a random forest approach to forecast cross-buying behavior of customers of a large German bank within a time frame of six months. Therefore, N = 100,000 customers were included in a dataset to build a random forest model which was subsequently optimized. The prediction accuracy of the final model was 84.38%.
As a second step, the most important variables regarding the prediction of cross-buying behavior were identified, including income, over-draft, the number of mobile and desktop logins, customer tenure, and customer age. Direct mailing as a marketing activity, however, did not show a remarkable impact on cross-buying behavior. 

## What is part of this repo?
Within this repository, interested users can find this readme file, the raw data and the R syntax.

## What was done?
The empirical dataset contained 30 explanatory variables describing general customer demographics, transaction behavior, and some external information (e.g., city size). Most variables had to be preprocessed to be useful for further analyses. 
All clients in the dataset at hand were customers from a large retail bank who already owned a payment account at this bank. The size of the random sample was N = 100,000, implying that data of 100,000 different customers of the bank were analyzed.
All data analyses were conducted with R and RStudio.

After initial data exploration and pre-processing, the dataset was split into 80% training data and 20% validation data.
The training data were re-sampled to create a balanced training sample with an evenly distributed target variable xsell (i.e., likelihood of cross-buying was set to approximately 50%). This was done to facilitate and improve later classification.

An initial random forest model for classification with 13 independent variables and the target variable xsell was run with model parameters at the default level.

To optimize the initial model at hand, the number of variables selected at each node (mtry), the number of trees within the forest (ntree), as well as the node size (nodesize) were tuned.

Finally, to open the "black box" of machine learning, the optimized model was disentangled with respect to variable importance. 
