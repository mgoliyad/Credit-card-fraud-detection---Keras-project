# Credit-card-fraud-detection---Keras-project

This project for the following kaggle challenge. 

https://www.kaggle.com/mlg-ulb/creditcardfraud

Github does not allow to load creditcard.csv which is 150MB but you can get it directly from Kaggle.com.

 This dataset presents credit card transactions, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

To train the model I decided to split all data into three buckets train/validate/test. To have the fair representation I had to split “fraud” and “normal” transaction separately and then concatenated them.

My model was able to reach accuracy: 0.9960 on train data and 0.9983 on validation data. Prediction for 71202 test cases was 100% correct. 
