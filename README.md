# Celebrity-profiling

   The Celebrity Profiling task is to predict four traits of a celebrity from their social media communication. I worked on logistic regression
model,Naives bayes ,lstm classifier model to classify the celebrities on the basis of following features.
 

* degree of fame
* occupation
* age
* gender

   The social media communication is given as the teaser messages from past tweets. The goal is to develop a piece of software which predicts celebrity traits from the teaser history.

# Data
The training dataset contains of two files: a feeds.ndjson as input and a labels.ndjson as output. Each file lists all celebrities as JSON objects, one per line and identified by the id key.

# Input Format
The input file contains the cid and a list of all teaser messages for each celebrity.

    {"id": 1234, "text": ["a tweet", "another tweet", ...]}
    {"id": 5678, "text": ["a tweet", "another tweet", ...]}
    ...
feeds.ndjson

# Output Format
The output file contains the cid and and a value for each trait for each celebrity from the input file.

    {"id": 1234, "fame": "star", "occupation": "sports", "gender": "female", "birthyear": 2002}
    {"id": 5678, "fame": "rising", "occupation": "professional", "gender": "male", "birthyear": 1990}
    ...
labels.ndjson

The following values are possible for each of the traits:

    fame        := {rising, star, superstar}
    occupation  := {sports, performer, creator, politics, manager,
            science, professional, religious}
    birthyear   := {1940, ..., 2012}
    gender      := {male, female, nonbinary}
    
   We use 3 models to distinguish the features of the celebrity.Those are
   
   * Logistic Regression model
   * Naive Bayes model
   * LSTM classifier model
  
  we take the accurate one out of these models for  each of the trait prediction.
  
  # Logistic regression model
  
  This model uses TF-IDF matrix along with logistic regression classifier for multi-class and binary classification tasks and obtained good accuracy.
  
   Logistic regression is the best model for classification problem.Instead of fitting a straight line or hyperplane, the logistic regression model uses the logistic function to squeeze the output of a linear equation between 0 and 1. The logistic function is defined as:
   
   logistic(n) = 1 / 1 + exp(-n)


And it looks like this:

![Alt text](https://saedsayad.com/images/LogReg_1.png)


For classification, we prefer probabilities between 0 and 1, so we wrap the the equation into the logistic function. This forces the output to assume only values between 0 and 1. This model gave pretty good results compared to other models in gender prediction.

# Run code
      python celeb_profiling.py
      

# Naive Bayes model

Naive Bayes is a very simple but powerful algorithm used for prediction as well as classification. It follows the principle of “Conditional Probability".

We use this formula to calculate the probability of an event occuring.

![Alt text](https://analyticsprofile.com/wp-content/uploads/2019/06/1-1.jpg)

# Training and Testing
      from sklearn.naive_bayes import MultinomialNB 
      model = MultinomialNB().fit(X_train, y_train)  #Train the model on the training data 
      y_pred = model.predict(X_test) #Test the model on the testing data and comparing the result with the actual target
      
This gave much powerful results in Birth Year prediction.This model is best fit for multi-class problems.

# LSTM Classifier


![Alt text](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
These are the layers of my lstm model for gender prediction.

* Layer (type) 
* inputs (InputLayer) 
* embedding_2 (Embedding) 
* lstm_2 (LSTM) 
* FC1 (Dense) 
* activation_4 (Activation)
* out_layer (Dense) 
* activation_5 (Activation)


For gender prediction, i used Embedding layer and  activation layer of relu and sigmoid functions.Since it is a binary classification I used binary_crossentropy loss method with adam optimizer.

For,occupation,fame and birthyear prediction after tokenizing i applied lstm model(with some layers modified) to it with ‘softmax’ activation.

Since,all these are multi-class classification methods I used categorical_crossentrophy loss method with adam optimizer.

# Results and Analysis

Logistic Regression model is the first basic model i used for classification.I tried using naïve bayes classifier which also gave good results.Lstm model gave the following results on gender prediction and worked quite better than many other models. 
  
                        LOG REG         NB           LSTM
      Gender            82.50          72.38       77.30
      Occupation        68.94          45.49       69.36
      Fame              66.54          74.95       75.08
      Birth Year        51.47          50.07       51.60    
      


# Reference

https://pan.webis.de/clef19/pan19-web/celebrity-profiling.html
