# Celebrity-profiling

   The Celebrity Profiling task this year is to predict four traits of a celebrity from their social media communication. 

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
    
   We use 3 methods to distinguish the features
   
   *Logistic Regression model
   *Naive Bayes model
   *LSTM classifier model
  
  we take the accurae one out of these models for  each of the trait prediction.
  
  # Logistic regression model
  
  This model uses TF-IDF matrix along with logistic regression classifier for multi-class and binary classification tasks and obtained good accuracy.
  
   Logistic regression is the best model for classification problem.Instead of fitting a straight line or hyperplane, the logistic regression model uses the logistic function to squeeze the output of a linear equation between 0 and 1. The logistic function is defined as:
   
   logistic(n) = 1 / 1 + exp(-n)


And it looks like this:

![Alt text](https://saedsayad.com/images/LogReg_1.png)


The logistic function.For classification, we prefer probabilities between 0 and 1, so we wrap the the equation into the logistic function. This forces the output to assume only values between 0 and 1. This model gave pretty good results compared to other models in gender prediction.

# Run code
      python celeb_profiling.py
      

            
