import json
import csv
import re
import string
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


#regular expressions
text_re = re.compile("[^a-zA-Z\s]")
url_re = re.compile("http(s)*://[\w]+\.(\w|/)*(\s|$)")
hashtag_re = re.compile("[\W]#[\w]*[\W]")
mention_re = re.compile("(^|[\W\s])@[\w]*[\W\s]")
smile_re = re.compile("(:\)|;\)|:-\)|;-\)|:\(|:-\(|:-o|:o|<3)")
emoji_re = re.compile("(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])")
not_ascii_re = re.compile("([^\x00-\x7F]+)")
time_re = re.compile("(^|\D)[\d]+:[\d]+")
numbers_re = re.compile("(^|\D)[\d]+[.'\d]*\D")
space_collapse_re = re.compile("[\s]+")

#parameters
MAX_WORD_FEATURES = 10000
MAX_TWEETS_PER_USER = 10000


def _preprocess_feed(tweet: str):
 
    t = tweet.lower()
    t = re.sub(url_re, " <URL> ", t)
    t = t.replace("\n", "")
    t = t.replace("#", " <HASHTAG> ")
    t = re.sub(mention_re, " <USER> ", t)
    t = re.sub(smile_re, " <EMOTICON> ", t)
    t = re.sub(emoji_re, " <EMOJI> ", t)
    t = re.sub(time_re, " <TIME> ", t)
    t = re.sub(numbers_re, " <NUMBER> ", t)
    t = re.sub(not_ascii_re, "", t)
    t = re.sub(space_collapse_re, " ", t)
    t = t.strip()
    return t


def inp_f():   
    with open('feeds.ndjson',encoding='utf-8') as f:
        fc=f.read()
        fc1=fc.replace('}\n{','},{')
        data=json.loads(f'[{fc1}]')


    with open("input.csv",'w',encoding="utf-8",newline='') as csvf:
        csv_writer=csv.DictWriter(csvf,fieldnames=["id","tweet"])
        csv_writer.writeheader()
        for i in range(len(data)):
            csv_writer.writerow({"id":data[i]['id'],"tweet":data[i]['text'][:200]})

    csvf.close()
    
    
def out_f():
    with open('labels.ndjson',encoding='utf-8') as lf:
        lfc=lf.read()
        lfc1=lfc.replace('}\n{','},{')
        data1=json.loads(f'[{lfc1}]')
        with open("input.csv",encoding="utf-8",newline='') as csvf:
            csv_r=csv.reader(csvf)
            with open("label.csv","w",encoding="utf-8",newline='') as lb:
                csv_writer=csv.DictWriter(lb,fieldnames=["id","occ","gen","fame","by"])
                csv_writer.writeheader()
                for cid in csv_r:
                    if cid[0]!='id':     
                        #print(cid[0])

                        for i in range(len(data1)):
                            if str(data1[i]["id"])==str(cid[0]):
                                csv_writer.writerow({"id":data1[i]['id'],"occ":data1[i]['occupation'],"gen":data1[i]['gender'],"fame":data1[i]['fame'],"by":data1[i]['birthyear']})
                                break

    lb.close()
    csvf.close()
    
def logreg(label):
    data_feeds=pd.read_csv("input.csv")
    data_feeds=data_feeds.iloc[:,1]
    data=pd.read_csv("label.csv",)
    #print(label)
    if label == 'occ':
        y =data.iloc[:,1:2]
    elif label == 'gen':
        y = data.iloc[:,2:3]
    elif label == 'fame':
        y = data.iloc[:,3:4]
    elif label == 'by':
        y = data.iloc[:,4]    
    X_train,X_test,y_train,y_test = train_test_split(data_feeds,y,test_size=0.2,random_state=0)
    

    vec = TfidfVectorizer(preprocessor=_preprocess_feed, ngram_range=(1,2),
                              max_features=MAX_WORD_FEATURES, analyzer='word', min_df=3)

    X_train_counts = vec.fit_transform(X_train.values.astype('U'))
    X_train_counts.shape
    #print(X_train_counts.shape)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape

    X_test_counts = vec.transform(X_test.values.astype('U'))
    X_test_counts.shape
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    X_test_tfidf.shape


    model = LogisticRegression(multi_class='multinomial', solver="newton-cg")
    model.fit(X_train_tfidf, y_train)

    predicted = model.predict(X_test_tfidf)
    #print(predicted)
    accuracy = accuracy_score(y_test, predicted, normalize=True, sample_weight=None)
    print("\n\n{} accuracy: {}\n".format(label, accuracy))
    cm = confusion_matrix(y_test, predicted)
    print("\nclassification report\n {}\n\n".format(classification_report(y_test, predicted)))


    
if __name__ == '__main__':
    labels = ['gen', 'occ', 'fame', 'by']
    #gen is gender , occ is occupation ,fame,by is birth year   
        
    #input feed
    inp_f()
    
    #output label
    out_f()
    
    for label in labels:
        logreg(label)