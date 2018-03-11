# encoding=utf8

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train=pd.read_csv("C:/Users/Malte/Documents/My repositories/Toxic-Comments-Classification/train.csv",sep=",")


ratings=[u'toxic', u'severe_toxic', u'obscene',
       u'threat', u'insult', u'identity_hate']

swears=["cunt","faggot","homo","asshole","nigger","negro","ass","fuck","fucker","motherfucker","Jew","Pig","moron","idiot","wanker"]

def compute_tfidf(desc):
    tfidf_score=0
    word_count=0
    for w in desc.lower().split():
        if w in tfidf_dict:
            tfidf_score+=tfidf_dict[w]
        word_count+=1
    if word_count>0:
        return tfidf_score
    else:
        return 0

def rate_comment(s):
    c=0
    s=s.lower()
    s=s.split()
    for e in swears:
        if e in s:
            c+=1
    return c

tfidf=TfidfVectorizer(strip_accents="ascii",min_df=5,lowercase=True,token_pattern=r'\w+',analyzer="word",ngram_range=(1,3),stop_words="english")
tfidf.fit_transform(train["comment_text"].apply(str))
tfidf_dict=dict(zip(tfidf.get_feature_names(),tfidf.idf_))
train["tfidf"]=train["comment_text"].apply(lambda x: compute_tfidf(x))
train["swear_score"]=train["comment_text"].apply(lambda x: rate_comment(x))




vectorizer=HashingVectorizer(n_features=15)
vector=vectorizer.transform(train["comment_text"])
df_vect=pd.DataFrame(vector.toarray(),index=range(0,len(train)))
train_vect=pd.concat([train,df_vect],axis=1)

# Testing Area
# Best SO FAR has been linear svc, followed by kneighbors and GaussianNB


def test_gauntlet(cat,ml):
    print cat
    x=train[["tfidf","swear_score"]]
    print x.columns
    y=train[cat]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    clf=ml
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    print accuracy_score(y_test,pred)

    x=train_vect.drop([ u'id',  u'comment_text',         u'toxic',  u'severe_toxic',
             u'obscene',        u'threat',        u'insult', u'identity_hate',],axis=1)
    print x.columns
    y=train[cat]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    clf=ml
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    print accuracy_score(y_test,pred)

for e in ratings:
    test_gauntlet(e,svm.LinearSVC())
    print " "

# DRAWING

tox=train.loc[train["toxic"]==1]["comment_text"].astype(str)
print tox

cloud = WordCloud(width=1440, height=1080).generate(" ".join(tox))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
