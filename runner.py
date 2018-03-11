# encoding=utf8

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn import svm


train=pd.read_csv("C:/Users/Malte/Documents/My repositories/Toxic-Comments-Classification/train.csv",sep=",")
test=pd.read_csv("C:/Users/Malte/Documents/My repositories/Toxic-Comments-Classification/test.csv",sep=",")

ratings=[u'toxic', u'severe_toxic', u'obscene',u'threat', u'insult', u'identity_hate']

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

##########################################################################


tfidf=TfidfVectorizer(strip_accents="ascii",min_df=5,lowercase=True,token_pattern=r'\w+',analyzer="word",ngram_range=(1,3),stop_words="english")
tfidf.fit_transform(test["comment_text"].apply(str))
tfidf_dict=dict(zip(tfidf.get_feature_names(),tfidf.idf_))
test["tfidf"]=test["comment_text"].apply(lambda x: compute_tfidf(x))
test["swear_score"]=test["comment_text"].apply(lambda x: rate_comment(x))

vectorizer=HashingVectorizer(n_features=15)
vector=vectorizer.transform(test["comment_text"])
df_vect=pd.DataFrame(vector.toarray(),index=range(0,len(test)))
test_vect=pd.concat([test,df_vect],axis=1)

# OUTPUT
x_train=train_vect.drop([ u'id',  u'comment_text',         u'toxic',  u'severe_toxic',
         u'obscene',        u'threat',        u'insult', u'identity_hate',],axis=1)
x_test=test_vect.drop([ u'id',"comment_text"],axis=1)

submission=pd.DataFrame()
submission["id"]=test_vect["id"]

for e in ratings:
    y=train[e]
    clf=svm.LinearSVC()
    clf.fit(x_train,y)
    submission[e]=np.array(clf.predict(x_test))

submission.set_index('id', inplace=True)
submission.to_csv("C:/Users/Malte/Documents/My repositories/Toxic-Comments-Classification/submission.csv",encoding="utf-8")