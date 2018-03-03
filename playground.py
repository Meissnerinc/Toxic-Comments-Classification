# encoding=utf8

import pandas as pd
import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

train=pd.read_csv("C:/Users/Malte/Documents/My repositories/Toxic/train.csv",sep=",")

print train.columns

cloud = WordCloud(width=1440, height=1080).generate(" ".join(train['comment_text'].astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')

print train["comment_text"].loc[train["toxic"]==1]
