# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:43:37 2020

@author: amare
"""


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

raw=pd.read_csv('C:/Users/amare/Downloads/Dataset_recommendations_takehome_tags/Dataset_recommendations_takehome_tags.csv')

# Option 1
vectorizer=CountVectorizer()
vectorizer.fit(raw['list_of_tags'])
df_tags=vectorizer.vocabulary_

for key in list(df_tags.keys()):
    if df_tags[key] < 11 or df_tags[key] > 3010:
        del df_tags[key]
        
tags_3000=pd.DataFrame.from_dict(df_tags, orient='index')
tags_trans=tags_3000.transpose()
raw.merge=pd.concat([raw,tags_trans], axis=1)

#option 2

from keras.preprocessing.text import Tokenizer

t = Tokenizer()

t.fit_on_texts(raw['list_of_tags'])


tags_num = t.texts_to_matrix(raw['list_of_tags'], mode='count')
tags_num= pd.DataFrame(tags_num)


tags_num['label','petition_id'] =raw['label','petition_id']
#tags_num['petition_id']
#tags_num.head()
cols = list(tags_num.columns)
cols = [cols[-1]] + cols[:-1]
tags_num = tags_num[cols]
