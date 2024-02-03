
# !pip install nltk
# !pip install transformers
# !pip install datasets
# !pip install evaluate
# !pip install sentencepiece
# !pip install rouge_score
# !pip install tensorflow
# pip install neattext

import pandas as pd
import numpy as np

import seaborn as sns

import neattext.functions as nfx

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df = pd.read_csv("/content/emotion_dataset_raw.csv")
df['Emotion'].replace({'joy': 'happy', 'sadness': 'sad'}, inplace=True)

df.head()

df['Emotion'].value_counts()
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)


Xfeatures = df['Clean_Text']
ylabels = df['Emotion']
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)


from sklearn.pipeline import Pipeline

pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression(solver='liblinear'))])
pipe_lr.fit(x_train,y_train)
pipe_lr.score(x_test,y_test)
txt1 = "Such a lovely weather it is"
pipe_lr.predict([txt1])
pipe_lr.predict_proba([txt1])
import joblib
pipeline_file = open("emotion_classifier_pipe_lr.pkl","wb")
joblib.dump(pipe_lr,pipeline_file)
pipeline_file.close()

