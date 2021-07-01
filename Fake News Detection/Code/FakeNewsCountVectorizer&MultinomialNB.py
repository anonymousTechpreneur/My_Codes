#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import itertools


# In[ ]:


df = pd.read_csv('fake_or_real_news.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df = df.set_index('Unnamed: 0')


# In[ ]:


df.head()


# In[ ]:


y = df.label


# In[ ]:


df = df.drop('label', axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)


# In[ ]:


count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)


# In[ ]:


count_vectorizer.get_feature_names()[:10]


# In[ ]:


count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())


# In[ ]:


count_df.head()


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


model = MultinomialNB() 


# In[ ]:


model.fit(count_train, y_train)


# In[ ]:


pred = model.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


# In[ ]:


cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[ ]:




