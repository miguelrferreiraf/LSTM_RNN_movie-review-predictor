#!/usr/bin/env python
# coding: utf-8

# ## LSTM Neural Nets and Flask/Streamlit Deploy

# This small project deploy a LSTM/RNN Neural Net that can learn to read. Deployment occurs through Flask and Streamlit.

# ### Importing libraries

# In[4]:


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pickle


# ### Importing and working on dataset 

# In[5]:


df = pd.read_csv('labeledTrainData.tsv',header=0, delimiter="\t", quoting=3)
df = df[['review','sentiment']]

df.shape


# In[5]:


df.sentiment.value_counts()


# Adding some cleaning methods for text so it make easier for the algorithm to comprehend it.

# In[6]:


df['review'] = df['review'].apply(lambda x: x.lower())
df['review'] = df['review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))


# In[6]:


max_features = 1000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['review'].values)
X = tokenizer.texts_to_sequences(df['review'].values)
X = pad_sequences(X)
X.shape


# In[8]:


embed_dim = 50
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(LSTM(10))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])


# In[9]:


print(model.summary())


# In[10]:


y = pd.get_dummies(df['sentiment']).values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25,
                                                    random_state = 99)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# ## Training the model

# In[11]:


model.fit(X_train, y_train, epochs = 5, verbose = 1)


# ### Testing the model

# Now we create a variable with a string of test to be availed by the model.

# In[9]:


test = ['Movie was pathetic']
test = tokenizer.texts_to_sequences(test)
test = pad_sequences(test, maxlen=X.shape[1],dtype='int32', value=0)
print(test.shape)

sentiment = model.predict(test)[0]
if(np.argmax(sentiment) == 0):
    print("Negative")
elif (np.argmax(sentiment) == 1):
    print("Positive")


# ### Saving the model

# In[11]:


with open('tokenizer.pickle', 'wb') as tk:
    pickle.dump(tokenizer, tk, protocol=pickle.HIGHEST_PROTOCOL)
    
model_json = model.to_json()
with open("model.json", "w") as js:
    js.write(model_json)


# In[12]:


model.save_weights("model.h5")


# # TUDO ACIMA FUNCIONOU PERFEITAMENTE BEM. POR FAVOR, NÃO ESTRAGUE TUDO!

# ## Finally, the app creation

# In[13]:


import os
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import streamlit
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ### Loading the model and JSON archive

# In[14]:


with open('tokenizer.pickle', 'rb') as tk:
    tokenizer = pickle.load(tk)
    
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
lstm_model = model_from_json(loaded_model_json)
lstm_model.load_weights("model.h5")


# In[15]:


def sentiment_prediction(review):
    sentiment=[]
    input_review = [review]
    input_review = [x.lower() for x in input_review]
    input_review = [re.sub('[^a-zA-z0-9\s]','',x) for x in input_review]
    input_feature = tokenizer.texts_to_sequences(input_review)
    input_feature = pad_sequences(input_feature,1473, padding='pre')
    sentiment = lstm_model.predict(input_feature)[0]
    if(np.argmax(sentiment) == 0):
        pred="Negative"
    else:
        pred= "Positive"
    return pred


# ### Creating HTML template with Streamlit

# At the end, we create the run function to load the HTML page and accept the user input using Streamlit functionality (similar to the earlier model deployment).

# In[18]:


def run():
    streamlit.title("Sentiment Analysis - LSTM Model")
    html_temp=""" """
    streamlit.markdown(html_temp)
    review=streamlit.text_input("Enter the Review ")
    prediction=""
    


# In[20]:


if streamlit.button("Predict Sentiment"):
    prediction=sentiment_prediction(review)
    streamlit.success("The sentiment predicted by Model : {}".format(prediction))
    
if __name__=='__main__':
    run()


# # APARENTEMENTE, O STREAMLIT NÃO FUNCIONA NO JUPYTER! QUE ABSURDO! MAS EU VOU DAR UM JEITO NISSO AMANHÃ

# In[ ]:


jupyter nbconvert --to script Streamlit_Jupyter.ipynb

