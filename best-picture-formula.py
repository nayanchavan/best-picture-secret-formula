# -*- coding: utf-8 -*-

# Original file is located at: https://colab.research.google.com/drive/11lCtre4npV2afWVh1YjSkmTPjC31ezB8

# Nayan's Final Notebook. 
# This is the notebook I used to various analysis methods with my dataset of the scripts from the lat 50 movies that have won the Best Picture Academy Award. I have compiled code from various notebook to give me my desired results, which I have linked throughout the notebook. I also have the various sections in this notebook demarcated by methodology saved as seperate notebooks, as well, for my own use if needed.

# Make CSV Files of my Dataset
# The following code, adapted from "[Make_csv_tutorial.ipynb](https://colab.research.google.com/drive/11lCtre4npV2afWVh1YjSkmTPjC31ezB8#scrollTo=k_n-kZcb0MdE&line=3&uniqifier=1)", create 6 CSV files in total. One file includes all of the 50 movies in my databate, and the 5 other files contain the same data of movies divided over 5 ten-year intervals.

import os
import pandas as pd

# First we have to give google permission to mount our drive. Click the short link below after you run this cell.
# This link will take you to your google account, select your account, then copy whatever the webpage asks you to copy.
# Paste it below

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Next we find the directory of the folder in which you store all your txt files in the drive
file_directory = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/50 Best Picture Oscar Scripts/')

Title = []
Content = []
for file in file_directory:
  if file.endswith('.txt'):
    Title.append(file)
    Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/50 Best Picture Oscar Scripts/' + file, 'r').read()) ##You still need to replace the path here

your_csv = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv

# Now we need to save our csv to our drive
your_csv.to_csv('50_scripts_dataset.csv', index=False) ##change the name of your csv if you want
!cp 50_scripts_dataset.csv '/content/drive/My Drive/DH100 Nayan Chavan/' #You can decide which path you want your file in.
# The file name after !cp must be the same as the one you put in the first line.

# 2011-2020 

file_directory1 = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2020-2011/')

Title = []
Content = []
for file in file_directory1:
  Title.append(file)
  Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2020-2011/' + file, 'r').read()) ##You still need to replace the path here

your_csv1 = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv1

your_csv1.to_csv('2020-2011.csv', index=False) ##change the name of your csv if you want
!cp 2020-2011.csv '/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/'

# 2001-2010 

file_directory2 = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2010-2001/')

Title = []
Content = []
for file in file_directory2:
  Title.append(file)
  Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2010-2001/' + file, 'r').read()) ##You still need to replace the path here

your_csv2 = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv2

your_csv2.to_csv('2010-2001.csv', index=False) 
!cp 2010-2001.csv '/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/'

# 1991-2000

file_directory3 = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2000-1991/')

Title = []
Content = []
for file in file_directory3:
  Title.append(file)
  Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2000-1991/' + file, 'r').read()) ##You still need to replace the path here

your_csv3 = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv3

your_csv3.to_csv('2000-1991.csv', index=False) 
!cp 2000-1991.csv '/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/'

# 1981-1990

file_directory4 = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/1990-1981/')

Title = []
Content = []
for file in file_directory4:
  Title.append(file)
  Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/1990-1981/' + file, 'r').read()) ##You still need to replace the path here

your_csv4 = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv4

your_csv4.to_csv('1990-1981.csv', index=False) 
!cp 1990-1981.csv '/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/'

# 1971-1980

file_directory5 = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/1980-1971/')

Title = []
Content = []
for file in file_directory5:
  Title.append(file)
  Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/1980-1971/' + file, 'r').read()) ##You still need to replace the path here

your_csv5 = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv5

# Now we need to save our csv to our drive
your_csv5.to_csv('1980-1971.csv', index=False) # change the name of your csv if you want
!cp 1980-1971.csv '/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/' # You can decide which path you want your file in.
# The file name after !cp must be the same as the one you put in the first line.

import os
import pandas as pd

# First we have to give google permission to mount our drive. Click the short link below after you run this cell.
# This link will take you to your google account, select your account, then copy whatever the webpage asks you to copy.
# Paste it below
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Next we find the directory of the folder in which you store all your txt files in the drive
file_directory = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/50 Best Picture Oscar Scripts/') # Here we use an example in our drive, you should replace this with your path

Title = []
Content = []
for file in file_directory:
  if file.endswith('.txt'):
    Title.append(file)
    Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/50 Best Picture Oscar Scripts/' + file, 'r').read()) # You still need to replace the path here

your_csv = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv

# Now we need to save our csv to our drive
your_csv.to_csv('50_scripts_dataset.csv', index=False) # change the name of your csv if you want
!cp 50_scripts_dataset.csv '/content/drive/My Drive/DH100 Nayan Chavan/' # You can decide which path you want your file in.
# The file name after !cp must be the same as the one you put in the first line.

# 2020 - 2011


# Next we find the directory of the folder in which you store all your txt files in the drive
file_directory1 = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2020-2011/') # Here we use an example in our drive, you should replace this with your path

Title = []
Content = []
for file in file_directory1:
  # if file.endswith('.txt'):
  Title.append(file)
  Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2020-2011/' + file, 'r').read()) # You still need to replace the path here

your_csv1 = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv1

# Now we need to save our csv to our drive
your_csv1.to_csv('2020-2011.csv', index=False) # change the name of your csv if you want
!cp 2020-2011.csv '/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/' # You can decide which path you want your file in.
# The file name after !cp must be the same as the one you put in the first line.

# 2010-2001

# Next we find the directory of the folder in which you store all your txt files in the drive
file_directory2 = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2010-2001/') # Here we use an example in our drive, you should replace this with your path

Title = []
Content = []
for file in file_directory2:
  # if file.endswith('.txt'):
  Title.append(file)
  Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2010-2001/' + file, 'r').read()) # You still need to replace the path here

your_csv2 = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv2

# Now we need to save our csv to our drive
your_csv2.to_csv('2010-2001.csv', index=False) # change the name of your csv if you want
!cp 2010-2001.csv '/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/' # You can decide which path you want your file in.
# The file name after !cp must be the same as the one you put in the first line.

# 2000-1991

# Next we find the directory of the folder in which you store all your txt files in the drive
file_directory3 = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2000-1991/') # Here we use an example in our drive, you should replace this with your path

Title = []
Content = []
for file in file_directory3:
  # if file.endswith('.txt'):
  Title.append(file)
  Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/2000-1991/' + file, 'r').read()) # You still need to replace the path here

your_csv3 = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv3

# Now we need to save our csv to our drive
your_csv3.to_csv('2000-1991.csv', index=False) # change the name of your csv if you want
!cp 2000-1991.csv '/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/' # You can decide which path you want your file in.
# The file name after !cp must be the same as the one you put in the first line.

# 1990-1981

# Next we find the directory of the folder in which you store all your txt files in the drive
file_directory4 = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/1990-1981/') # Here we use an example in our drive, you should replace this with your path

Title = []
Content = []
for file in file_directory4:
  # if file.endswith('.txt'):
  Title.append(file)
  Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/1990-1981/' + file, 'r').read()) # You still need to replace the path here

your_csv4 = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv4

# Now we need to save our csv to our drive
your_csv4.to_csv('1990-1981.csv', index=False) # change the name of your csv if you want
!cp 1990-1981.csv '/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/' # You can decide which path you want your file in.
# The file name after !cp must be the same as the one you put in the first line.

# 1980-1971

# Next we find the directory of the folder in which you store all your txt files in the drive
file_directory5 = os.listdir('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/1980-1971/') # Here we use an example in our drive, you should replace this with your path

Title = []
Content = []
for file in file_directory5:
  # if file.endswith('.txt'):
  Title.append(file)
  Content.append(open('/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/1980-1971/' + file, 'r').read()) # You still need to replace the path here

your_csv5 = pd.DataFrame({'Title':Title, 'Content':Content})

your_csv5

# Now we need to save our csv to our drive
your_csv5.to_csv('1980-1971.csv', index=False) # change the name of your csv if you want
!cp 1980-1971.csv '/content/drive/My Drive/DH100 Nayan Chavan/Movies by Year/' # You can decide which path you want your file in.
# The file name after !cp must be the same as the one you put in the first line.

"""## Doc2Vec 
In the following section of code, using the large 50 movie CSV, I produce a Doc2Vec CSV file. I will use this data in Gephi to form a visualization. The code is adapted from "[Doc2Vec(Kenan).ipynb](https://drive.google.com/file/d/1TK6rZ0m1jMQW1fEvKDr-EWNhcnbfaNa9/view?usp=sharing)". This code basically creates vectors between targets and sources (movies) based on their similarity scores. Doc2Vec does not use the "bag of words" approach, and instead more so describes the documents rather than providing topics. 
"""

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.doc2vec import Doc2Vec

downloaded = drive.CreateFile({'id':'1gDzsjZMLj1tbSv7CjtFTw_4p-FAbXAQ9'}) ## Load Secondary_Source.csv in the edge_list_creation folder
downloaded.GetContentFile('50_scripts_dataset.csv')  
file_dictionary = pd.read_csv('50_scripts_dataset.csv')

file_dictionary

file = file_dictionary.values

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

### Get the stopwords from different languages from nltk package
english_stopwords = stopwords.words('english')
french_stopwords = stopwords.words('french')
german_stopwords = stopwords.words('german')
turkish_stopwords = stopwords.words('turkish')
corpus_stopwords = ["continue", "continued", "cont'd", "continuous", "exit", "continuing", "title"] #add stopwords unique to your corpus

## This is a function designed to remove all the stopwords in our text
def preprocess(text): # one little problem, gensim.utils.simple_preprocess gets rid off "I" automatically. 
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
          if token not in english_stopwords:
            if token not in french_stopwords:
              if token not in german_stopwords:
                if token not in turkish_stopwords:
                  if token not in corpus_stopwords:
                    result.append(token)
    return result

file_dictionary = {}
for i in file:
  file_dictionary[i[0]] = i[1]

for key in file_dictionary:
  #print("working on file "+ key)
  file_dictionary[key] = preprocess(file_dictionary[key])

##this function tag all processed files so it is the form for Doc2Vec model 
def tag_doc(processed, i):
  return gensim.models.doc2vec.TaggedDocument(processed, [str(i)])

temp_a = [(k,v) for k,v in file_dictionary.items()]

temp = [i[1] for i in temp_a]

processed_files = []
for i in range(len(temp)):
  processed_files.append(tag_doc(temp[i], i))

test_files = []
for i in range(len(temp)):
  test_files.append(temp[i])

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, epochs=40)

model.build_vocab(processed_files)

model.train(processed_files, total_examples=model.corpus_count, epochs=model.epochs)

threshold = .5
Source = []
Target = []
Weight = []
number_of_doc = 50

for i in range(number_of_doc):
  for k in range(i+1, number_of_doc):
    print("working on " + str(i) + "comparing" + str(k))
    similar = model.n_similarity(test_files[i],test_files[k])
    if similar > threshold:
      Source.append(file[i][0])
      Target.append(file[k][0])
      Weight.append(similar)

result = pd.DataFrame({'Source':Source, 'Target':Target, 'Similarity':Weight})

result

from google.colab import drive
drive.mount('drive')

result.to_csv('Doc2Vec_edge_list_large_new.csv', index=False)
!cp Doc2Vec_edge_list_large_new.csv drive/My\ Drive/DH100\ Nayan\ Chavan/Notebooks/Doc2Vec

"""## Topic Modeling
The following section of code focuses on topic modeling and network analysis. This code is adapted from ["TM2Net_edge_list_(Kenan).ipynb"](https://drive.google.com/file/d/1Sg_FF5SPL_AxEKvk9_E-meCyHm7E-1A_/view?usp=sharing). The CSV produced from this code is then visualized in Gephi to show edges between movies. This code also returns 10 topics of 10 words which were the most prevalent between the movies. This will be very helpful later when analyzing the LDA topics. 
"""

downloaded = drive.CreateFile({'id':'1gDzsjZMLj1tbSv7CjtFTw_4p-FAbXAQ9'}) ## Load Secondary_Source.csv in the edge_list_creation folder
downloaded.GetContentFile('50_scripts_dataset.csv')  
file_dictionary = pd.read_csv('50_scripts_dataset.csv')

file = file_dictionary.values   ## this cell is for putting all the files in a dictionary shape.
file_dictionary = {}
for i in file:
  file_dictionary[i[0]] = i[1]

file_dictionary['Rocky (1977)  '] ## Now we can check the key to be the file name, the content is matched to the key

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

### Get the stopwords from different languages from nltk package
english_stopwords = stopwords.words('english')
french_stopwords = stopwords.words('french')
german_stopwords = stopwords.words('german')
turkish_stopwords = stopwords.words('turkish')
corpus_stopwords = ["continue", "continued", "cont'd", "continuous", "exit", "continuing", "title", "cont", "ext"] #add stopwords unique to your corpus

## This is a function designed to remove all the stopwords in our text
def preprocess(text): # one little problem, gensim.utils.simple_preprocess gets rid off "I" automatically. 
    result = []
    for token in gensim.utils.simple_preprocess(text):
      if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
        if token not in english_stopwords:
          if token not in french_stopwords:
            if token not in german_stopwords:
              if token not in turkish_stopwords:
                if token not in corpus_stopwords: 
                  result.append(token)
    return result

for key in file_dictionary:
  print("working on file "+ key)
  file_dictionary[key] = preprocess(file_dictionary[key])

test_file = file_dictionary['Rocky (1977)  '] # an example to show what the file is stored in the dictionrary after preprocessed.

test_file

processed_files = []
for key in file_dictionary:
  processed_files.append(file_dictionary[key])

dictionary = gensim.corpora.Dictionary(processed_files)

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_files]

from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

topic_number = 10 # how many topic you want, by default 10 topics

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=topic_number, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.show_topics(num_topics=10, num_words=7, log=False, formatted=True): ## change num_words to see number of words you want to see in a topic
    print('Topic: {} \nWords: {}'.format(idx+1, topic))

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx+1, topic))

top = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[],8:[],9:[], 10:[]}

for key in file_dictionary:
  topics = sorted(lda_model[bow_corpus[processed_files.index(file_dictionary[key])]], key = lambda tup: -1*tup[1])
  for i in topics:
    topic_number = i[0]+1
    inside = [key, i[1]]
    if i[1] > 0.25: ## change the threshold for topic weights
      top[topic_number].append(inside)

Source = []
Target = []
Source_weight = []
Target_weight = []
Topic = []

for i in range(1, 11):
  for j in range(len(top[i])):
    for k in range(j+1, len(top[i])):
      if top[i][j][1] >= top[i][k][1]:
        Source.append(top[i][j][0])
        Target.append(top[i][k][0])
        Source_weight.append(top[i][j][1])
        Target_weight.append(top[i][k][1])
      else:
        Source.append(top[i][k][0])
        Target.append(top[i][j][0])
        Source_weight.append(top[i][k][1])
        Target_weight.append(top[i][j][1])
      Topic.append(i)

result = pd.DataFrame({'Source':Source, 'Target':Target, 'Source_Weight':Source_weight, 'Target_Weight': Target_weight, 'Topic': Topic})

Type = ['directed'] * len(result)

result = pd.DataFrame({'Source':Source, 'Target':Target, 'Source_Weight':Source_weight, 'Target_Weight': Target_weight, 'Type': Type, 'Topic': Topic})

result

from google.colab import drive
drive.mount('drive')

result.to_csv('Topic_edge_list2.csv', index=False)
!cp Topic_edge_list2.csv drive/My\ Drive/DH100\ Nayan\ Chavan/

"""## LDA Topic Modeling
The following section of code creates 6 LDA models using the 6 CSV files that were created earlier. The last visualization created is with all 50 movie files, while the first 5 visualizations are those from the respective movies in ten-year intervals over fifty years. The following code is adapted from the ["BasicTopicModeling.ipynb"](https://drive.google.com/file/d/15Kr7LqD_znL3CM7dae5DfBwg4geageUz/view?usp=sharing). The LDA model uses the "bag of words" approach so most of its analysis is based on word frequency and prevalance because of this the topics differ from those produced with the Topic Modeling code. 
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sklearn
import seaborn as sns
import os
import re
import plotly.express as px
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords 
from nltk import word_tokenize


import warnings
warnings.filterwarnings('ignore')

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

"""2011-2020 Dataset"""

import pandas as pd
import os

#upload csv
from google.colab import files
csv1 = drive.CreateFile({'id':'1-PP1PcXd6HONI6VdMWT6z8BVY3CpLLSh'}) ## Load Secondary_Source.csv in the edge_list_creation folder
csv1.GetContentFile('2020-2011.csv')  
dataset1 = pd.read_csv('2020-2011.csv')
dataset1

"""2001-2010 Dataset"""

csv2 = drive.CreateFile({'id':'1-J6IIEgk8mh0JBT6lkv_nXEagKjEKyhA'}) ## Load Secondary_Source.csv in the edge_list_creation folder
csv2.GetContentFile('2010-2001.csv')  
dataset2 = pd.read_csv('2010-2001.csv')
dataset2

"""1991-2000 Dataset"""

csv3 = drive.CreateFile({'id':'1-Sd1CV66YZDt8gsL92XMSTl4GpRr81fc'}) ## Load Secondary_Source.csv in the edge_list_creation folder
csv3.GetContentFile('2000-1991.csv')  
dataset3 = pd.read_csv('2000-1991.csv')
dataset3

"""1981-1990 Dataset"""

csv4 = drive.CreateFile({'id':'1-S_8A-quSyLipiKZn1spFgdcQHqy91Y-'}) ## Load Secondary_Source.csv in the edge_list_creation folder
csv4.GetContentFile('1990-1981.csv')  
dataset4 = pd.read_csv('1990-1981.csv')
dataset4

"""1971-1980 Dataset"""

csv5 = drive.CreateFile({'id':'1-aejmTQt-uF31hwy7AmE51NAu_fqmRvs'}) ## Load Secondary_Source.csv in the edge_list_creation folder
csv5.GetContentFile('1980-1971.csv')  
dataset5 = pd.read_csv('1980-1971.csv')
dataset5

dataset1['Content'] = dataset1['Content'].str.replace('io', '').str.replace('cont', '').str.replace('continued','').str.replace('contd', '').str.replace('exit', '').str.replace('ext', '').str.replace('continuing', '').str.replace('continue', '')
dataset2['Content'] = dataset2['Content'].str.replace('io', '').str.replace('cont', '').str.replace('continued','').str.replace('contd', '').str.replace('exit', '').str.replace('ext', '').str.replace('continuing', '').str.replace('continue', '')
dataset3['Content'] = dataset3['Content'].str.replace('io', '').str.replace('cont', '').str.replace('continued','').str.replace('contd', '').str.replace('exit', '').str.replace('ext', '').str.replace('continuing', '').str.replace('continue', '')
dataset4['Content'] = dataset4['Content'].str.replace('io', '').str.replace('cont', '').str.replace('continued','').str.replace('contd', '').str.replace('exit', '').str.replace('ext', '').str.replace('continuing', '').str.replace('continue', '')
dataset5['Content'] = dataset5['Content'].str.replace('io', '').str.replace('cont', '').str.replace('continued','').str.replace('contd', '').str.replace('exit', '').str.replace('ext', '').str.replace('continuing', '').str.replace('continue', '')

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
english_stopwords = stopwords.words('english')

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# Load the regular expression library
import re
# Remove punctuation
dataset1['text_processed'] = dataset1['Content'].map(lambda x: re.sub('[,\.!?\n/-]', '', x))
dataset2['text_processed'] = dataset2['Content'].map(lambda x: re.sub('[,\.!?\n/-]', '', x))
dataset3['text_processed'] = dataset3['Content'].map(lambda x: re.sub('[,\.!?\n/-]', '', x))
dataset4['text_processed'] = dataset4['Content'].map(lambda x: re.sub('[,\.!?\n/-]', '', x))
dataset5['text_processed'] = dataset5['Content'].map(lambda x: re.sub('[,\.!?\n/-]', '', x))

# Convert the titles to lowercase
dataset1['text_processed'] = dataset1['text_processed'].map(lambda x: x.lower())
dataset2['text_processed'] = dataset2['text_processed'].map(lambda x: x.lower())
dataset3['text_processed'] = dataset3['text_processed'].map(lambda x: x.lower())
dataset4['text_processed'] = dataset4['text_processed'].map(lambda x: x.lower())
dataset5['text_processed'] = dataset5['text_processed'].map(lambda x: x.lower())

# Remove digits
dataset1['text_processed'] = dataset1['text_processed'].str.replace('\d+', '')
dataset2['text_processed'] = dataset2['text_processed'].str.replace('\d+', '')
dataset3['text_processed'] = dataset3['text_processed'].str.replace('\d+', '')
dataset4['text_processed'] = dataset4['text_processed'].str.replace('\d+', '')
dataset5['text_processed'] = dataset5['text_processed'].str.replace('\d+', '')

# Save each dataset as one long string 
long_string1 = ','.join(list(dataset1['text_processed'].values))
long_string2 = ','.join(list(dataset2['text_processed'].values))
long_string3 = ','.join(list(dataset3['text_processed'].values))
long_string4 = ','.join(list(dataset4['text_processed'].values))
long_string5 = ','.join(list(dataset5['text_processed'].values))

# Tokenize preprocessed long strings 
token1 = long_string1.split()
token2 = long_string2.split()
token3 = long_string3.split()
token4 = long_string4.split()
token5 = long_string5.split()

filters = ["int", "continue", "continued", "cont'd", "continuous", "exit", "continuing", "title", "cont", "like", "know", "ext"]
token1 = [x for x in token1 if
              all(y not in x for y in filters)]
token2 = [x for x in token2 if
              all(y not in x for y in filters)]
token3 = [x for x in token3 if
              all(y not in x for y in filters)]
token4 = [x for x in token4 if
              all(y not in x for y in filters)]
token5 = [x for x in token5 if
              all(y not in x for y in filters)]

# Commented out IPython magic to ensure Python compatibility.
# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA

!pip install -q pyLDAvis
from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis

import re

from google.colab import files
downloaded = drive.CreateFile({'id':'1b9P9VgeuAjHQMHYRGxN1mD3dUKLXGNF9'}) ## Load Secondary_Source.csv in the edge_list_creation folder
downloaded.GetContentFile('50_scripts_dataset (6).csv')
dataset = pd.read_csv('50_scripts_dataset (6).csv')
dataset['text_processed'] = dataset['Content'].map(lambda x: re.sub('[,\.!?\n/-]', '', x))
dataset['text_processed'] = dataset['text_processed'].map(lambda x: x.lower())
dataset['text_processed'] = dataset['text_processed'].str.replace('\d+', '')
long_string = ','.join(list(dataset['text_processed'].values))
tokens = long_string.split()

import re 

def print_top_words(model, feature_names, n_top_words):
  for topic_idx, topic in enumerate(model.components_):
    print("\nTopic #{}:".format(topic_idx))
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

from google.colab import files
downloaded = drive.CreateFile({'id':'1b9P9VgeuAjHQMHYRGxN1mD3dUKLXGNF9'}) ## Load Secondary_Source.csv in the edge_list_creation folder
downloaded.GetContentFile('50_scripts_dataset (6).csv')
dataset = pd.read_csv('50_scripts_dataset (6).csv')
dataset['text_processed'] = dataset['Content'].map(lambda x: re.sub('[,\.!?\n/-]', '', x))
dataset['text_processed'] = dataset['text_processed'].map(lambda x: x.lower())
dataset['text_processed'] = dataset['text_processed'].str.replace('\d+', '')
long_string = ','.join(list(dataset['text_processed'].values))
tokens = long_string.split()

tokens = [x for x in tokens if
              all(y not in x for y in filters)] 

def create_topic_model(tokens, n_topics =5):
    tfidf_vectorizer = TfidfVectorizer(max_df= .9,
                                       max_features = 5000,
                                       stop_words = "english")
    tfidf = tfidf_vectorizer.fit_transform(tokens)
    lda = LDA(n_components = n_topics,
              max_iter = 20, 
              random_state = 5)
    lda = lda.fit(tfidf)

    tf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, 15)
    panel = pyLDAvis.sklearn.prepare(lda_model= lda, 
                                     dtm = tfidf, 
                                     vectorizer = tfidf_vectorizer,
                                     mds = "tsne")

    return panel

"""## 2011-2020 LDA Topic Model"""

panel1 = create_topic_model(token1)
pyLDAvis.display(panel1)

"""### 2001-2010 LDA Topic Model """

panel2 = create_topic_model(token2)
pyLDAvis.display(panel2)

"""### 1991-2000 LDA Topic Model"""

panel3 = create_topic_model(token3)
pyLDAvis.display(panel3)

"""### 1981-1990 LDA Topic Model"""

panel4 = create_topic_model(token4)
pyLDAvis.display(panel4)

"""### 1971-1980 LDA Topic Model"""

panel5 = create_topic_model(token5)
pyLDAvis.display(panel5)

"""### 1971-2020 LDA Topic Model (All 50 Movies)"""

panel = create_topic_model(tokens)
pyLDAvis.display(panel)
