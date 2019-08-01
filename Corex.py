from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as ss
from corextopic import corextopic as ct
import numpy as np
import pandas as pd
from helper import *
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
import string

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation) 

def cleaning(article):
    one = " ".join([i for i in article.lower().split() if i not in stopwords])
    two = "".join(i for i in one if i not in punctuation)
    return two

df = pd.read_csv(open("abstracts for 'automat'.csv", errors='ignore'))
df=df.astype(str)

text = df.applymap(cleaning)['paperAbstract']
text_list = [i.split() for i in text]

all_joined=[]

for element in text_list:
    joined=' '.join(element)
    all_joined.append(joined)

allll=[]

for element in all_joined:
    within=element.split(' ')
    allll.append(within)
       
vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df=0.01)
vector=vectorizer.fit_transform(allll).todense()

vocab=vectorizer.vocabulary_ 

numpy_array=np.array(vector, dtype=int)

# Sparse matrices are also supported
X = ss.csr_matrix(numpy_array)

# WORD LABELS for each column can be provided to the model
all_vocabs=list(vocab.keys())

# DOCUMENT LABELS for each row can be provided
topics = np.arange(len(allll)) #['fruit doc', 'animal doc', 'mixed doc'] #10 any-given topic names

seed = 1 #CHANGE THE SEED HERE

# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=10, seed=seed, n_words=30) #define the number of latent (hidden) topics to use
topic_model.fit(X, words=all_vocabs, docs=topics)

topics = topic_model.get_topics()

print (topics) #shows MIs too
print ('\n')

print ('Corex Topics:')
for topic_n,topic in enumerate(topics):
    all_topics=[]
    for item in topic:
        list_topic=list(item)
        all_topics.append(list_topic[0])
    print ('Topic '+ str(topic_n+1) + ': ' + str(all_topics))

top_docs = topic_model.get_top_docs()
for topic_n, topic_docs in enumerate(top_docs):
    docs,probs = zip(*topic_docs)
    topic_str = str(topic_n+1)+': '+ ''.join(str(docs))
    print(topic_str)