import pandas as pd
import json
from collections import defaultdict, Counter
import glob
from analysis_func.text_preproc import preproc_text
import numpy as np
from collections import defaultdict, Counter
import glob
import json
from random import sample
import sklearn

import re
import string
import warnings
from bs4 import BeautifulSoup
import pickle as pkl

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('stsb-bert-large')

from pandarallel import pandarallel
pandarallel.initialize()

data = pd.read_csv("/home/phadke/recovery/conspirecovery/big_data/full_data.csv", header=0)
data = data.dropna(subset=['title','id'])

print("cleaning data")
#text preproessing - filter engligh, hindi, marathi stop words, remove puncts, hash, mentions, urls, weird spaces etc.
data['clean_text'] = data['title'].parallel_apply(lambda x: preproc_text(x))

#data = data.sample(100)

corpus = data['clean_text'].tolist()
corpus_embeddings = model.encode(corpus)
idlist = data['id'].tolist()


with open("/home/phadke/recovery/conspirecovery/big_data/full_data_bert_embeddings.pkl", "wb") as cfile:
    pkl.dump(corpus_embeddings, cfile)
    
with open("/home/phadke/recovery/conspirecovery/big_data/full_data_bert_idlist.pkl", "wb") as ifile:
    pkl.dump(idlist, ifile)
     
with open("/home/phadke/recovery/conspirecovery/big_data/full_data_bert_corpus.pkl", "wb") as corpusfile:
    pkl.dump(corpus, corpusfile)