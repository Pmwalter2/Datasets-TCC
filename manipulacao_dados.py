import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection
from sklearn import metrics
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import tokenize
import keras
import re
import codecs
import itertools
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import gensim
import unidecode
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from collections import defaultdict
import seaborn as sns
from sklearn.datasets import load_iris



# ======================================================================================================================
# df = pd.read_csv('/content/drive/MyDrive/Datasets TCC Etapa 1/Datasets Corretos/Dataset_Completo_Token_TESTE.csv')
# ======================================================================================================================
sampling_strategy = "not minority"
counter = Counter(df['Classificação Normalizada'].tolist()) #A quantidade de classes como 0, 1, -1
# print(counter)
top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
df = df[df['Classificação Normalizada'].map(lambda x: x in top_10_varieties)]

description_list = df['Texto Processado'].tolist() #Coloca todos os textos nessa variavel
varietal_list = [top_10_varieties[i] for i in df['Classificação Normalizada'].tolist()]
varietal_list = np.array(varietal_list)

count_vect = CountVectorizer() #Conta o número de palavras em um texto, transformando em uma matriz
x_train_counts = count_vect.fit_transform(description_list)


tfidf_transformer = TfidfTransformer() #A importância de uma palavra dentro de um texto
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, varietal_list, test_size=0.3)
rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
# oss = OneSidedSelection(random_state = 32)
X_res, y_res = rus.fit_resample(train_x, train_y)

# Esses são os modelos de classificação
# clf = MultinomialNB().fit(X_res, y_res)
# clf = DecisionTreeClassifier().fit(X_res, y_res)
# clf = svm.SVC(kernel='rbf', break_ties=True).fit(X_res, y_res)

y_pred = clf.predict(test_x)



print("Precision Score : ",precision_score(test_y, y_pred,
                                           average='macro'))
print("Recall Score : ",recall_score(test_y, y_pred,
                                           average='macro'))

print("Accuracy Score :", accuracy_score(test_y, y_pred))

print("F1-Score : ", f1_score(test_y, y_pred,
                              average='macro'))
# Matriz de confusão
# cf_matrix = confusion_matrix(test_y, y_pred)
# sns.heatmap(cf_matrix, cmap='coolwarm', annot=True, linewidth=1, fmt='d',  xticklabels=['0', '-1', '1'],yticklabels=['0', '-1', '1'])
# plt.show()
