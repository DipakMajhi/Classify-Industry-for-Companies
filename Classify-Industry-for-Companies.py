
# coding: utf-8

# # Problem : Classifier to classify industries for companies

# In[36]:

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from itertools import chain
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve


# In[37]:

#Loading data into dataframe :
dataframe = pd.read_csv("ML-Assignment.csv")

# Combining important features (Description, Titles and Subsector 1) for our text classification task

features = dataframe['description']+" "+dataframe['titles']+" "+dataframe['subsector1']

print ("Features (Description, Titles and Subsector 1):--- \n\n"+str(features.head()))

labels = dataframe['real_industries']

print ("\n\nLabels (Real Industries):--- \n\n"+str(labels.head()))


# ## Analysis of Data :

# In[38]:


print ("Unique Industries and their counts : ---\n\n"+str(labels.value_counts()))
print ("\nTotal number of available industries = "+str(len(labels.value_counts()))+"\n")


labels = labels.map({'Enterprise Solutions': 1, 'Ecommerce': 2, 'Social Media': 3, 'Finance': 4, 'Healthcare': 5, 'Gaming': 6,                      'Real Estate': 7, 'Food & Beverage': 8, 'Education': 9, 'Hardware': 10, 'Travel & Hospitality': 11,                      'Energy': 12,'Human Resources': 13, 'Fashion & Beauty': 14, 'Transportation': 15, 'Consumer Software/Apps': 16,                      'Telecommunication': 17, 'Logistics': 18, '3D Printing/Scanning': 19, 'Wearables': 20})

#print labels.head()


# ## Extracting features from text files :
# 
# In order to run machine learning algorithms we need to convert the text files into numerical feature vectors. We will be using bag of words model. Briefly, we segment each text file into words, and count number of times each word occurs in each document and finally assign each word an integer id. Each unique word in our dictionary will correspond to a descriptive feature.
# 
# Scikit-learn has a high level component which will create feature vectors for us ‘CountVectorizer’.

# In[39]:

def split_into_lemmas(features):
    features = unicode(features, 'utf8', errors='replace').lower()
    words = TextBlob(features).words 
    return [word.lemma for word in words]

bow = CountVectorizer(analyzer=split_into_lemmas).fit(features)
print ("Length of Vocabulary : "+str(len(bow.vocabulary_)))


# ### Term Frequency times inverse document frequency (TF-IDF): 
# 
# We can reduce the weightage of more common words like (the, is, an etc.) which occurs in the document. 

# In[40]:

bow_list = bow.transform(features)

tfidf_transformer = TfidfTransformer().fit(bow_list)

bow_tfidf = tfidf_transformer.transform(bow_list)
print ("Dimension of the Document-Term matrix : "+str(bow_tfidf.shape))


# ## Applying Machine Learning Algorithms :
# 

# In[41]:

train, test, label_train, label_test = train_test_split(features, labels, test_size=0.1)

print ("Number of samples in Training Dataset : "+str(len(train)))
print ("Number of samples in Testing Dataset : "+str(len(test)))


# ### Naive Bayes (MultinomialNB) :

# In[42]:

from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # Naive Bayes classifier
])

pipeline = pipeline.fit(train, label_train)

predicted = pipeline.predict(test)

print ("Accuracy Score with MultinomialNB : "+str(accuracy_score(label_test, predicted)))


# ### Stochastic gradient descent (SGD) :

# In[43]:

from sklearn.linear_model import SGDClassifier

pipeline = Pipeline([('bow', CountVectorizer(analyzer=split_into_lemmas)),
                      ('tfidf', TfidfTransformer()),
                      ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

pipeline = pipeline.fit(train, label_train)

predicted = pipeline.predict(test)

print ("Accuracy Score SGDClassifier : "+str(accuracy_score(label_test, predicted)))


# In[44]:

from __future__ import print_function

print ("Actual Result : \n")
for i,j in enumerate(label_test):
    print (str(j)+", ", end='')

print ("\n\n")

print ("Predicted Result : \n")
print (str(predicted) + "\n\n")


# In[45]:

label_test = label_test.map({1: 'Enterprise Solutions', 2: 'Ecommerce', 3: 'Social Media', 4: 'Finance', 5: 'Healthcare', 6: 'Gaming',                      7: 'Real Estate',8: 'Food & Beverage', 9: 'Education', 10: 'Hardware', 11: 'Travel & Hospitality',                      12: 'Energy', 13: 'Human Resources', 14: 'Fashion & Beauty', 15: 'Transportation', 16: 'Consumer Software/Apps',                      17: 'Telecommunication', 18: 'Logistics', 19: '3D Printing/Scanning', 20: 'Wearables'})


predicted = pd.DataFrame(predicted, columns = ['name'])
predicted = predicted['name'].map({1: 'Enterprise Solutions', 2: 'Ecommerce', 3: 'Social Media', 4: 'Finance', 5: 'Healthcare', 6: 'Gaming',                      7: 'Real Estate',8: 'Food & Beverage', 9: 'Education', 10: 'Hardware', 11: 'Travel & Hospitality',                      12: 'Energy', 13: 'Human Resources', 14: 'Fashion & Beauty', 15: 'Transportation', 16: 'Consumer Software/Apps',                      17: 'Telecommunication', 18: 'Logistics', 19: '3D Printing/Scanning', 20: 'Wearables'})


# In[46]:


for i,j in enumerate(label_test.index):
    print ("\n("+str(i)+") - Index:("+str(j)+") - Description:["+str(features[j])+"]\n"+"-->> (Actual : "+str(label_test[j])+") -->> (Predicted : "+str(predicted[i])+")\n")




# ## A try using NLTK :

# In[47]:

stop_words = set(stopwords.words('english'))
stop_words.update(['=','+','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '-','`','!','@','#','$','%','^','&','*','/','|','~']) 
porter = PorterStemmer()
feature_list=[]

for line in features:
    list_of_words = [i.lower() for i in wordpunct_tokenize(line) if i.lower() not in stop_words]
    list_of_words = [ porter.stem(word) for word in list_of_words if word.isalnum() ]
    feature_list.append(list_of_words)


feature_list = list(chain.from_iterable(feature_list))
bow_list = CountVectorizer()
bow_counts = bow_list.fit_transform(feature_list)

#print len(bow_list.vocabulary_)

tfidf_transformer = TfidfTransformer()
bow_tfidf = tfidf_transformer.fit_transform(bow_counts)



# In[ ]:



