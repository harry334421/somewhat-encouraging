import json
import pickle
import re
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.naive_bayes import MultinomialNB


warnings.filterwarnings("ignore")

pd.options.display.max_colwidth = -1

data = pd.read_json('G:/Python/test2/News_Category_Dataset_v2.json', lines=True)
data = data.drop(['authors', 'date'], axis=1)

plotdata = data['category'].value_counts()

def generalize(x):
    if x in ['BUSINESS', 'MONEY']:
        return 'BUSINESS'
    if x in ['SCIENCE', 'TECH', 'ARTS', 'ARTS & CULTURE', 'COLLEGE', 'CULTURE & ARTS', 'EDUCATION']:
        return 'EDUCATION'
    if x in ['WELLNESS', 'TRAVEL', 'STYLE & BEAUTY', 'PARENTING', 'HEALTHY LIVING', 'HOME & LIVING', 'FOOD & DRINK', 'PARENTS', 'WEDDINGS', 'WOMEN', 'DIVORCE', 'RELIGION', 'STYLE', 'TASTE', 'FIFTY']:
        return 'LIFESTYLE'
    if x in ['QUEER VOICES', 'BLACK VOICES', 'THE WORLDPOST', 'WORLDPOST', 'WORLD NEWS', 'LATINO VOICES', 'IMPACT', 'MEDIA']:
        return 'GLOBAL'
    if x in ['COMEDY', 'CRIME', 'WEIRD NEWS', 'GREEN', 'GOOD NEWS', 'ENVIRONMENT']:
        return 'MISCELLANEOUS'
    else:
        return x

data['category'] = data['category'].apply(generalize)
plotdata = data['category'].value_counts()

#print(plotdata.index,len(plotdata.index))

#print(plotdata)
data['article'] = data['headline'] + ' ' + data['short_description']
##清理文本##
remove_punc = lambda x : re.sub(r"\W", ' ', x)

remove_extra_spaces = lambda x : re.sub(r"\s+", ' ', x)

lower_case = lambda x : x.lower()

stop_words = set(nltk.corpus.stopwords.words('english'))
remove_stopwords = lambda x: ' '.join(word for word in x.split() if word not in stop_words)

ps = PorterStemmer()
ps_stem = lambda x: ' '.join(ps.stem(word) for word in x.split())

wnl = WordNetLemmatizer()
wnl_lemmatize = lambda x: ' '.join(wnl.lemmatize(word) for word in x.split())

def tag_pos(x):
    tag_list =  nltk.pos_tag(nltk.word_tokenize(x))
    pos = ""
    for t in tag_list:
        pos += t[0] +'(' + t[1] +')' + ' '
    return pos

def cleanText(x, rsw, stm, lem, tgps):
    x = remove_punc(x)
    x = remove_extra_spaces(x)
    x = lower_case(x)
    if rsw:
        x = remove_stopwords(x)
    if stm:
        x = ps_stem(x)
    if lem:
        x = wnl_lemmatize(x)
    if tgps:
        x = tag_pos(x)
    return x
data['article_clean'] = data['article'].apply(lambda x : cleanText(x, True, False, True, False))

##Vectorization##
vectorizer = TfidfVectorizer()
vectorizer.fit(data['article_clean'])
X_vect = vectorizer.transform(data['article_clean'])
Y = pd.get_dummies(data['category'])
pickle.dump(vectorizer, open('Tvect', 'wb'))

#交叉验证#
model_accuracies = pd.DataFrame()
'''
#逻辑回归
accuracy_means = []
for y in Y.columns:
    model = LogisticRegression()
    scores = cross_val_score(model, X_vect, Y[y], cv=10)
    print('Accuracy: %0.2f (+/- %0.2f)' %(scores.mean(), scores.std()))
    accuracy_means.append(scores.mean())
model_accuracies['LR'] = accuracy_means
'''
'''
#离散型朴素贝叶斯
accuracy_means = []
for y in Y.columns:
    model = MultinomialNB()
    scores = cross_val_score(model, X_vect, Y[y], cv=10)
    print('Accuracy: %0.2f (+/- %0.2f)' %(scores.mean(), scores.std()))
    accuracy_means.append(scores.mean())
model_accuracies['MNB'] = accuracy_means
'''
#Train
#LR

lr01 = LogisticRegression()
lr01.fit(X_vect, Y['LIFESTYLE'])

lr02 = LogisticRegression()
lr02.fit(X_vect, Y['POLITICS'])

lr03 = LogisticRegression()
lr03.fit(X_vect, Y['GLOBAL'])

lr04 = LogisticRegression()
lr04.fit(X_vect, Y['MISCELLANEOUS'])

lr05 = LogisticRegression()
lr05.fit(X_vect, Y['ENTERTAINMENT'])

lr06 = LogisticRegression()
lr06.fit(X_vect, Y['EDUCATION'])

lr07 = LogisticRegression()
lr07.fit(X_vect, Y['BUSINESS'])

lr08 = LogisticRegression()
lr08.fit(X_vect, Y['SPORTS'])
'''
pickle.dump(lr01, open('lifestyleLR.pickle', 'wb'))
pickle.dump(lr02, open('politicsLR.pickle', 'wb'))
pickle.dump(lr03, open('globalLR.pickle', 'wb'))
pickle.dump(lr04, open('miscellaneousLR.pickle', 'wb'))
pickle.dump(lr05, open('entertainmentLR.pickle', 'wb'))
pickle.dump(lr06, open('educationLR.pickle', 'wb'))
pickle.dump(lr07, open('businessLR.pickle', 'wb'))
pickle.dump(lr08, open('sportsLR.pickle', 'wb'))
'''

#MNB
mnb01 = MultinomialNB()
mnb01.fit(X_vect, Y['LIFESTYLE'])

mnb02 = MultinomialNB()
mnb02.fit(X_vect, Y['POLITICS'])

mnb03 = MultinomialNB()
mnb03.fit(X_vect, Y['GLOBAL'])

mnb04 = MultinomialNB()
mnb04.fit(X_vect, Y['MISCELLANEOUS'])

mnb05 = MultinomialNB()
mnb05.fit(X_vect, Y['ENTERTAINMENT'])

mnb06 = MultinomialNB()
mnb06.fit(X_vect, Y['EDUCATION'])

mnb07 = MultinomialNB()
mnb07.fit(X_vect, Y['BUSINESS'])

mnb08 = MultinomialNB()
mnb08.fit(X_vect, Y['SPORTS'])
'''
pickle.dump(mnb01, open('lifestyleMNB.pickle', 'wb'))
pickle.dump(mnb02, open('politicsMNB.pickle', 'wb'))
pickle.dump(mnb03, open('globalMNB.pickle', 'wb'))
pickle.dump(mnb04, open('miscellaneousMNB.pickle', 'wb'))
pickle.dump(mnb05, open('entertainmentMNB.pickle', 'wb'))
pickle.dump(mnb06, open('educationMNB.pickle', 'wb'))
pickle.dump(mnb07, open('businessMNB.pickle', 'wb'))
pickle.dump(mnb08, open('sportsMNB.pickle', 'wb'))
'''
#分类
def classify(news):
    news_clean = cleanText(news, True, False, True, False)
    news_vect = vectorizer.transform([news_clean])
    lrpred = []
    if lr01.predict(news_vect):
        lrpred.append('LIFESTYLE')
    if lr02.predict(news_vect):
        lrpred.append('POLITICS')
    if lr03.predict(news_vect):
        lrpred.append('GLOBAL')
    if lr04.predict(news_vect):
        lrpred.append('MISCELLANEOUS')
    if lr05.predict(news_vect):
        lrpred.append('ENTERTAINMENT')
    if lr06.predict(news_vect):
        lrpred.append('EDUCATION')
    if lr07.predict(news_vect):
        lrpred.append('BUSINESS')
    if lr08.predict(news_vect):
        lrpred.append('SPORTS')
    mnbpred = []
    if mnb01.predict(news_vect):
        mnbpred.append('LIFESTYLE')
    if mnb02.predict(news_vect):
        mnbpred.append('POLITICS')
    if mnb03.predict(news_vect):
        mnbpred.append('GLOBAL')
    if mnb04.predict(news_vect):
        mnbpred.append('MISCELLANEOUS')
    if mnb05.predict(news_vect):
        mnbpred.append('ENTERTAINMENT')
    if mnb06.predict(news_vect):
        mnbpred.append('EDUCATION')
    if mnb07.predict(news_vect):
        mnbpred.append('BUSINESS')
    if mnb08.predict(news_vect):
        mnbpred.append('SPORTS')
    return lrpred, mnbpred

def interpret(classification):
    print('Tags identified by Logistic Regression:')
    for x in classification[0]:
        print(x)
    print()
    print('Tags identified by MultinomialNB:')
    for x in classification[1]:
        print(x)
    print()
    print('Common Tags:')
    for x in set(classification[0])&set(classification[1]):
        print(x)
    print()
    print('All Tags:')
    for x in set(classification[0])|set(classification[1]):
        print(x)

    
    #测验新闻
sample_news=[
    "Brit paratrooper jailed in Iraq for killing two comrades is flown to UK prison",
    "Impeachment witnesses tested Republican defenses of Trump one by one",
    "Best Christmas party dresses to buy now for the festive season from the high street",
    "Giant 50,000C 'Wall of Fire' surrounding our Solar System discovered by Nasa"
    ]
test_results = pd.DataFrame(columns=['news', 'LR', 'MNB', 'Common', 'All'])
for i in range(len(sample_news)):
    x = classify(sample_news[i])
    test_results.loc[i] = [sample_news[i], x[0], x[1], set(x[0])&set(x[1]), set(x[0])|set(x[1])]

print(test_results)

'''result:
0  Brit paratrooper jailed in Iraq for killing two comrades is flown to UK prison       [GLOBAL]     [GLOBAL]     {GLOBAL}     {GLOBAL}   
1  Impeachment witnesses tested Republican defenses of Trump one by one                 [POLITICS]   [POLITICS]   {POLITICS}   {POLITICS} 
2  Best Christmas party dresses to buy now for the festive season from the high street  [LIFESTYLE]  [LIFESTYLE]  {LIFESTYLE}  {LIFESTYLE}
3  Giant 50,000C 'Wall of Fire' surrounding our Solar System discovered by Nasa         [EDUCATION]  []           {}           {EDUCATION}
'''