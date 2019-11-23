import gensim
import re
import json
import numpy as np
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
import gc
from nltk.corpus import stopwords

MAX_NEWS = 30000 


def headline2vec(s, model):
    r = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    s = re.sub(r, '', s)
    s = s.lower().split()

    s = [word for word in s if word not in stopwords.words('english')]  

    ans = []
    for word in s:
        if word in model:
            ans.append(list(model[word]))

    return list(np.mean(ans, axis=0))


def preprocess():
    model = gensim.models.KeyedVectors.load_word2vec_format('G://Python//GoogleNews-vectors-negative300.bin', binary=True)

    file = open('G://Python//test1//News_Category_Dataset_v2.json', 'r', encoding='utf-8')
    headlines = []
    cate = []

    count = 0
    for line in file.readlines():
        try:
            dic = json.loads(line)
            headline = headline2vec(dic['headline'], model)

            headlines.append(headline)
            cate.append(dic['category'])
            count += 1

        except:
            pass

        if count >= MAX_NEWS:
            break

    category_list = list(set(cate))
    category_to_id = {category_list[i]: i for i in range(len(category_list))}
    length = len(cate)
    targets = []
    for i in range(length):
        temp = np.zeros(shape=(41,))
        temp[category_to_id[cate[i]]] = 1
        targets.append(temp)

    headlines = np.array(headlines)
    targets = np.array(targets)
    print("shape info:", headlines.shape, '\n', targets.shape)

    X_train, X_test, y_train, y_test = train_test_split(headlines,
                                                        targets,
                                                        test_size=0.1,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess()

    clf = models.Sequential()
    clf.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    clf.add(layers.Dense(256, activation='relu'))
    clf.add(layers.Dense(256, activation='relu'))
    clf.add(layers.Dropout(rate=0.1))
    clf.add(layers.Dense(256, activation='relu'))
    clf.add(layers.Dropout(rate=0.1))
    clf.add(layers.Dense(41))  
    clf.add(layers.Activation('softmax'))
    clf.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    clf.fit(X_train, y_train, epochs=35, batch_size=100, verbose=2)

    

    aa = clf.predict(X_test)  # 预测情况
    for i in range(300):
        print(aa[i].argmax(), y_test[i].argmax(), '\n')

    print("RESULT ON TEST: ", clf.evaluate(X_test, y_test, batch_size=1000))
