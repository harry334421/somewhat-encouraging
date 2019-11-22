import jieba
import numpy as np

def get_word_vector():
    s1 = input("title0：")
    s2 = input("title1：")
    
    cut1 = jieba.cut(s1)
    cut2 = jieba.cut(s2)
    
    list_word1 = (','.join(cut1)).split(',')
    list_word2 = (','.join(cut2)).split(',')
    
    key_word = list(set(list_word1 + list_word2))#句子并集
    print(key_word)
    
    word_vector1 = np.zeros(len(key_word))#给定形状和类型的用0填充的矩阵存储向量
    word_vector2 = np.zeros(len(key_word))
  
    for i in range(len(key_word)):#依次确定向量的每个位置的值
        for j in range(len(list_word1)):#遍历key_word中每个词在句子中的出现次数
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    print(word_vector1)
    print(word_vector2)
    return word_vector1, word_vector2

def cosine():
    v1, v2 = get_word_vector()
    return float(np.sum(v1 * v2))/(np.linalg.norm(v1) * np.linalg.norm(v2)) 

print('cosine distance:',1-cosine())

#result:"0.109006418193463"