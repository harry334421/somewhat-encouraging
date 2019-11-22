import gensim
from gensim.models import word2vec

if __name__=='__main__':
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
    distance = model.distance('woman', 'man')
    print(distance)
    