# _*_ coding: utf-8 _*_
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('floresta')
nltk.download('machado')
nltk.download('mac_morpho')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    divide a sentença em uma array de palavras/tokens
    um token pode ser uma palavra, caractere de pontuação ou número
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = encontrar a forma base da palavra
    exemplos:
    palavras = ["organize", "organizes", "organizing"]
    palavras = [stem(w) for w in palavras]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    retorna um array de bag of words:
    1 para cada palavra conhecida que existe na sentença, 0 caso contrário
    exemplo:
    sentença = ["olá", "como", "está", "você"]
    palavras = ["oi", "olá", "eu", "você", "tchau", "obrigado", "legal"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem de cada palavra
    sentence_words = [stem(word) for word in tokenized_sentence]
    # inicializa a bag com 0 para cada palavra
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
