import sys
import os
_snlp_book_dir = "../../../../"
sys.path.append(_snlp_book_dir) 
import statnlpbook.lm as lm
import statnlpbook.ohhla as ohhla
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

_snlp_train_dir = _snlp_book_dir + "/data/ohhla/train"
_snlp_dev_dir = _snlp_book_dir + "/data/ohhla/dev"
_snlp_train_song_words = ohhla.words(ohhla.load_all_songs(_snlp_train_dir))
_snlp_dev_song_words = ohhla.words(ohhla.load_all_songs(_snlp_dev_dir))
assert(len(_snlp_train_song_words)==1041496)

from statnlpbook.lm import *

 
#laplace lm -- 547.8596406456934
"""
my_LM = lm.NGramLM(oov_train, 1)
my_LM = lm.LaplaceLM(my_LM, 0.004)
"""

#laplace lm -- 300.97971886348046
"""
my_LM = lm.NGramLM(oov_train, 2)
my_LM = lm.LaplaceLM(my_LM, 0.004)
"""

#Interpolatedlm -- 277.52895847161693
"""
my_LM1 = lm.NGramLM(oov_train, 2)
my_LM2 = lm.NGramLM(oov_train, 2)
my_LM = lm.LaplaceLM(my_LM1,0.04)
my_LM = lm.InterpolatedLM(my_LM2, my_LM, 0.526)
"""

#Interpolatedlm -- 189.05623239905705
"""
my_LM1 = lm.NGramLM(oov_train, 1)
my_LM2 = lm.NGramLM(oov_train, 2)
my_LM = lm.LaplaceLM(my_LM1,0.04)
my_LM = lm.InterpolatedLM(my_LM2,my_LM,0.714)
"""

#StupidBackoff -- 126.36654628232233 ///FALSE
"""
my_LM1 = lm.NGramLM(oov_train, 1)
my_LM2 = lm.NGramLM(oov_train, 2)
my_LM = lm.LaplaceLM(my_LM1,0.04)
my_LM = lm.StupidBackoff(my_LM2,my_LM,0.99)

"""
"""
# --- 164.6857647840312
my_LM1 = lm.NGramLM(oov_train, 1)
my_LM2 = lm.NGramLM(oov_train, 2)
my_LM3 = lm.LaplaceLM(my_LM1,0.04)
my_LM4 = lm.InterpolatedLM(my_LM2,my_LM3,0.714)
my_LM5 = lm.NGramLM(oov_train,3)
my_LM = lm.InterpolatedLM(my_LM5, my_LM4, 0.217)


#Interpolatedlm2 -- 193.04639944406102
unigram = lm.LaplaceLM(lm.NGramLM(oov_train, 1),0.4)
bigram = lm.LaplaceLM(lm.NGramLM(oov_train, 2),0.04)
trigram = lm.NGramLM(oov_train, 3)
my_LM = InterpolatedLM2 (trigram, bigram, unigram, 0.3, 0.4)


my_LM1 = lm.NGramLM(oov_train, 1)
my_LM2 = lm.NGramLM(oov_train, 2)
my_LM = lm.LaplaceLM(my_LM1,0.04)
my_LM = StupidBackoffNormalized(my_LM2,my_LM,0.8)
## You should improve this cell
"""
oov_train = lm.inject_OOVs(_snlp_train_song_words)
oov_def = lm.inject_OOVs(_snlp_dev_song_words)
oov_vocab = set(oov_train)

def create_lm(vocab):
    """
    Return an instance of `lm.LanguageModel` defined over the given vocabulary.
    Args
        vocab: the vocabulary the LM should be defined over. It is the union of the training and test words.
    Returns:
        a language model, instance of `lm.LanguageModel`.
    """
    return lm.OOVAwareLM(my_LM, vocab - oov_vocab)


_snlp_test_dir = _snlp_book_dir + "/data/ohhla/dev"
_snlp_test_song_words = ohhla.words(ohhla.load_all_songs(_snlp_test_dir))
_snlp_test_vocab = set(_snlp_test_song_words)
_snlp_dev_vocab = set(_snlp_dev_song_words)
_snlp_train_vocab = set(_snlp_train_song_words)
_snlp_vocab = _snlp_test_vocab | _snlp_train_vocab | _snlp_dev_vocab


"""
_snlp_lm = create_lm(_snlp_vocab)

_snlp_test_token_indices = [100, 1000, 10000]
_eps = 0.000001
for i in _snlp_test_token_indices:
    result = sum([_snlp_lm.probability(word, *_snlp_test_song_words[i-_snlp_lm.order+1:i]) for word in _snlp_vocab])
    print("Sum: {sum}, ~1: {approx_1}, <=1: {leq_1}".format(sum=result, 
                                                            approx_1=abs(result - 1.0) < _eps, 
                                                            leq_1=result - _eps <= 1.0))
print(lm.perplexity(_snlp_lm, _snlp_test_song_words))
"""

# find appropriate alpha
"""
my_LM1 = lm.NGramLM(oov_train, 1)
my_LM2 = lm.NGramLM(oov_train, 2)
my_LM3 = lm.LaplaceLM(my_LM1,0.04)
my_LM4 = lm.InterpolatedLM(my_LM2,my_LM3,0.714)
my_LM5 = lm.NGramLM(oov_train,3)

for alpha in np.arange(0,1,0.1):
    my_LM = lm.InterpolatedLM(my_LM5, my_LM4, alpha)
    _snlp_lm = create_lm(_snlp_vocab)
    print(alpha,lm.perplexity(_snlp_lm, _snlp_test_song_words))
 
my_LM1 = lm.NGramLM(oov_train, 1)
my_LM2 = lm.NGramLM(oov_train, 2)
my_LM3 = lm.LaplaceLM(my_LM1,0.04)
my_LM4 = lm.LaplaceLM(my_LM2, 0.04)

for alpha in np.linspace(0,1,10):
    my_LM = StupidBackoffNormalized(my_LM4, my_LM3, alpha)
    _snlp_lm = create_lm(_snlp_vocab)
    print(alpha,lm.perplexity(_snlp_lm, _snlp_test_song_words))

"""  
import collections
counts = collections.defaultdict(int)
for word in oov_train:
    counts[word] += 1
# how many times the frequency r occurs
sorted_counts = sorted(counts.values(),reverse=True)
# the frequency r
ranks = range(1,len(sorted_counts)+1)

class KatzSmoothing(LanguageModel):
    def __init__(self, main, backoff,k):
        super().__init__(main.vocab, main.order)
        self.main = main
        self.backoff = backoff
        self.k = k

    def probability(self, word, *history):
        r = counts[word]
        if (r > 0 and r <= self.k):
            
            r_star = (r+1) * (sorted_counts[r+1]/sorted_counts[r])
            dr = ((r_star/r) - ((self.k + 1) * sorted_counts[self.k + 1] / sorted_counts[1])) / \
                (1 - ((self.k+1) * sorted_counts[self.k + 1] / (sorted_counts[1])))
                
            return dr * self.backoff.counts((word,)+tuple(history[:-1])) / r 
        elif (r > self.k):
            return self.backoff.counts((word,)+tuple(history[:-1])) / r
        else:
            return (1 - np.sum(self.main.probability(word, *history))) / (1 - np.sum(self.backoff.probability(word, *history)))



class InterpolatedLM2(LanguageModel):
    def __init__(self, main, backoff1,backoff2, alpha1, alpha2):
        super().__init__(main.vocab, main.order)
        self.main = main
        self.backoff1 = backoff1
        self.backoff2 = backoff2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def probability(self, word, *history):
        return self.alpha1 * self.main.probability(word, *history) + \
                self.alpha2 * self.backoff1.probability(word, *history) + \
               (1.0 - self.alpha1 - self.alpha2) * self.backoff2.probability(word, *history)



class StupidBackoffNormalized(LanguageModel):
    def __init__(self, main, backoff, alpha):
        super().__init__(main.vocab, main.order)
        self.main = main
        self.backoff = backoff
        self.alpha = alpha               

    def probability(self, word, *history):
        main_counts = self.main.counts((word,)+tuple(history))
        main_norm = self.main.norm(history)        
        backoff_order_diff = self.main.order - self.backoff.order
        backoff_counts = self.backoff.counts((word,)+tuple(history[:-backoff_order_diff]))
        backoff_norm = self.backoff.norm(history[:-backoff_order_diff])        
        counts = main_counts + self.alpha * backoff_counts
        norm = main_norm + self.alpha * backoff_norm
        return counts / norm


class AbsoultDiscount(LanguageModel):
    def __init__(self, main, backoff, alpha):
        super().__init__(main.vocab, main.order)
        self.main = main
        self.backoff = backoff
        self.alpha = alpha

"""
    def probability(self, word, *history):
        main_counts = self.main.counts((word,)+tuple(history))
        main_norm = self.main.norm(history)   
        backoff_counts = self.backoff.counts((word,)+tuple(history[1:]))
        backoff_norm = self.backoff.norm(history[1:])
        lambda_ = 
        if main_norm > 0:
            return main_counts/backoff_counts - self.d
        else:
            return self.d
"""
unigram = lm.LaplaceLM(lm.NGramLM(oov_train, 1),0.4)
bigram = lm.NGramLM(oov_train, 2)
trigram = lm.NGramLM(oov_train, 3)
quadgram = lm.NGramLM(oov_train, 4)
pentagram = lm.NGramLM(oov_train, 5)
sixgram = lm.NGramLM(oov_train, 6)
    
my_LM1=lm.InterpolatedLM(bigram,unigram,0.7)        
my_LM2=lm.InterpolatedLM(trigram,my_LM1,0.17)
my_LM3=lm.InterpolatedLM(quadgram,my_LM2,0.08)
my_LM4=lm.InterpolatedLM(pentagram,my_LM3,0.01)

for alpha1 in np.arange(0,0.2,0.01):
    my_LM = lm.InterpolatedLM (sixgram,my_LM4, alpha1)
    _snlp_lm = create_lm(_snlp_vocab)
    print(alpha1, lm.perplexity(_snlp_lm, _snlp_test_song_words))

"""
unigram = lm.LaplaceLM(lm.NGramLM(oov_train, 1),0.4)
bigram = lm.NGramLM(oov_train, 2)
trigram = lm.NGramLM(oov_train, 3)
quadgram = lm.NGramLM(oov_train, 4)
pentagram = lm.NGramLM(oov_train, 5)
sixgram = lm.NGramLM(oov_train, 6)

my_LM1=lm.InterpolatedLM(bigram,unigram,0.7)    
my_LM2=lm.InterpolatedLM(trigram,my_LM1,0.17)
my_LM3=lm.InterpolatedLM(quadgram,my_LM2,0.022)
for alpha1 in np.arange(0.06,0.1,0.01):
    my_LM=lm.InterpolatedLM(pentagram,my_LM3,alpha1)
    _snlp_lm = lm.OOVAwareLM(my_LM, _snlp_vocab - oov_vocab)
    print(alpha1,lm.perplexity(_snlp_lm, _snlp_test_song_words))


my_LM1=lm.InterpolatedLM(bigram,unigram,0.7)    
my_LM2=lm.InterpolatedLM(trigram,my_LM1,0.17)
my_LM3=lm.InterpolatedLM(quadgram,my_LM2,0.02)
for alpha1 in np.arange(0.04,0.07,0.01):
    for alpha2 in np.arange(0.03,0.07,0.01):
        my_LM=InterpolatedLM2(sixgram,pentagram,my_LM3,alpha1,alpha2)
        _snlp_lm = lm.OOVAwareLM(my_LM, _snlp_vocab - oov_vocab)
        print(alpha1,alpha2,lm.perplexity(_snlp_lm, _snlp_test_song_words))


unigram = lm.LaplaceLM(lm.NGramLM(oov_train, 1),0.4)
bigram = lm.NGramLM(oov_train, 2)
trigram = lm.NGramLM(oov_train, 3)
quadgram = lm.NGramLM(oov_train, 4)
pentagram = lm.NGramLM(oov_train, 5)
for alpha1 in np.arange(0.7,0.75,0.01):
    my_LM1=lm.InterpolatedLM(bigram,unigram,alpha1)
    for alpha2 in np.arange(0.2,0.4,0.01):
        my_LM2=lm.InterpolatedLM(trigram,my_LM1,alpha2)
        for alpha3 in np.arange(0,0.4,0.1):
            my_LM3=lm.InterpolatedLM(quadgram,my_LM2,alpha3)
            for alpha4 in np.arange(0,0.4,0.1):
                my_LM=lm.InterpolatedLM(pentagram,my_LM1,alpha4)
                _snlp_lm = lm.OOVAwareLM(my_LM, _snlp_vocab - oov_vocab)
                print(alpha1,alpha2,alpha3,alpha4,lm.perplexity(_snlp_lm, _snlp_test_song_words))
"""