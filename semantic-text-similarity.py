import nltk 
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.collocations import *


'''

Which are more similar?

1. Deer and Elk
2. Deer and Giraffe
3. Deer and Horse

'''

# Using path similarity
# Find appropriate sense of word (Noun(n) - First meaning(01))

deer = wn.synset('deer.n.01')
elk = wn.synset('elk.n.01')
horse = wn.synset('horse.n.01')

print "Path similarity"
print "Deer and Elk" , deer.path_similarity(elk)
print "Deer and Horse", deer.path_similarity(horse)

# Using information criteria 
# Lin similarity 

brown_ic = wordnet_ic.ic('ic-brown.dat')

print "\nLin similarity"
print "Deer and Elk", deer.lin_similarity(elk, brown_ic)
print "Deer and Horse", deer.lin_similarity(horse, brown_ic)


# Collocations

print "\nCollocations"
bigram_measures = nltk.collocations.BigramAssocMeasures()
text = nltk.corpus.genesis.words('english-web.txt')
finder = BigramCollocationFinder.from_words(text)
print finder.nbest(bigram_measures.pmi, 10) # These words may be infrequent. So, we can apply frequency filter
freq = 30
print "\nCollocations with frequency filter of ", freq
finder.apply_freq_filter(freq)  # Words with frequency more than 3
print finder.nbest(bigram_measures.pmi, 10)