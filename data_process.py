# load the data and organize it into training and test set.

import string
import random
import numpy as np
from sklearn import preprocessing

# Global variables

TAGGED_FILENAME = "news_tagged_data.txt"
WORDVEC_FILENAME = "wordvecs.txt"
WORDVEC_LENGTH = 300
TRAINING_DATA = 0.8
WINDOW = 3
RANDOM_SEED = 300
PAD_STRING = 'O'

def loadData():

    print "Loading Sentences..."
    inFile = open(TAGGED_FILENAME, 'r')
    sentenceList = inFile.read().split('\n\n')

    if sentenceList[-1] == "":
        del sentenceList[-1]

    return sentenceList
    
    

def loadWordVec():

    print "Loading  Word2Vec Matrix..."
    wordVec = np.loadtxt(WORDVEC_FILENAME,str)
    return wordVec
    
def splitSentence(sentence,labels):

    words = sentence.split()

    if labels == False: 
        return words[0:len(words)]

    else:
        return words[0:(len(words)):2],words[1:(len(words)):2]  

def contextWindow(l, win):
    '''
    win :: size of window

    l :: numpy array containing the word or indices

    it will return a list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1    

    for i in range(win/2):
    
        if (l.dtype.char == 'S'):

            l = np.insert(l,0,PAD_STRING)
            l = np.append(l,PAD_STRING)

        else:
    
            l = np.vstack([l[0],l])
            l = np.vstack([l,l[-1]])
    
    out = np.array([])
    
    for i in range(len(l)-win+1):

        window = l[i:(i+win)]

        if out.size:

            out = np.vstack([out,window])

        else:
            
            out = window
            
    
            
    #assert len(out) == len(l)      
    return out

def getFeatures(sentence, wordVec):
    '''
    sentence :: sentence split into words
    wordVec  :: matrix of values for the words

    '''

    valArr = []
    wordList = [item.lower() for item in list(wordVec[:,0])]
    size = len(sentence)

    for i in range(size):

        if sentence[i].lower() in wordList:

            ind = wordList.index(sentence[i].lower())
            valArr.append(map(float,list(wordVec[ind,1:WORDVEC_LENGTH+1])))
            
        else:    
            ind = random.randint(0,len(wordList)-1)
            valArr.append(map(float,list(wordVec[ind,1:WORDVEC_LENGTH+1])))
                
            
    return np.array(valArr)

def encodeLabel(labels, label_set):
    '''
    label :: vector of IOB tags corresponding to the training sentence
    '''
    for i in range(len(labels)):

        labels[i] = label_set.index(labels[i])

    return labels

def decodeLabel(labels,label_set):


    for i in range(len(labels)):

        labels[i] = label_set[labels[i]]

    return labels


def getMatrixLabel(tagged, wordVec, indices):

    x = np.array([])
    y = np.array([])

    for i in indices:

        [sentence,tags] = splitSentence(tagged[i],True)
        tags = contextWindow(np.array(tags,str), WINDOW).flatten()
        featureVect = getFeatures(sentence, wordVec)
        featureVect = contextWindow(featureVect, WINDOW)

        if x.size:
            x = np.concatenate([x,featureVect], axis = 0)
        else:
            x = featureVect

        if y.size:
            y = np.concatenate([y,tags], axis = 0)
            
        else:
            y = tags

    return preprocessing.scale(x),y

def getMatrix(query, wordVec):

    x = np.array([])
    
    sentence = splitSentence(query,False)
    featureVect = getFeatures(sentence, wordVec)
    featureVect = contextWindow(featureVect, WINDOW)

    return preprocessing.scale(featureVect)

def getData():
    '''
    Returns the training feature matrix, the test feature matix
    and label vector as numpy arrays to be fed into the ML algorithm
            
    '''
    tagged = loadData()
    wordVec = loadWordVec()
    num_sentence = len(tagged)

    random.seed(RANDOM_SEED)
    complete = range(num_sentence)
    training = sorted(random.sample(range(num_sentence),int(num_sentence*TRAINING_DATA)))
    test = sorted(list(set(complete)-set(training)))

    x_train, y_train = getMatrixLabel(tagged, wordVec, training)
    x_test, y_test = getMatrixLabel(tagged, wordVec, test)
    label_set = set(y_train) | set(y_test)
    
    
    y_train = np.array(encodeLabel(list(y_train), list(label_set)),dtype = 'int32')
    y_test = np.array(encodeLabel(list(y_test), list(label_set)),dtype = 'int32')

    return x_train,y_train,x_test,y_test,wordVec,label_set
        


