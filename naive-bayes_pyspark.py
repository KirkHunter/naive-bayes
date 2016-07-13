
# coding: utf-8

from pyspark import SparkContext
from collections import Counter
from math import log
import re

sc = SparkContext()

stop_words = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also',
        'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be',
        'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
        'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for',
        'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers',
        'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is',
        'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
        'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor',
        'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our',
        'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since',
        'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
        'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
        've', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
        'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet',
        'you', 'your', '', '~', '-']



AWS_ACCESS_KEY_ID = "AKIAI2RULIS6VKBC2A4A"
AWS_SECRET_ACCESS_KEY = "TUsq2P4Hmq74zd4jzeuEVgSzuUaexcD5z2DuRZED"

sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)

pos_train_file = sc.textFile('s3n://sparkstuff/train_pos.txt')
neg_train_file = sc.textFile('s3n://sparkstuff/train_neg.txt')

pos_test_file = sc.textFile('s3n://sparkstuff/test_pos.txt')
neg_test_file = sc.textFile('s3n://sparkstuff/test_neg.txt')



def remove_punctuation(line):
    return [re.sub('(<br)|(/><br)|(</)|(/>)|[!.,?:"();\s\*]', '', w.lower()) for w in line]


def filter_stop_words(line):
    return filter(lambda w: w not in stop_words, line)


def class_probs(len_pos_train, len_neg_train):
    """
    Return the proportion of positive files and negative files
    in the total training set.
    """
    len_all_train = len_pos_train + len_neg_train
    pos_prob = len_pos_train / float(len_all_train)
    return pos_prob, 1 - pos_prob


def prob_word_given_class(class_counter, word, len_cls, V):
    try:
        c = class_counter[word]
    except KeyError:
        c = 0
    return (c + 1.0) / (len_cls + V + 1)


def make_class_prediction(test_file, n, m, V, pos_counter, neg_counter, 
                          log_pos_prob, log_neg_prob):
    
    """ Given a test file, returns a class prediction. """

    doc_counter = Counter(test_file)

    set_doc_words = doc_counter.keys()
    
    def c_pos(word):
        return doc_counter[word] * log(prob_word_given_class(pos_counter, 
                                                             word, n, V))
    
    def c_neg(word):
        return doc_counter[word] * log(prob_word_given_class(neg_counter, 
                                                             word, m, V))

    S_pos = map(c_pos, set_doc_words)
    S_pos = sum(S_pos)
    S_neg = map(c_neg, set_doc_words)
    S_neg = sum(S_neg)
    prob_pos_class = log_pos_prob + S_pos
    prob_neg_class = log_neg_prob + S_neg
    
    return 1 if prob_pos_class > prob_neg_class else 0



if __name__ == '__main__':

    print "\n\t*********************************************\n"
    print "\n\tRunning file...\n"

    stop_words = {w:'' for w in stop_words}

    print "\n\tReading text files...\n"

    pos_train_raw = pos_train_file.map(lambda line: line.split())
    neg_train_raw = neg_train_file.map(lambda line: line.split())

    pos_test_raw = pos_test_file.map(lambda line: line.split())
    neg_test_raw = neg_test_file.map(lambda line: line.split())


    print "\n\tParsing text...\n"

    pos_train = pos_train_raw.map(remove_punctuation)
    neg_train = neg_train_raw.map(remove_punctuation)

    pos_test = pos_test_raw.map(remove_punctuation)
    neg_test = neg_test_raw.map(remove_punctuation)

    for i in range(4):
        pos_train = pos_train.map(remove_punctuation)
        neg_train = neg_train.map(remove_punctuation)
        
        pos_test = pos_test.map(remove_punctuation)
        neg_test = neg_test.map(remove_punctuation)

    pos_train = pos_train.map(filter_stop_words)
    neg_train = neg_train.map(filter_stop_words)

    pos_test = pos_test.map(filter_stop_words)
    neg_test = neg_test.map(filter_stop_words)


    pos_train_words = pos_train.flatMap(lambda line: [(w, 1) for w in line])
    pos_train_counter = pos_train_words.reduceByKey(lambda x, y: x + y).collectAsMap()

    neg_train_words = neg_train.flatMap(lambda line: [(w, 1) for w in line])
    neg_train_counter = neg_train_words.reduceByKey(lambda x, y: x + y).collectAsMap()


    n = sum(pos_train_counter.values())
    m = sum(neg_train_counter.values())
    V = len(set(pos_train_counter.keys() + neg_train_counter.keys()))
    log_pos_prob, log_neg_prob = log(.5), log(.5)


    pos_test_labeled = pos_test.map(lambda line: {"text":line, "label":1})
    neg_test_labeled = neg_test.map(lambda line: {"text":line, "label":0})

    testing = pos_test_labeled.union(neg_test_labeled)

    test_labels = testing.map(lambda p: p["label"])
    test_docs = testing.map(lambda p: p["text"])


    print "\n\tClassifying test observations...\n"
    
    pred = test_docs.map(lambda f: make_class_prediction(f, n, m, V,
                                                        pos_train_counter,
                                                        neg_train_counter,
                                                        log_pos_prob, log_neg_prob))



    labels = test_labels.collect()
    preds = pred.collect()

    predictions = zip(labels, preds)

    len_predictions = len(preds)

    num_correct = sum([1 if label == p else 0 for label, p in predictions]) 

    print "\n\tAccuracy: %.4f\n" % (num_correct / float(len(predictions)))

