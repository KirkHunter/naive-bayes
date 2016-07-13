from collections import Counter
from math import log
import argparse
import os
import random
import re

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
        'you', 'your']

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args

def get_files(my_dir):
    """
    Return list of positive review filenames and 
    list of negative review filenames from directory.
    """
    pos_files = os.listdir(my_dir + "/pos")
    neg_files = os.listdir(my_dir + "/neg")
    return pos_files, neg_files

def get_words(my_dir, my_file, cls, stop_words=stop_words):
    """
    Return a counter of all the words in a text file, filter stop words.
    """
    with open(my_dir + "/" + cls + "/" + my_file) as f:
        words = re.findall("[a-z]+[\-']?[a-z]*", f.read())
    return Counter(filter(lambda x: x not in stop_words, words))

def make_file_dicts(my_dir, pos_files, neg_files):
    """
    Make a dictionary of positive filenames and a dictionary of negative
    filenames containing their respective word counters.
    """
    pos_dict = {f:get_words(my_dir, f, "pos") for f in pos_files}
    neg_dict = {f:get_words(my_dir, f, "neg") for f in neg_files}
    return pos_dict, neg_dict

def make_class_counters(pos_files, pos_dict, neg_files, neg_dict):
    """
    Return counter of words in positive train files and counter of
    words in negative train files.
    """
    pos_counter = Counter()
    neg_counter = Counter()
    [pos_counter.update(pos_dict[f]) for f in pos_files]
    [neg_counter.update(neg_dict[f]) for f in neg_files]
    return pos_counter, neg_counter

def make_test_train_sets(pos_files, neg_files):
    """
    Shuffle all files and return 2 lists of training files and one list of
    test files each of length 1/3 * len(all files)  
    """
    all_files = pos_files + neg_files
    n = len(all_files)
    i = int(1. / 3 * n) + 1 
    random.shuffle(all_files)
    return all_files[:i], all_files[i:2 * i], all_files[2 * i:]

def class_probs(train1, train2, pos_files, neg_files):
    """
    Return the proportion of positive files and negative files
    in the total training set.
    """
    all_train = train1 + train2
    len_pos_train = len(filter(lambda x: x in pos_files, all_train))
    pos_prob = len_pos_train / float(len(all_train))
    return pos_prob, 1 - pos_prob

def p_W_Given_C(class_counter, word, len_cls, V):
    try:
        c = class_counter[word]
    except KeyError:
        c = 0
    return (c + 1.0) / (len_cls + V + 1)

def make_class_prediction(test_file, n, m, V, train1, train2, pos_dict, 
                          neg_dict, pos_counter, neg_counter, log_pos_prob, 
                          log_neg_prob):
    
    """ Given a test file, returns a class prediction. """

    try:
        doc_counter = pos_dict[test_file]
    except KeyError:
        doc_counter = neg_dict[test_file]

    set_doc_words = doc_counter.keys()
    
    def c_pos(word):
        return doc_counter[word] * log(p_W_Given_C(pos_counter, word, n, V))
    
    def c_neg(word):
        return doc_counter[word] * log(p_W_Given_C(neg_counter, word, m, V))

    S_pos = map(c_pos, set_doc_words)
    S_pos = sum(S_pos)
    S_neg = map(c_neg, set_doc_words)
    S_neg = sum(S_neg)
    prob_pos_class = log_pos_prob + S_pos
    prob_neg_class = log_neg_prob + S_neg
    
    return "pos" if prob_pos_class > prob_neg_class else "neg"

def naive(test, n, m, V, train1, train2, pos_dict, neg_dict, pos_counter,
              neg_counter, pos_files, neg_files):
    """
    Make a prediction for every file in the test set, and return 
    the number of correct predictions for each class.
    """
    num_pos_correct = 0
    num_neg_correct = 0
    pos_prob, neg_prob = class_probs(train1, train2, pos_dict, neg_dict)
    log_pos_prob = log(pos_prob)
    log_neg_prob = log(neg_prob)
    
    for f in test:
        p = make_class_prediction(f, n, m, V, train1, train2, pos_dict, 
                                  neg_dict, pos_counter, neg_counter, 
                                  log_pos_prob, log_neg_prob)
        if p == "pos":
            if pos_files.count(f):
                num_pos_correct += 1
        elif neg_files.count(f):
            num_neg_correct += 1
    return num_pos_correct, num_neg_correct

def cross_validation(directory):
    pos_files, neg_files = get_files(directory)
    train1, train2, test = make_test_train_sets(pos_files, neg_files)
    pos_dict, neg_dict = make_file_dicts(directory, pos_files, neg_files)
    total_correct = 0
    total = 0
    for i in (1, 2, 3):
        all_train = train1 + train2
        pos_train_files = filter(lambda f: f in pos_files, all_train)
        neg_train_files = filter(lambda f: f in neg_files, all_train)
        pos_counter, neg_counter = make_class_counters(pos_train_files, 
                                                       pos_dict,
                                                       neg_train_files, 
                                                       neg_dict)
        n = sum(pos_counter.values())                       
        m = sum(neg_counter.values())
        V = len(set(pos_counter.keys() + neg_counter.keys()))

        num_pos_train_docs = len(pos_train_files)
        num_pos_test_docs = len(filter(lambda x: x in pos_files, test))
        num_neg_train_docs = len(neg_train_files)
        num_neg_test_docs = len(filter(lambda x: x in neg_files, test))
        results = naive(test, n, m, V, train1, train2, pos_dict, neg_dict,
                        pos_counter, neg_counter, pos_files, neg_files)
        num_correct = sum(results)
        accuracy = float(num_correct) / len(test)
        print "iteration %d:" % i
        print "    num_pos_test_docs: %d" % num_pos_test_docs
        print "    num_pos_training_docs: %d" % num_pos_train_docs
        print "    num_pos_correct_docs: %d" % results[0]
        print "    num_neg_test_docs: %d" % num_neg_test_docs
        print "    num_neg_training_docs: %d" % num_neg_train_docs
        print "    num_neg_correct_docs: %d" % results[1]
        print "    accuracy: %s" % '{:.1%}'.format(accuracy)
        total_correct += num_correct
        total += len(test)
        temp_train1 = train1
        temp_train2 = train2
        temp_test = test
        train1 = temp_train2
        train2 = temp_test
        test = temp_train1
    ave_accuracy = total_correct / float(total)
    print "ave_accuracy: %s" % '{:.1%}'.format(ave_accuracy)

def main():
    args = parse_argument()
    directory = args['d'][0]
    cross_validation(directory)
    
if __name__ == "__main__":
    main()
