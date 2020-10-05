#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import scipy.stats
import nltk
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic


""" Part 1 """

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')


part1_data_path = '/Users/clairevanpoperin/LING83600 langtech/lab1-cvanpop-master/data/ws353.tsv'


#read data to list of first synset sense of each word pair and human score
def get_sense_lst():
    sense_lst = []
    with open (part1_data_path, 'r') as data_file:
        data = csv.reader(data_file, delimiter='\t')
        count = 0
        count_found = 0   
        for row in data:
            count += 1
            w1 = row[0]
            w2 = row[1]
            h_score = row[2]
            w1_sense = wordnet.synsets(w1)[0]
            w2_sense = wordnet.synsets(w2)[0]
            if w1_sense and w2_sense:
                count_found += 1
                sense_lst.append((w1_sense, w2_sense, h_score))
    #print('Wordnet Sense:')
    #print('Total pairs: ', count)
    #print('Found pairs: ', count_found)
    #print('Percent found: ', round(count_found/count,4))
    return(sense_lst)




"""Path Similarity"""

def get_path_sim():
    sense_lst2 = get_sense_lst()
    path_sim_lst = []
    path_sim_human_score_lst = []
    count_found_path_sim = 0
    for pair in sense_lst2:
        path_sim = pair[0].path_similarity(pair[1])
        if path_sim:
            count_found_path_sim += 1
            path_sim_lst.append(path_sim)
            path_sim_human_score_lst.append(pair[2])
        else:
            continue
    (stat, p) = scipy.stats.spearmanr(path_sim_lst, path_sim_human_score_lst)
    spearman_rho = round(stat, 4)
    print('Path Similarity:')    
    print('Spearman Rho: ', spearman_rho)
    print('Total pairs: ', len(sense_lst2))
    print('Found pairs: ', count_found_path_sim)
    print('Percent found: ', round(count_found_path_sim/len(sense_lst2),4))
    return path_sim_lst



"""Resnik Similarity"""

def get_res_sim():
    sense_lst2 = get_sense_lst()
    res_sim_lst = []
    res_sim_human_score_lst = []
    count_found_res_sim = 0
    for pair in sense_lst2:
        if pair[0].pos() == pair[1].pos():
            count_found_res_sim += 1
            res_sim = pair[0].res_similarity(pair[1], semcor_ic)
            res_sim_lst.append(res_sim)
            res_sim_human_score_lst.append(pair[2])
        else:
            continue
    (stat, p) = scipy.stats.spearmanr(res_sim_lst, res_sim_human_score_lst)
    spearman_rho = round(stat, 4)
    print('Resnik Similarity:')    
    print('Spearman Rho: ', spearman_rho)
    print('Total pairs: ', len(sense_lst2))
    print('Found pairs: ', count_found_res_sim)
    print('Percent found: ', round(count_found_res_sim/len(sense_lst2),4))    



"""Leacock-Chodorow Similarity"""

def get_lch_sim():
    sense_lst2 = get_sense_lst()
    lch_sim_lst = []
    lch_sim_human_score_lst = []
    count_found_lch_sim = 0
    for pair in sense_lst2:
        if pair[0].pos() == pair[1].pos():
            count_found_lch_sim += 1
            lch_sim = pair[0].lch_similarity(pair[1])
            lch_sim_lst.append(lch_sim)
            lch_sim_human_score_lst.append(pair[2])
        else:
            continue
    (stat, p) = scipy.stats.spearmanr(lch_sim_lst, lch_sim_human_score_lst)
    spearman_rho = round(stat, 4)
    print('Leacock-Chodorow Similarity:')    
    print('Spearman Rho: ', spearman_rho)
    print('Total pairs: ', len(sense_lst2))
    print('Found pairs: ', count_found_lch_sim)
    print('Percent found: ', round(count_found_lch_sim/len(sense_lst2),4))
     



     
    
"""Jiang-Conrath Similarity"""
def get_jcn_sim():
    sense_lst2 = get_sense_lst()
    jcn_sim_lst = []
    jcn_sim_human_score_lst = []
    count_found_jcn_sim = 0
    for pair in sense_lst2:
        if pair[0].pos() == pair[1].pos():
            count_found_jcn_sim += 1
            jcn_sim = pair[0].jcn_similarity(pair[1], brown_ic)
            jcn_sim_lst.append(jcn_sim)
            jcn_sim_human_score_lst.append(pair[2])
        else:
            continue
    (stat, p) = scipy.stats.spearmanr(jcn_sim_lst, jcn_sim_human_score_lst)
    spearman_rho = round(stat, 4)
    print('Jiang-Conrath Similarity:')    
    print('Spearman Rho: ', spearman_rho)
    print('Total pairs: ', len(sense_lst2))
    print('Found pairs: ', count_found_jcn_sim)
    print('Percent found: ', round(count_found_jcn_sim/len(sense_lst2),4))

    

"""Lin Similarity"""

def get_lin_sim():
    sense_lst2 = get_sense_lst()
    lin_sim_lst = []
    lin_sim_human_score_lst = []
    count_found_lin_sim = 0
    for pair in sense_lst2:
        if pair[0].pos() == pair[1].pos():
            count_found_lin_sim += 1
            lin_sim = pair[0].lin_similarity(pair[1], brown_ic)
            lin_sim_lst.append(lin_sim)
            lin_sim_human_score_lst.append(pair[2])
        else:
            continue
    (stat, p) = scipy.stats.spearmanr(lin_sim_lst, lin_sim_human_score_lst)
    spearman_rho = round(stat, 4)
    print('Lin Similarity:')     
    print('Spearman Rho: ', spearman_rho)
    print('Total pairs: ', len(sense_lst2))
    print('Found pairs: ', count_found_lin_sim)
    print('Percent found: ', round(count_found_lin_sim/len(sense_lst2),4))



"""Wu Palmer Similarity"""

def get_wup_sim():
    sense_lst2 = get_sense_lst()
    wup_sim_lst = []
    wup_sim_human_score_lst = []
    count_found_wup_sim = 0
    for pair in sense_lst2:
        if pair[0].pos() == pair[1].pos():
            count_found_wup_sim += 1
            wup_sim = pair[0].wup_similarity(pair[1], brown_ic)
            wup_sim_lst.append(wup_sim)
            wup_sim_human_score_lst.append(pair[2])
        else:
            continue
    (stat, p) = scipy.stats.spearmanr(wup_sim_lst, wup_sim_human_score_lst)
    spearman_rho = round(stat, 4)
    print('Wu Palmer Similarity:')   
    print('Spearman Rho: ', spearman_rho)
    print('Total pairs: ', len(sense_lst2))
    print('Found pairs: ', count_found_wup_sim)
    print('Percent found: ', round(count_found_wup_sim/len(sense_lst2),4))


#calls
"""
path_sim = get_path_sim()
lch_sim = get_lch_sim()
res_sim = get_res_sim()
jcn_sim = get_jcn_sim()
lin_sim = get_lin_sim()
wup_sim = get_wup_sim()
"""



""" Part 2 """


data_path = "news.2007.en.shuffled.deduped"
tokenized_text_path = "tokenized_text.txt"

#tokenize source file
def tokenize_source(data_path):
    with open(data_path, 'r') as data, open(tokenized_text_path, "a") as sink:
        for line in data:
            sentence_tokens = nltk.word_tokenize(line)
            joined = " ".join(sentence_tokens)
            case_folded_joined = joined.casefold()
            print(case_folded_joined, file=sink) 




source_file = '/Users/clairevanpoperin/LING83600 langtech/lab1-cvanpop-master/data/ws353.tsv'
output_file = 'token_pairs.tsv'


#write two column of tsv of source file
def get_word_pairs():
    with open(source_file, "r") as source, open(output_file, "w") as sink:
        reader = csv.reader(source, delimiter="\t")
        writer = csv.writer(sink, delimiter="\t")  
        for row in reader:
            writer.writerow(row[:2])

            

#calculate spearman rho based on output of ppmi.py, results_ppmi.tsv            
ppmi_tsv = '/Users/clairevanpoperin/results_ppmi.tsv'
original_tsv = '/Users/clairevanpoperin/LING83600 langtech/lab1-cvanpop-master/data/ws353.tsv'

def get_spearmanr_ppmi(original_tsv, ppmi_tsv):
    with open(original_tsv, newline='') as original:
        reader = csv.reader(original, delimiter='\t')
        orig_pairs = []
        count = 0
        count_found = 0
        for row in reader:
            count += 1
            pair = ((row[0], row[1]), row[2])
            orig_pairs.append(pair)
    with open(ppmi_tsv, newline='') as ppmi:
        ppmi_reader = csv.reader(ppmi, delimiter='\t')
        human_score_matches = []
        ppmi_score_matches = []
        for ppmi_row in ppmi_reader:
            ppmi_pair1 = (ppmi_row[0], ppmi_row[1])
            ppmi_pair2 = (ppmi_row[1], ppmi_row[0])
            for pair in orig_pairs:
                if ppmi_pair1 == pair[0] or ppmi_pair2 == pair[0]:
                    count_found += 1
                    human_score_matches.append(pair[1])
                    ppmi_score_matches.append(ppmi_row[2])
    (stat, p) = scipy.stats.spearmanr(human_score_matches, ppmi_score_matches)
    spearman_rho = round(stat, 4)
    print('PPMI Similarity:')    
    print('Spearman Rho: ', spearman_rho)
    print('Total pairs: ', count)
    print('Found pairs: ', count_found)
    print('Percent found: ', round(count_found/count,4))

   

#calls
"""
text = tokenize_source(data_path)
word_pairs = get_word_pairs()
ppmi = get_spearmanr_ppmi(original_tsv, ppmi_tsv)
"""

#command line call ppmi.py
"""
python ppmi.py --results_path results_ppmi_window15.tsv --pairs_path token_pairs.tsv --tok_path tokenized_text.txt --window 4

--window WINDOW       symmetric window size (default: 10)


"""

#logging info
"""
INFO: 277 words tracked
INFO: 203 pairs tracked
INFO: 152218018 tokens counted
INFO: 152 pairs covered
"""


""" Part 3 """
#calculate spearman rho based on output of word2vec.py, results_word2vec.tsv            

word2vec_tsv = '/Users/clairevanpoperin/results_word2vec.tsv'
original_tsv = '/Users/clairevanpoperin/LING83600 langtech/lab1-cvanpop-master/data/ws353.tsv'

def get_spearmanr_word2vec(original_tsv, word2vec_tsv):
    with open(original_tsv, newline='') as original:
        reader = csv.reader(original, delimiter='\t')
        orig_pairs = []
        count = 0
        count_found = 0
        for row in reader:
            count += 1
            pair = ((row[0], row[1]), row[2])
            orig_pairs.append(pair)
    with open(word2vec_tsv, newline='') as word2vec:
        word2vec_reader = csv.reader(word2vec, delimiter='\t')
        human_score_matches = []
        word2vec_score_matches = []
        for word2vec_row in word2vec_reader:
            word2vec_pair1 = (word2vec_row[0], word2vec_row[1])
            word2vec_pair2 = (word2vec_row[1], word2vec_row[0])
            for pair in orig_pairs:
                if word2vec_pair1 == pair[0] or word2vec_pair2 == pair[0]:
                    count_found += 1
                    human_score_matches.append(pair[1])
                    word2vec_score_matches.append(word2vec_row[2])
                
    (stat, p) = scipy.stats.spearmanr(human_score_matches, word2vec_score_matches)
    spearman_rho = round(stat, 4)
    print('Word2Vec Similarity:')    
    print('Spearman Rho: ', spearman_rho)
    print('Total pairs: ', count)
    print('Found pairs: ', count_found)
    print('Percent found: ', round(count_found/count,4))



#command line for word2vec.py
"""
python word2vec.py --results_path results_word2vec.tsv --pairs_path token_pairs.tsv --tok_path tokenized_text.txt
word2vec = get_spearmanr_word2vec(original_tsv, word2vec_tsv)
"""

#logging info
"""
INFO: training on a 761090090 raw words (568813539 effective words) took 1504.1s, 378183 effective words/s
word2vec.py:28: DeprecationWarning: Call to deprecated `similarity` (Method will be removed in 4.0.0, use self.wv.similarity() instead).
  score = round(w2v.similarity(x, y), 6)
"""  