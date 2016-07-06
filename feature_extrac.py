# -*- coding: utf-8 -*-
import nltk
import os
import pickle
import data_preprocess
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

data_path = os.getcwd() + '/data'
pos_resultfile = os.path.join(data_path, 'pos_data.txt')
neg_resultfile = os.path.join(data_path, 'neg_data.txt')
pos_word_pkl = os.path.join(data_path, 'pos_word_seq.pkl')
neg_word_pkl = os.path.join(data_path, 'neg_word_seq.pkl')
global posWords
global negWords
global bestWords

#将neg_data.txt、pos_data.txt写入pickle文件，方便再次读入
def write_pickle_file(data_txt, word_pkl):
	if not os.path.exists(data_txt):
		get_data(data_path + '/pos', data_txt)
	word_list = []
	with open(data_txt, 'r') as fin:
		for line in fin.readlines():
			rst_lst = line[:-1].split('  ')
			word_list.append(rst_lst)
	output_file = open(word_pkl, 'wb')
	pickle.dump(word_list, output_file, 1)
	output_file.close()

#从pickle文件读取词语数据到内存
def init_data():
	global posWords
	global negWords
	if not os.path.exists(pos_word_pkl):
		write_pickle_file(pos_resultfile, pos_word_pkl)
	if not os.path.exists(neg_word_pkl):
		write_pickle_file(neg_resultfile, neg_word_pkl)
	posWords = pickle.load(open(pos_word_pkl,'rb'))
	negWords = pickle.load(open(neg_word_pkl,'rb'))
	posWords = list(itertools.chain(*posWords))
	negWords = list(itertools.chain(*negWords))

#为积极和消极情感词表打分
def get_scores(posWordsLst, negWordsLst):	
	word_fd = FreqDist() 
	cond_word_fd = ConditionalFreqDist() 
	for word in posWordsLst:
		word_fd[word] += 1
		cond_word_fd['pos'][word] += 1
	for word in negWordsLst:
		word_fd[word] += 1
		cond_word_fd['neg'][word] += 1

	pos_word_count = cond_word_fd['pos'].N()
	neg_word_count = cond_word_fd['neg'].N()
	total_word_count = pos_word_count + neg_word_count

	word_scores = {}
	for word, freq in word_fd.iteritems():
		pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
		neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
		word_scores[word] = pos_score + neg_score
	return word_scores

'''
def create_word_scores():
	return get_scores(posWords, negWords)
'''

#积极情感词表=词+双词，消极情感词表=词+双词，构建两个词表然后调用get_scores对其中的词打分
def create_word_bigram_scores():
	bigram_finder = BigramCollocationFinder.from_words(posWords)
	bigram_finder = BigramCollocationFinder.from_words(negWords)
	posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
	negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
	pos = posWords + posBigrams #词和双词搭配
	neg = negWords + negBigrams
	return get_scores(pos, neg)

#提取打分前number个词作为best word,并将积极情感和消极情感样本用特征表示
def get_features(number = 1500):
	global bestWords
	init_data()
	#word_scores_1 = create_word_scores()
	word_scores_2 = create_word_bigram_scores()
	bestWords = find_best_words(word_scores_2,number)
	posFeatures = pos_features(best_word_features)
	negFeatures = neg_features(best_word_features)
	return posFeatures, negFeatures

#找到分数最高的前number个词
def find_best_words(word_scores, number):
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number] #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
	best_words = set([w for w, s in best_vals])
	return best_words

#将积极情感文本用特征表示
def pos_features(feature_extraction_func):
	posFeatures = []
	pos = pickle.load(open(pos_word_pkl,'rb'))
	for sample in pos:
		sample_tag = [feature_extraction_func(sample), 'pos']
		posFeatures.append(sample_tag)
	return posFeatures

#将消极情感文本用特征表示
def neg_features(feature_extraction_func):
	negFeatures = []
	neg = pickle.load(open(neg_word_pkl,'rb'))
	for sample in neg:
		sample_tag = [feature_extraction_func(sample), 'neg']
		negFeatures.append(sample_tag)
	return negFeatures
 
#将样本词列表表示成字典的形式
def best_word_features(words):
 	global bestWords
 	#print words
 	#print bestWords
 	#print dict([(word, True) for word in words if word in bestWords])
 	return dict([(word, True) for word in words if word in bestWords])

if __name__ == '__main__':
	get_features(number = 1500)
