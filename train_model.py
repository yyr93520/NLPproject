# -*- coding: utf-8 -*-
import feature_extrac
import sklearn
import pickle
import os
import getopt
import data_preprocess
import sys

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier 

posFeatures, negFeatures = feature_extrac.get_features(number=1500)
trainData = posFeatures[:1414] + negFeatures[:1722]
#validData = posFeatures[1238:1414] + negFeatures[1506:1722]
testData = posFeatures[1414:] + negFeatures[1722:]
#validSam, validTag = zip(*validData)
testSam, testTag = zip(*testData)
data_path = os.getcwd() + '/data/model/'

def frange(x, y, jump):
	while x < y:
		yield x
		x += jump

#通用的模型训练和测试函数
def train_model(classifier, name, printout = False):
	classifier = SklearnClassifier(classifier)
	classifier.train(trainData)
	#predict = classifier.classify_many(validSam)
	predict = classifier.classify_many(testSam)
	accuracy = accuracy_score(testTag, predict)
	if printout:
		print '*******模型: %s的测试结果*********' % name
		print '\n'
		print '%s`s accuracy is %f' % (name, accuracy)
		print '%s`s score report is \n' % name
		print classification_report(testTag, predict)
		print '%s`s confusion is \n' % name
		print confusion_matrix(testTag, predict)
		print '\n'
		model_file = data_path + name + ".pkl"
		pickle.dump(classifier, open(model_file, 'w'))
	return accuracy

#伯努利朴素贝叶斯模型调参
#best alpha = 5.15, best binarize = 0,  best accuracy = 0.846
def BernoulliNB_param():
	best_alpha = 0
	best_binarize = 0
	best_accuracy = 0
	for a in frange(5,5.5,0.05):
		for b in frange(0,1,0.1):
			accuracy = train_model(BernoulliNB(alpha = a, binarize=b), 'BernoulliNB')
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_alpha = a
	print best_alpha
	print best_binarize
	print best_accuracy

#多项式朴素贝叶斯模型调参
#alpha = 0.02, best accuracy = 0.862
def MultinomialNB_param():
	best_alpha = 0
	best_accuracy = 0
	for a in frange(0,0.04,0.005):
		accuracy = train_model(MultinomialNB(alpha = a), 'MultinomialNB')
		print a
		print accuracy
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_alpha = a
	print best_alpha
	print best_accuracy

#随机森林调参
#n_estimators = 13  best_accuracy = 0.855
def RandomForestClassifier_param():
	best_n_estimators = 0
	best_accuracy = 0
	for a in range(1,21,1):
		accuracy = train_model(RandomForestClassifier(n_estimators = a), 'LogisticRegression')
		print a, accuracy
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_n_estimators = a
	print best_n_estimators	
	print best_accuracy

#Logistic回归调参
##C = 8.87, intercept_scaling = 0.153, best accuracy = 0.864
def LogisticRegression_param():
	best_C = 0
	best_accuracy = 0
	best_intercept_scaling = 0
	best_verbose = 0
	for a in frange(8.87,8.9,0.01):
		for b in frange(0.15, 0.16, 0.001):
			accuracy = train_model(LogisticRegression(C = a, intercept_scaling=b), 'LogisticRegression')
			print a,b,accuracy	
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_C = a
				best_intercept_scaling = b
	print best_C	
	print best_intercept_scaling
	print best_accuracy

#线性SVM分类器调参
##C = 6.005, intercept_scaling = 5.345, best accuracy = 0.861
def LinearSVC_param():
	best_C = 0
	best_accuracy = 0
	best_intercept_scaling = 0
	for a in frange(6.0,6.04,0.005):
		for b in frange(5.34, 5.38, 0.005):
			accuracy = train_model(LinearSVC(C = a, intercept_scaling=b), 'LogisticRegression')
			print a,b, accuracy
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_C = a
				best_intercept_scaling = b
	print best_C	
	print best_intercept_scaling
	print best_accuracy

#按调参后的结果训练所有模型，并生成模型的pkl文件
def valid_model():
	train_model(BernoulliNB(alpha = 5.15), 'BernoulliNB', printout = True)
	train_model(MultinomialNB(alpha = 0.02), 'MultinomialNB', printout = True)
	train_model(RandomForestClassifier(n_estimators = 20), 'RandomForestClassifier', printout = True)
	train_model(LogisticRegression(C = 8.87, intercept_scaling = 0.153), 'LogisticRegression', printout = True)
	train_model(LinearSVC(C = 6.005, intercept_scaling = 5.345), 'LinearSVC', printout = True)

#根据所有模型训练的结果投票，投票后占多数的标签是最终的分类标签，输出最终的测试准确度分析结果
def vote_model(printout = True, test = testSam):
	predict_all = []
	for parent, dirnames, filenames in os.walk(data_path):
		for filename in filenames:
			model_file = os.path.join(parent, filename)
			classifier = pickle.load(open(model_file,'r'))
			predict_all.append(classifier.classify_many(test))
	predict = []
	for i in range(0, len(test)):
		pos_num = 0
		neg_num = 0
		for j in range(0, 5):
			if predict_all[j][i] == 'pos':
				pos_num += 1
			else:
				neg_num += 1
		predict.append('pos' if pos_num > neg_num else 'neg')
	if printout == True:
		print '\n'
		print '*******模型融合的最终测试结果*********'		
		print '\n'
		print '训练集：积极样本1414条，消极样本1722条'
		print '测试集：积极样本354条，消极样本430条\n'
		print 'accuracy is %f' % accuracy_score(testTag, predict)
		print 'score report is \n'
		print classification_report(testTag, predict)
		print 'confusion is \n' 
		print confusion_matrix(testTag, predict)
		print '\n'	
	return predict

def get_model_tag(input_file, output_file):
	words_list, text_list = data_preprocess.repre_test_data(input_file)
	test_list = []
	for words in words_list:
		test_list.append(feature_extrac.best_word_features(words))
	predict = vote_model(printout = False, test = test_list)
	with open(output_file, 'w') as fout:
		for i in range(0, len(predict)):
			fout.write(predict[i] + ":  " + text_list[i] + "\n")
	print '情感分析结果已写入到文件：%s\n' % output_file

def usage():
	print 'Usage:'
	print 'python train_model.py [opts]'
	print '-t input_file output_file: input_file给出测试数据文件（./data/input_file.txt），output_file给出结果文件（默认为./data/output_file.txt）'
	print '-r: 输出模型融合的测试结果'
	print '-d: 输出5个模型分别的测试结果'

if __name__ == '__main__':
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'rdt:', [])
	except Exception, ex:
		print ex
		usage()
		exit()
	if len(opts) == 0:
		usage()
		exit()
	for name, value in opts:
		if name == '-r':
			vote_model()
		elif name == '-d':
			valid_model()
		elif name == '-t':
			input_file = value
			if len(args) > 0:
				output_file = args[0]
			else:
				output_file = os.getcwd() + '/data/output_file.txt'
			get_model_tag(input_file, output_file)
		else:
			usage()
			exit()


