# -*- coding: utf-8 -*-
import os
import sys
import jieba
import re
import chardet
import jieba.posseg as pseg

pos_dirroot = os.getcwd() + '/data/pos'
pos_resultfile = os.path.join(os.getcwd() + '/data', 'pos_data.txt')
neg_dirroot = os.getcwd() + '/data/neg'
neg_resultfile = os.path.join(os.getcwd() + '/data', 'neg_data.txt')
#jieba.load_userdict(file_name) user-define dictionary

#从样本数据集pos、neg中读取样本，分词，去除停用词后写入pos_data.txt，neg_data.txt
def get_data(dirroot, resultfile):
	stop = [line.strip() for line in open(os.getcwd() + '/data/stopword.txt', 'r').readlines() ]
	
	with open(resultfile, 'w') as fout:
		for parent, dirnames, filenames in os.walk(dirroot):
			for filename in filenames:
				file = os.path.join(parent, filename)
				with open(file, 'r') as fin:
					for line in fin.readlines():
						line = line.strip().decode('gb2312', 'ignore').encode('utf-8')
						if len(line) > 0:
							seg_list = pseg.cut(line)
							rst_str = ''
							for cut_word in seg_list:
								if cut_word.flag[0] != 'x' :
									cut_word = cut_word.word.strip().encode('utf-8')
									if cut_word not in stop:
										rst_str += cut_word + "  "
							if len(rst_str) > 0:
								fout.write(rst_str[:-2] + '\n')

if __name__ == '__main__':
	get_data(pos_dirroot, pos_resultfile)
	get_data(neg_dirroot, neg_resultfile)


							
