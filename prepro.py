# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2019/11/25 0025 19:47
# @Author   : yuenobel
# @File     : prepro.py.py
# @Software : PyCharm


'''该文件用于生产源语言和目标文件'''
from __future__ import print_function
from hyperparams import HyperParams as hp
import codecs
import os
import regex
from collections import Counter

#作用是构建一个词频词典，把训练数据集中的所有词分出来统计频率，以倒序形式输入到一个文档中
def make_vocab(fpath, fname):
    '''
    结构化数据
    :param fpath: 输入文件
    :param fname: 处理后的输出文件
    '''
    text = codecs.open(fpath, 'r', 'utf-8').read()
    #[...]匹配列出的任意字符
    #[^...]代表匹配为列出的任意字符，所以这里应该可以理解为只匹配空格和latin编码文字
    text = regex.sub('[^\s\p{Latin}]', '', text)#^匹配开头，\s匹配空白字符，\S匹配非空白字符 
    words = text.split()
    word2cnt = Counter(words) #Counter类计算出每个单词的词频
    if not os.path.exists('./preprocessed'):
        os.mkdir('./preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write('{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n'.
                   format('<PAD>', '<UNK>', '<S>', '</S>'))
        for word, cnt in word2cnt.most_common(len(word2cnt)):   # 按照单词出现的频率写入文件  most_common用于实现topn功能，在此处的作用是将统计出的词频从大到小排序写入
            fout.write('{}\t{}\n'.format(word, cnt)) #以特定输出格式写入


if __name__ == '__main__':
    make_vocab(hp.source_train, 'de.vocab.tsv')
    make_vocab(hp.target_train, 'en.vocab.tsv') #此处原来数据调用的是hp.source.test 是否有错?
    print('Done')
