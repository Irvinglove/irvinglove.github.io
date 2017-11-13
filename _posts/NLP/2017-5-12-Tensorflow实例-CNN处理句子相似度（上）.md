---
layout: post
title: Tensorflow实例-CNN处理句子相似度（上）
category: NLP
keywords: NLP
---
在开始阅读本篇之前，希望你已经看过[cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)，使用CNN做文本分类项目，start两千多。因为很经典，网上的解读也随处可见，因此就不介绍。但是看了这个项目，可以了解tensorflow构建项目的关键步骤，可以养成良好的代码习惯，这在初学者来说是很重要的。

Tensorflow中关键的两个步骤，首先对数据进行处理，转化为合适的tensor作为input输入到图中。其次使用tensorflow对神经网络进行构建。本篇文章作为深度学习实战的第一篇，会尽量用通俗的语言解释在这两部分中的关键的点。本篇任务为计算句子相似度，代码上传至[github](https://github.com/Irvinglove/TF_Sentence_Similarity_CNN)(2017年10月24日对代码进行重构)，感兴趣的同学可以下载并进行仿真，加深对tensorflow的理解。代码较长，因此分为上下两篇对该实例进行讲解。上篇讲解句子相似度问题中数据处理的相关工作。

1、 读取glove词向量文件。

glove文件的共有40000个词，词在每一行开头，其余的为该词的词向量。我们用notepad打开，在第一行添加400000 50两个数字，中间用空格或者\t隔开，400000代表词的个数，50代表词向量的维度。这样我们就可以方便的通过word2vec工具读取词向量。
![这里写图片描述](http://img.blog.csdn.net/20171024144032751)


选用glove文件是google预先训练好的文件，这里新建了两个变量sr_word2id, word_embedding。

sr_word2id的数据类型是pandas工具中的Series，在数据处理中非常常用，可以当做是一个字典，根据word来得到id，负责将数据集中的word转成id。

word_embedding负责将word转成的id转成词向量，shape为[vocabulary_size, embedding_size]，在本例中是由[400000, 50]，行号和sr_word2id中的id对应。

word_embedding变量存在是因为我们在实际操作过程中发现，如果直接使用词向量作为input的话，那么input中传入的shape就是[s_count, sentence_length, embedding_size]，其中，s_count是数据集大小，sentence_length就是最大句子长度，embedding_length就是词向量的大小。

那么placeholder的维度就太大，然后程序就变得特别卡顿。所以，参考[Text Classification(CNN)](https://github.com/dennybritz/cnn-text-classification-tf)的方法，使用embedding层进行转换。即我们使用词的索引值，而不是词的向量作为input。那么input的shape为[s_count, sentence_length]，以词汇在word_embedding的索引值代替词向量，传入placeholder后再通过tf.nn.embedding_lookup函数将id变为词向量，这样就大大节省了placeholder的空间。

比如句子“cat sit on the mat”，一般的思路是用“cat”,"sit","on","the","mat"这五个单词的词向量表示这句话:

```
[[1,1,1],
[2,2,2],
[3,3,3],
[4,4,4],
[5,5,5]]
```
那么整个数据集为9840行，placeholder就变成了9840*5*3大小的tensor，采用embedding_w以后呢，我们采用“cat”,"sit","on","the","mat"这五个词在embedding_w中的索引代替该词:
```
[1,2,3,4,5]
```
然后placeholder就变成了9840*5大小的tensor，然后在之后的操作中，根据索引查找embedding_w的词向量即可，当embedding_size为3时，影响不大，但是当embedding_size为50、100或者更大时，节省的时间就很可观了。

```
def build_glove_dic():
    # 从文件中读取 pre-trained 的 glove 文件，对应每个词的词向量
    # 需要手动对glove文件处理，在第一行加上
    # 400000 50
    # 其中400000代表共有四十万个词，每个词50维，中间为一个空格或者tab键
    # 因为word2vec提取需要这样的格式，详细代码可以点进load函数查看
    glove_path = 'glove.6B.50d.txt'
    wv = word2vec.load(glove_path)
    vocab = wv.vocab
    sr_word2id = pd.Series(range(1,len(vocab) + 1), index=vocab)
    sr_word2id['<unk>'] = 0
    word_embedding = wv.vectors
    word_mean = np.mean(word_embedding, axis=0)
    word_embedding = np.vstack([word_mean, word_embedding])

    return sr_word2id, word_embedding
```
word_embedding中有400000个词，那么肯定也会有文中出现，但是不在word_embedding中的词，这个时候通过sr_word2id查找是找不到id的，那么我们就使用这400000个词向亮的均值代表未知的词，使用$<unk>$表示。


2、读取数据集。本实例中数据集选取[SICK](http://clic.cimec.unitn.it/composes/sick.html)，在句子相似度领域经典的数据集。该数据集主要包括两个句子和它们之间的相似度。整个函数的想法就是从数据集中提取数据，使用s1，s2，score存储两个句子和它们之间的相似度。然后对句子进行处理，记数据集中句子的最大长度sentence_length，不够该长度的使用‘unk’进行填充。
值得一提的是，这里的s1_image和s2_image的形状是[s_count,sentence_length
]，label的形状是[s_count,1]。这里说道的s1_image和s2_image刚好和placeholder的input的形状保持一致，然后通过
```
import pandas as pd
import numpy as np

def read_data_sets(train_dir):
    #
    # s1代表数据集的句子1
    # s2代表数据集的句子2
    # score代表相似度
    # sample_num代表数据总共有多少行
    #
    SICK_DIR = "SICK_data/SICK.txt"
    df_sick = pd.read_csv(SICK_DIR, sep="\t", usecols=[1,2,4], names=['s1', 's2', 'score'],
                          dtype={'s1':object, 's2':object, 'score':object})
    df_sick = df_sick.drop([0])
    s1 = df_sick.s1.values
    s2 = df_sick.s2.values
    score = np.asarray(map(float, df_sick.score.values), dtype=np.float32)
    sample_num = len(score)

    # 引入embedding矩阵和字典
    global sr_word2id, word_embedding
    sr_word2id, word_embedding = build_glove_dic()

    # word2id, 多线程将word转成id
    p = Pool()
    s1 = np.asarray(p.map(seq2id, s1))
    s2 = np.asarray(p.map(seq2id, s2))
    p.close()
    p.join()

    # 填充句子
    s1, s2 = padding_sentence(s1, s2)
    new_index = np.random.permutation(sample_num)
    s1 = s1[new_index]
    s2 = s2[new_index]
    score = score[new_index]

    return s1 ,s2, score

def get_id(word):
    if word in sr_word2id:
        return sr_word2id[word]
    else:
        return sr_word2id['<unk>']

def seq2id(seq):
    seq = clean_str(seq)
    seq_split = seq.split(' ')
    seq_id = map(get_id, seq_split)
    return seq_id


```
3、填充句子
```
def padding_sentence(s1, s2):
    #
    # 得到句子s1,s2以后，很直观地想法就是先找出数据集中的最大句子长度，
    # 然后用<unk>对句子进行填充
    #
    s1_length_max = max([len(s) for s in s1])
    s2_length_max = max([len(s) for s in s2])
    sentence_length = max(s1_length_max, s2_length_max)
    sentence_num = s1.shape[0]
    s1_padding = np.zeros([sentence_num, sentence_length], dtype=int)
    s2_padding = np.zeros([sentence_num, sentence_length], dtype=int)

    for i, s in enumerate(s1):
        s1_padding[i][:len(s)] = s

    for i, s in enumerate(s2):
        s2_padding[i][:len(s)] = s

    print "9840个句子填充完毕"
    return s1_padding, s2_padding
```
注意：在处理数据的过程中，要把列表转化成np数组，不然传入placeholder中会一直报错“setting an array element with a sequence”。np数组的转换也很简单，使用np.array()就可以进行转换，但是np数组在插入过程不是很熟练，因此也可以采用先对列表进行操作，在转换为np数组。但一定要注意，多维的列表的每一层都需要转化成np数组，可以在tensorflow的调试过程中慢慢处理。