---
layout: post
title: Tensorflow实例-CNN处理句子相似度（MPCNN）
category: NLP
keywords: NLP
---
前两篇使用CNN处理句子相似度的文章，实际上使用了很简单的CNN模型，按照自己对CNN的理解实现了代码。完成以后，就开始翻阅论文，想了解一些经典的处理句子相似度的方法。 [这篇文章](https://pdfs.semanticscholar.org/0f69/24633c56832b91836b69aedfd024681e427c.pdf) 发表在仅次于ACL会的EMNLP会议上，使用多个粒度窗口大小的卷积滤波器，后面跟着多种类型的池化方式，目的是为了从多个角度去解析句子（Multi-perspective），尽可能多的提取句子的语义和句法结构，具体的方式大家可以参阅论文，这里给出一个 [实验室同学](http://my.csdn.net/liuchonge) 找到的关于本篇文章的具体卷积和池化的[tensor变化PDF](http://pan.baidu.com/s/1nvHOso5)，在此感谢刘同学对我的指导，顺利完成本篇代码，代码已上传[github](https://github.com/Irvinglove/MP-CNN-Tensorflow-sentence-similarity)。
	![这里写图片描述](http://img.blog.csdn.net/20170410205914337)
	如上图，底层左右两个句子计算相似度；由于是多个角度去解析句子，就采用了多个窗口大小和多个卷积方式来处理句子，是倒数第二层；接着就跟了一个相似度计算层，使用多种计算相似度的方式，如cos函数和第一第二欧氏距离；接着就再跟一个全连接层，用来调整输出。
	1、闲话不多说，来看代码，具体实现的时候只采用了一种计算相似度的方法，就是cos函数，再加两种方法跟该方法类似。由此就可以计算出论文中algorithm 1中的feah的大小。这里值得一提的是tf.diag_part()函数，取对称矩阵的对角元素作为一个新的向量，因为我们只需要对应channel的相似度，所以只取对角元素就可以了。
	注意feah的tensor的shape为[batch,1,1,filter_nums]
	

```
# ws11_x1 窗口大小为ws1 = 1，句子x1的tensor
# ws130_x2 窗口大小为ws1 = 30， 句子x2的tensor
def cul_feah_sim(ws11_x1, ws11_x2, ws12_x1, ws12_x2, ws130_x1, ws130_x2):
    x1_concat = tf.concat([ws11_x1, ws12_x1, ws130_x1], 3)
    x2_concat = tf.concat([ws11_x2, ws12_x2, ws130_x2], 3)
    x1_flat = tf.reshape(x1_concat, [-1, 1, 3, 100])
    x2_flat = tf.reshape(x2_concat, [-1, 1, 3, 100])
    regM_matmul = tf.matmul(x1_flat, x2_flat, transpose_a=True)
    feah = []
    for batch in range(50):
        feah.append(tf.diag_part(regM_matmul[batch][0]))
    feah_flat = tf.reshape(feah,[-1,1,1,100])
    return feah_flat
```
2、 algorithm 2中的feaa同样也只用cos函数结果表示相似度。

```
def cul_feaa_sim(ws11_x1, ws11_x2, ws12_x1, ws12_x2, ws130_x1, ws130_x2):
    x1_concat = tf.concat([ws11_x1, ws12_x1, ws130_x1], 3)
    x2_concat = tf.concat([ws11_x2, ws12_x2, ws130_x2], 3)
    x1_flat = tf.reshape(x1_concat, [-1, 1, 3, 100])
    x2_flat = tf.reshape(x2_concat, [-1, 1, 3, 100])
    regM_matmul = tf.matmul(x1_flat, x2_flat, transpose_b=True)
    feaa = tf.reshape(regM_matmul,[-1,1,1,9])
    return feaa
```
![图二](http://img.blog.csdn.net/20170410211221637)
在1、和2、中，卷积和池化的tensor的变化如上图所示，得到的max池化方式的tensor的shape是[batch，1，1，filter_nums]，其他池化方式一样。参照下图对各种池化方式计算相似度的直观描述，可以推测出具体的返回结果feah和feaa的shape。
![这里写图片描述](http://img.blog.csdn.net/20170410213011566)
3、而algorithm 2中的feab的卷积方式就和图二不同，而是如下图所示：
![这里写图片描述](http://img.blog.csdn.net/20170410213223958)
因此计算feab的方式知识在feah的基础上改变了tensor的形状，在算法的伪代码中也可以清楚地看出来，因此这里就不在贴出feab的代码，感兴趣的同学可以到该代码的地址下载并运行，观察代码进行分析：
![这里写图片描述](http://img.blog.csdn.net/20170410213502246)

4、写代码的过程中，再次感受到在tensorflow中，对梯度求导、最优化算法这种数学推算要求不高，都已经在tensorflow中封装的很好了，而最重要的就是对数据的处理，将数据从文本信息，一步步变为input的tensor，然后再不停地使用tf.reshape、tf.concat、tf.stack、tf.matmul等函数，就可以构造出一个比较基础的神经网络架构。然后就可以慢慢调参了。