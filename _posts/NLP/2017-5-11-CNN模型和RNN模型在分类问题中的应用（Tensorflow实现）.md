---
layout: post
title: CNN模型和RNN模型在分类问题中的应用（Tensorflow实现）
category: NLP
keywords: NLP
---
在这篇文章中，我们将实现一个卷积神经网络和一个循环神经网络语句分类模型。 本文提到的模型（rnn和cnn）在一系列文本分类任务（如情绪分析）中实现了良好的分类性能，并且由于模型简单，方便实现，成为了竞赛和实战中常用的baseline。

[cnn-text-classification-tf博客](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)，使用CNN做文本分类项目，start两千多。阅读这个[项目源码](https://github.com/dennybritz/cnn-text-classification-tf)，可以了解tensorflow构建项目的关键步骤，可以养成良好的代码习惯，这在初学者来说是很重要的。[原始论文Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)

[rnn-text-classification-tf](https://github.com/Irvinglove/rnn-text-classification-tf)，是我以CNN的源码基础，使用RNN做文本分类项目，实现了类似的分类性能。下面只讲解CNN，略过RNN，感兴趣的同学可以把RNN也clone下来自己跑一边。自行给出两个代码的性能比较。

数据处理
--
数据集是 Movie Review data from Rotten Tomatoes，也是原始文献中使用的数据集之一。 数据集包含,包含5331个积极的评论和5331个消极评论，正负向各占一半。 数据集不附带拆分的训练/测试集，因此我们只需将10％的数据用作 dev set。数据集过小容易过拟合，可以进行10交叉验证。在github项目中只是crude将数据集以9:1的比例拆为训练集和验证集。
步骤：
1. 加载两类数据
2. 文本数据清洗
3. 把每个句子填充到最大的句子长度，填充字符是<PAD>，使得每个句子都包含59个单词。相同的长度有利于进行高效的批处理
4. 根据所有单词的词表，建立一个索引，用一个整数代表一个词，则每个句子由一个整数向量表示

```
# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

```

模型构建
--
模型结构图为：
![这里写图片描述](http://img.blog.csdn.net/20171105173542986)
在github项目中，对该模型有适当的改变。第一层把词嵌入到低维向量；第二层使用多个不同大小的filter进行卷积（分别为3,4,5）；第三层用max-pool把第二层多个filter的结果转换成一个长的特征向量并加入dropout正规化；第四层用softmax进行分类。
```
# Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

```
模型训练
--
模型训练部分的代码都是固定的套路，熟悉以后非常方便写出来。
1. summaries汇总
tensorflow提供了各方面的汇总信息，方便跟踪和可视化训练和预测的过程。summaries是一个序列化的对象，通过SummaryWriter写入到光盘
2. checkpointing检查点
用于保存训练参数，方便选择最优的参数，使用tf.train.saver()进行保存
3. 变量初始化
sess.run(tf.initialize_all_variables())，用于初始化所有我们定义的变量，也可以对特定的变量手动调用初始化，如预训练好的词向量
4. 定义单一的训练步骤
定义一个函数用于模型评价、更新批量数据和更新模型参数
feed_dict中包含了我们在网络中定义的占位符的数据，必须要对所有的占位符进行赋值，否则会报错
train_op不返回结果，只是更新网络的参数
5. 训练循环
遍历数据并对每次遍历数据调用train_step函数，并定期打印模型评价和检查点

训练结果
--
这里上传博客中的两个结果图，上图为loss变化，下图为accuracy的变化。实际仿真结果和下图相同，在测试集上的准确率为0.6-0.7之间，效果并不是很好。原因如下：
![这里写图片描述](http://img.blog.csdn.net/20171105174131934)
1. 训练的指标不是平滑的，原因是我们每个批处理的数据过少
2. 训练集正确率过高，测试集正确率过低，过拟合。避免过拟合：更多的数据；更强的正规化；更少的模型参数。例如对最后一层的权重进行L2惩罚，使得正确率提升到76%，接近原始paper。

