---
layout: post
title: NLP入门实例推荐（Tensorflow实现）
category: NLP
keywords: NLP
---
自然语言处理(NLP)是机器学习的应用之一，用于分析、理解和生成自然语言，以便人类与计算机，人类与人类更好的交流。自然语言处理按照任务类型可以分为分类、匹配、翻译、结构化预测、与序贯决策过程这五类。自然语言处理中的绝大多数问题皆可归入下图其中的一个[1]。这为我们学习自然语言处理提供了大的指导方向，让我们可以心无旁骛的寻找、理解和复现论文。

![这里写图片描述](http://img.blog.csdn.net/20171105155659740)

在近些年自然语言处理发展的过程中，有如下趋势[2]：
第一，传统的基于句法-语义规则的理性主义方法受到质疑，大规模分布式处理真是文本成为自然语言处理的主要战略目标。
第二，统计数学方法越来越受到重视，自然语言处理中越来越多地使用机器自动学习的方法来获取语言知识。深度学习一枝独秀，在很多领域超越传统机器学习方法，成为state-of-the-art的方法。

实际上，在近年来的自然语言处理方向的顶会上，深度学习也往往占据了大量的篇幅，自言语言处理方向成为模型与计算能力的较量。本专栏将聚焦深度学习方法在自然语言处理中的应用，给出多种任务的深度学习处理的方法，以期帮助入门者少走弯路，坚定地朝着大牛的方向前进。

本专栏的项目实现语言为Tensorflow。Tensorflow由google团队开发，神经网络结构的代码的简洁度，分布式深度学习算法的执行效率，还有部署的便利性使它是当前最火的深度学习框架。Tensorflow对CNN和RNN（LSTM）等常见模型有很好的支持，并且当前计算机视觉和自然语言处理的论文在github上都会有仿真，也为入门tensorflow提供了极大的便利。下图为Tensorflow与当前主流的深度学习框架对比：

![这里写图片描述](http://img.blog.csdn.net/20171020160303468)

网上关于tensorflow的教程也很多，tensorflow的官方文档写的很烂，但还是建议入门的同学先通读完[get_started](https://www.tensorflow.org/get_started/get_started)。遇到不懂的函数还是首选官方文档，选择version>=1.0。因为很多中文翻译的文档或者网络上的博客，版本还都停留在1.0之前，很多函数方法都已经进行了更新，所以还是会直接看官方文档进行学习效果更好。

感谢老师给我们宽松的学习环境，让我们研二的几个人可以学习自己感兴趣的知识。所以这学期想找一些实例和竞赛来练一下手。根据最近半年学习tensorflow的感受，列出了以下几个在github上自然语言处理方向优秀的论文项目，并附上我个人对论文及项目的导读。

 1. [MNIST(NN)](https://www.tensorflow.org/get_started/mnist/beginners)

 2. [MNIST(CNN)](https://www.tensorflow.org/get_started/mnist/pros)

 3. [Text Classification(CNN)](https://github.com/dennybritz/cnn-text-classification-tf)
 
 4. 导读：[Sentence Similarity(CNN)](http://blog.csdn.net/irving_zhang/article/details/69440789)
项目：[Tensorflow实例-CNN处理句子相似度](https://github.com/Irvinglove/TF_Sentence_Similarity_CNN)

 5.  论文：[Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](https://pdfs.semanticscholar.org/0f69/24633c56832b91836b69aedfd024681e427c.pdf)
 项目：[github](https://github.com/Irvinglove/MP-CNN-Tensorflow-sentence-similarity)
导读：[Tensorflow实例-CNN处理句子相似度（MPCNN）](http://blog.csdn.net/irving_zhang/article/details/70036708)

 6.  论文：[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)
 项目：[github](https://github.com/Irvinglove/char-CNN-text-classification-tensorflow)
 导读：[基于字符的卷积神经网络实现文本分类（char-level CNN）-论文详解及tensorflow实现](http://blog.csdn.net/irving_zhang/article/details/75634108)

 7. 文章：[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
项目：[github](https://github.com/sherjilozair/char-rnn-tensorflow)
导读：[基于循环神经网络实现基于字符的语言模型（char-level RNN Language Model）-tensorflow实现](http://blog.csdn.net/irving_zhang/article/details/76038710)

 8.  论文：[Implementation of Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
 项目：[github](https://github.com/Irvinglove/HAN-text-classification)
 导读：[Implementation of Hierarchical Attention Networks for Document Classification的讲解与Tensorflow实现](http://blog.csdn.net/irving_zhang/article/details/77868620)
 9. 论文：[Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/pdf/1506.07285.pdf)
 项目：[github](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow)
 导读：[Ask Me Anything: Dynamic Memory Networks for Natural Language Processing 阅读笔记及tensorflow实现](http://blog.csdn.net/irving_zhang/article/details/78113251)
 
 10.  实战：[ 基于深度学习的大规模多标签文本分类任务总结](http://blog.csdn.net/irving_zhang/article/details/78273486)

总体来说，难度由浅入深，适合初学者入门。大部分项目都会按照论文、项目、和导读三个部分，首先阅读论文，其次在github上寻找开源实现，如果其中有不懂的问题就在网上找解决办法，我想这也是大多数初学者适合的道路。在写这篇入门实例推荐时，我也会把实例中涉及的代码风格进行统一，以方便记忆。目前涉及的模型主要有：

1. Deep Neural Network(DNN)
2.  Convolution Neural Network(CNN)
3. Recurrent Neural Network(RNN)以及LSTM
4. Recurrent Convolution  Neural Network(RCNN)
5. sequence2sequence(Attention)
6. Hierarchical Attention Networks(HAN)
7. Dynamic Memory Networks(DMN)
8. EntityNetwork

以下是我在刚开本篇专栏时的一些想法，现在看起来仍然觉得很有帮助，如果你对深度学习和自然语言处理方向感兴趣，也赶快行动起来吧！

 1. 入门简单，学习有瓶颈。入门Tensorflow确实挺简单的，按照官方文档进行傻瓜式安装，然后代码一拷，就能跑通最简单的MNIST手写数字识别。但是，你会发现MNIST的训练集和测试集的读取都是封装在Tensorflow中的，所以很轻松就能得到需要的Tensor。但是，当我们需要自己读入数据的时候，就会发生各种各样的问题。所以在写Tensorflow代码解决深度学习的问题时，初学者大部分都把时间花费在数据集的处理方面，而真正使用tensorflow构建网络却并不是那么困难。掌握了处理常规数据集的方法以后，我们需要做的就是坐在电脑前面一步一步地调参了。
 
 2. Tensor!Tensor!Tensor! 重要的事情说三遍。从Tensorflow的命名方式可以看出，Tensor在graph中进行流动，以此来进行神经网络的训练。Tensor可以说是Tensorflow中最重要的概念了，同时也是Tensorflow中最容易出错的地方了。对数据进行完处理以后，就要作为input输入到Tensorflow中，在神经网络运行的每一步，都要对Tensor的shape了如指掌，否则就一定会出错！所以这其实还是基本功的问题，强烈建议如果有不明白的就返回模型的原理进行回顾。

废话就不多讲了，判断一个人有没有理解一个模型的最简单的方法，就是看他能不能写出代码，并且讲清楚自己的代码。因此在以后的文章中，我会贴出自己的代码和github地址，重点解析代码运行中容易出错的地方，以及一些深度学习中调参的小trick。

共勉。

[1] [李航NSR论文：深度学习NLP的现有优势与未来挑战](https://www.jiqizhixin.com/articles/2017-10-04-5)
[2] [NLP维基百科](https://zh.wikipedia.org/wiki/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86)