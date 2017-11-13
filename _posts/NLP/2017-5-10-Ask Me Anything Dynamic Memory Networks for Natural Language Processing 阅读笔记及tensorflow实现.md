---
layout: post
title: Ask Me Anything Dynamic Memory Networks for Natural Language Processing 阅读笔记及tensorflow实现
category: NLP
keywords: NLP
---
本篇要介绍的论文：Ask Me Anything: Dynamic Memory Networks for Natural Language Processing 是DMN（Dynamic Memory Networks）的开端，在很多任务上都实现了state-of-the-art的结果，如：question answering (Facebook’s bAbI dataset), text classification for sentiment analysis (Stanford Sentiment Treebank) and sequence modeling for part-of-speech tagging (WSJ-PTB)。在[github](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow)上有具体的代码实现，因此本篇文章就该代码进行讲解，而不再具体实现。（本文依然是按照[brightmart](https://github.com/brightmart/text_classification) 项目选择的模型，该项目是做知乎多文本分类的任务。）

数据集
===

本篇文章是做问答系统，选择的数据集为[BABI数据集](https://research.fb.com/downloads/babi)，是FACEBOOK的AI实验室为自动阅独理解和问答系统所制作的数据集。QA的数据集下包含二十种不同任务，包括single-supporting-fact，two-supporting-fact以其中最简单的single-supporting-fact为例，文件的格式如下：
```
ID text
ID text
ID text
ID question[tab]answer[tab]supporting fact IDS.
```
具体打开qa1_single-supporting-fact_train文件，可以看到：
```
1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary?        bathroom        1
4 Daniel went back to the hallway.
5 Sandra moved to the garden.
6 Where is Daniel?      hallway 4
7 John moved to the office.
8 Sandra journeyed to the bathroom.
9 Where is Daniel?      hallway 4
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel?     office  11
13 John went back to the garden.
14 John moved to the bedroom.
15 Where is Sandra?     bathroom        8
1 Sandra travelled to the office.
2 Sandra went to the bathroom.
3 Where is Sandra?      bathroom        2
```
数据集总共包含四个部分：Context，Question， Answer， SupportContext。以上面数据为例，
第一行和第二行代表Context；
第三行问号之前为Question；
问号之后会有一个Answer；
Answer之后跟着一个数值SupportContext，即为支持这个答案得出context的ID。（如果这个ID唯一，则为single-supporting-fact，如果这个ID为2个或者3个，就会叫不同的任务名称。）
值得注意的是，当每句话之前的ID重新变为1时，意味着一段新的Context开始，否则随着ID递增，可以看做一个Context的内容信息不断添加，但都属于一个Context。上面带分号的叙述，可以当做处理数据的规则，因此处理结果应该如下图所示：
<img src="http://img.blog.csdn.net/20170927154108862" width = "1200px"  align=center />
Context直到第六句才进行更新，否则即使有Question，Answer和SupportContext的出现，Context也只进行append，而不是从头更新。
这里只截取数据处理中的关键函数load_babi进行注释：
```
def load_babi(config, split_sentences=False):
    vocab = {}
    ivocab = {}
    # 通过get_babi_raw函数得到如上图所示的数据
    babi_train_raw, babi_test_raw = get_babi_raw(config.babi_id, config.babi_test_id)

    if config.word2vec_init:
        assert config.embed_size == 100
        word2vec = load_glove(config.embed_size)
    else:
        word2vec = {}

    # set word at index zero to be end of sentence token so padding with zeros is consistent
    process_word(word = "<eos>", 
                word2vec = word2vec, 
                vocab = vocab, 
                ivocab = ivocab, 
                word_vector_size = config.embed_size, 
                to_return = "index")

    print('==> get train inputs')
    # 对babi_train_raw进行处理，得到train_data列表，共包含5项，分别是
    # inputs, questions, answers, input_masks, relevant_labels
    # 其中单词都使用index进行表示，后续会在embedding层进行lookup_table，并进行学习
    train_data = process_input(babi_train_raw, config.floatX, word2vec, vocab, ivocab, config.embed_size, split_sentences)
    print('==> get test inputs')
    # 同上
    test_data = process_input(babi_test_raw, config.floatX, word2vec, vocab, ivocab, config.embed_size, split_sentences)

    if config.word2vec_init:
        assert config.embed_size == 100
        # 初始化嵌入层
        word_embedding = create_embedding(word2vec, ivocab, config.embed_size)
    else:
        word_embedding = np.random.uniform(-config.embedding_init, config.embedding_init, (len(ivocab), config.embed_size))

    inputs, questions, answers, input_masks, rel_labels = train_data if config.train_mode else test_data

    if split_sentences:
        # 代码中split_sentences为true
        # input_lens ==> 10000维，代表每一个问题的context的个数
        # sen_lens ==> 10000维，代表每一个问题的每一个context的长度，如第一行[5,5]
        # max_sen_len ==> value=6，代表最大的一个context的长度为6
        input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)
        max_mask_len = max_sen_len
    else:
        input_lens = get_lens(inputs)
        mask_lens = get_lens(input_masks)
        max_mask_len = np.max(mask_lens)
    # q_lens ==> 10000维，代表每一个问题的长度,每一个问题都是3维的
    q_lens = get_lens(questions)

    max_q_len = np.max(q_lens)
    max_input_len = min(np.max(input_lens), config.max_allowed_inputs)

    #pad out arrays to max
    if split_sentences:
        # 对inputs进行填充，上面代码中得到每一个问题最多有10个context，每一个context最多有6个单词，因此
        # inputs的大小变为==>[10000,10,6]
        inputs = pad_inputs(inputs, input_lens, max_input_len, "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))
    else:
        inputs = pad_inputs(inputs, input_lens, max_input_len)
        input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")

    # 对questions进行填充，上面代码中得到每一个问题最多有3个单词，因此
    # inputs的大小变为==>[10000,3]
    questions = pad_inputs(questions, q_lens, max_q_len)

    answers = np.stack(answers)

    # 这里不太清楚为什么使用全零矩阵来处理rel_labels,应该是作者的笔误
    rel_labels = np.zeros((len(rel_labels), len(rel_labels[0])))

    for i, tt in enumerate(rel_labels):
        rel_labels[i] = np.array(tt, dtype=int)
        
    if config.train_mode:
        # 返回questions， inputs，q_lens，input_lens，input_masks, answers, rel_labels
        # 然后在DMN的Model中对这些数据进行处理
        train = questions[:config.num_train], inputs[:config.num_train], q_lens[:config.num_train], input_lens[:config.num_train], input_masks[:config.num_train], answers[:config.num_train], rel_labels[:config.num_train] 

        valid = questions[config.num_train:], inputs[config.num_train:], q_lens[config.num_train:], input_lens[config.num_train:], input_masks[config.num_train:], answers[config.num_train:], rel_labels[config.num_train:] 
        return train, valid, word_embedding, max_q_len, max_input_len, max_mask_len, rel_labels.shape[1], len(vocab)

    else:
        test = questions, inputs, q_lens, input_lens, input_masks, answers, rel_labels
        return test, word_embedding, max_q_len, max_input_len, max_mask_len, rel_labels.shape[1], len(vocab)
```
模型构建
==
通过数据层的处理，我们总结一下得到的重要的nparray有哪些：
questions ==> [10000,3]
inputs ==>[10000,10,6]
answers ==> [10000,1]
rel_labels ==>[10000,1]
建议不是很理解这几个变量的shape的同学再看一下数据层的处理过程。接下来我们分析模型的结构以及如何处理这几个重要的变量。原paper上给出了下图，共包含Question层，Input层，Episodic Memory层和Answer层：
![这里写图片描述](http://img.blog.csdn.net/20170927164831901)
对应的代码为inference函数：
```
def inference(self):
        """Performs inference on the DMN model"""

        # set up embedding
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
         
        # input fusion module
        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get question representation')
            q_vec = self.get_question_representation(embeddings)
         

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            fact_vecs = self.get_input_representation(embeddings)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.config.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs, i)

                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                            self.config.hidden_size,
                            activation=tf.nn.relu)

            output = prev_memory

        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(output, q_vec)

        return output
```

下面我们通过四个部分，来描述各层如何处理这些数据。

Question层：
----------

question的占位符shape为[self.config.batch_size, self.max_q_len]，即[100,3]，通过嵌入层，shape变为[100,3,80]。代码为：
```
    def get_question_representation(self, embeddings):
        """Get question vectors via embedding and GRU"""
        questions = tf.nn.embedding_lookup(embeddings, self.question_placeholder)

        gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        _, q_vec = tf.nn.dynamic_rnn(gru_cell,
                questions,
                dtype=np.float32,
                sequence_length=self.question_len_placeholder
        )

        return q_vec
```
通过GRU进行训练，得到q_vec=last_state，last_state其实就是最后一个output的输出，即q_vec的shape变为[100,80]。即得到每个问题的向量表示，batch_size=100，每个向量为80维。
Input层：
--
input的占位符shape为[self.config.batch_size, self.max_input_len, self.max_sen_len]，即[100,10,6,80]。通过encoding层的reduce_sum得到10个句子的表示，shape变为[100,10,80]。然后通过双向的GRU得到fw_output和bw_output，shape分别为[100,10,80]，将两者相加得到input层的输出（即代码中的# f<-> = f-> + f<-），shape为[100,10,80]。
```
    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        forward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                forward_gru_cell,
                backward_gru_cell,
                inputs,
                dtype=np.float32,
                sequence_length=self.input_len_placeholder
        )

        # f<-> = f-> + f<-
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)

        # apply dropout
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)

        return fact_vecs
```
Episodic memory层：
--

 - 这是Memory Network的核心模块，由一个Attention Mechannism和一个Memory Update
   Mechanism来组成。其中使用RNN来作Memory update。 
 - 该模块的输入分别是Input module的输出和Question Module的输出。 
 - 使用$m^{i}$来表示第i次迭代的Memory，使用q来代表Question Module的输出，$m_{0}=q$，使用c来代表Input Module的输出，$c_{t}$为第t个位置的向量。

Attention Mechannism:
输入：

 - $c_{t}$：第t个候选事实（candidate fact，即input的第t个单词） 
 - $m^{i-1}$：上一轮迭代的记忆
 - q：问题向量

输出：
  
 - $g_{t}^{i}$：第i次迭代的第t个位置的候选事实的权重（$g_{t}^{i}=G(c_{t},m^{i-1},q) $）
   
具体公式：

 - 对G建模，需要使用上述输入构建一个特征集 z(c,m,q)。
 -  G是一个两层的前向神经网络

![这里写图片描述](http://img.blog.csdn.net/20170927204931338)
 
构建特征集的方式有很多，代码中采用了上图公式的子集，即只有c和q的点乘、c和m的点乘、c和q的差的绝对值、c和m的差的绝对值四项，并在axis=1的维度上进行concat操作。
输入$c_{t}$：第t个候选事实，shape为[100,80]（因为是第t个句子）
q：shape为[100,80]
则concat操作过后z的shape为[100,320]。然后将特征集通过一个两层的全连接层，得到一个[100,1]的shape，即为该句子经过attention层后的权重，batch为100。

Memory Update Mechanism：
输入：

 - $c_{1}，c_{2}...c_{T_{c}}$： 事实序列
 - $g_{1}^{i}，g_{2}^{i}...g_{T_{c}}^{i}$：事实得分序列

输出：
  
 - $m^{T_{m}}$：最后一次迭代的记忆序列
   
具体公式：

 - 修改的GRU公式。Episodic Memory Module需要一个停止迭代的信号。如果是有监督的attention，我们可以加入一个特殊的end-of-passes的信号到inputs中，如果gate选中了该特殊信号，则停止迭代。对于没有监督的数据集，可以设一个迭代的最大值。
 ![这里写图片描述](http://img.blog.csdn.net/20170927212058393)
 - 每一次迭代得到最后一个output=$e^{i}=h_{T_{C}}^{i}$，然后使用公式$m^{i}=GRU(e^{i},m^{i-1})$来更新$m^{i}$（此处代码为了方便，作者使用了dense的全连接层代替了GRU来更新记忆单元，但是效果依然很好）

代码中选用了memory层数为3，每一层需要计算事实得分序列$g_{1}^{i}，g_{2}^{i}...g_{T_{c}}^{i}$，即第一个Attention Mechannism模块:
```
    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False
        
        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.config.hidden_size),
                    gru_inputs,
                    dtype=np.float32,
                    sequence_length=self.input_len_placeholder
            )

        return episode
```

其中，构建特征集和两层全连接层的代码如下：
```
    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):

            features = [fact_vec*q_vec,
                        fact_vec*prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            attention = tf.contrib.layers.fully_connected(feature_vec,
                            self.config.embed_size,
                            activation_fn=tf.nn.tanh,
                            reuse=reuse, scope="fc1")
        
            attention = tf.contrib.layers.fully_connected(attention,
                            1,
                            activation_fn=None,
                            reuse=reuse, scope="fc2")
            
        return attention
```
第二个模块Memory Update Mechanism的代码在inference函数的Memory的name_scope中表示，并把GRU替换为了全连接层来更新m状态：
```
 with tf.variable_scope("hop_%d" % i):
     prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
             self.config.hidden_size,
             activation=tf.nn.relu)
```
模型训练及训练结果：
==
因为都是一些常见的代码，这里就不在重复。不过，最近对写tensorflow的代码有了一些新的思考，记下来备忘。

 - 首先，应该找一个好的框架，按照data_helper.py、model.py、training.py对代码进行重构。
 - 其次，在该框架中关于可视化的部分、训练的部分、参数部分、batch部分这些边边角角都应该照顾到，因为都是一些常见的代码，仿真新的paper的时候，就可以吧注意力放在处理数据集和模型构建上，不会因为一些琐碎的事情耽误时间，而把所有的精力都放在让模型跑通的主要目标上。
 - 对于自然语言处理领域，常见的问题有分类系统、问答系统、知识图谱、信息抽取等，这些问题对应的数据集也就只有常见的几种类型。如果仿真paper的时候，如果能找到之前处理相似数据的代码或者在网上找到处理相同数据的代码，那么就可以把主要精力放在模型构建上。毕竟模型构建才是初学者在深度学习中的主要任务。

参考文献
==
1、[Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://www.zybuluo.com/ShawnNg/note/579387)
2、[原文短篇](http://www.thespermwhale.com/jaseweston/ram/papers/paper_21.pdf)        [原文长篇](https://arxiv.org/pdf/1506.07285.pdf)
3、[知乎文本分类比赛的文本分类模型总结](https://github.com/brightmart/text_classification)

