---
layout: post
title: 基于Char-RNN Language Model进行文本生成（Tensorflow生成唐诗）
category: NLP
keywords: NLP
---
上一篇文章利用CharRNN进行语言模型的训练，语言模型的本意就是为了判断一个句子的概率。在文本生成领域就可以根据当前词预测下一个词，因此大有用途。比如在各种科技网站上随处可见的生成唐诗，歌词，小说，以及代码，为了加深我们对RNN的实现熟练程度，这里再推荐阅读两篇源码[中文古诗自动作诗机器人](https://github.com/jinfagang/tensorflow_poems)和[生成英文、写诗、歌词、小说、生成代码、生成日文](https://github.com/hzy46/Char-RNN-TensorFlow/tree/28c8c67694df328a573ad4210a78c71ca5cade01)两个。个人觉得第二个代码封装性更好，并且训练效果更好。但是第一个github上星星较多，因此首先看的就是第一个，所以这里就分析第一篇代码。（另外，该代码基于python3版本，并且有少量错误，因此我将修改python2.7版本并且可以训练的代码[放置github](https://github.com/Irvinglove/tensorflow_poems/tree/master)上）
比着上一篇文章，本篇的代码具有如下优点：

 - 使用dynamic_rnn函数构造rnn_model，更方便快捷
 - 具有生成功能，因此可以看到训练的模型究竟有什么用
 - 断点处继续训练的功能，减少不必要的训练时间

数据处理
--
该代码同上一篇一样，使用collections.Counter、zip、map等几个函数，就将文本工作处理完。根据下面的process_poems函数总结一下使用这几个函数处理啊文本的步骤。

 - 通过文档构建列表，如poems[i]就代表文档中第i行诗。当然如果只做语言模型的话，就不用构建该列表，直接取定长的文本就行。
 - 使用counter = collections.Counter()函数对所有字符进行计数。
 - 通过count_pairs = sorted(counter.items(), key=lambda x: -x[1])对计数结果进行排序，返回的结果是一个tuple
 - 通过words, _ = zip(*count_pairs)对tuple进行解压，得到words列表代表所有字符。该字符的行号即为该字符索引。
 - 通过word_int_map = dict(zip(words, range(len(words))))，得到字符与行号对应的索引字典。
 - 通过poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]函数将字符数据集映射成为索引数据集。

```
def process_poems(file_name):
    # 诗集
    poems = []
    with codecs.open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))

    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    # 这里根据包含了每个字对应的频率
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)

    # 取前多少个常用字
    words = words[:len(words)] + (' ',)
    # 每个字映射为一个数字ID
    word_int_map = dict(zip(words, range(len(words))))
    # poems_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in poems]
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]

    return poems_vector, word_int_map, words
```

模型构建
--
上篇文章使用legacy_seq2seq.rnn_decoder，需要对文本进行squeeze和split的操作，感觉很麻烦，我画示意图都用了很久。但在实际创建RNN时，使用dynamic_rnn更多一些。所以总结一下构建RNN的步骤。以下代码都基于tensorflow1.2版本。
#1. RNNcell#
RNNcell是tensorflow中实现RNN的基本单元，是一个抽象类，在实际应用中多用RNNcell的实现子类BasicRNNCell或者BasicLSTMCell。这几个类的实现都在[github](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py)，建议大家看一下源码，只需要看完注释部分，就可以大致与RNN构建过程相结合，便于理解。每一个实现子类都包含一个call方法，该方法实现了cell的调用。具体使用方式是：(output, next_state) = call(input, state)。借助图片来说可能更容易理解。假设我们有一个初始状态h0，还有输入x1，调用call(x1, h0)后就可以得到(output1, h1)：
![这里写图片描述](http://img.blog.csdn.net/20170804101945914)
再调用一次call(x2, h1)就可以得到(output2, h2)：
![这里写图片描述](http://img.blog.csdn.net/20170804101919916)
也就是说，每调用一次RNNCell的call方法，就相当于在时间上“推进了一步”，这就是RNNCell的基本功能。

除了call方法外，对于RNNCell，还有两个类属性比较重要：

 - state_size
 - output_size

前者是隐层的大小，后者是输出的大小。比如我们通常是将一个batch送入模型计算，设输入数据的形状为(batch_size, input_size)，那么计算时得到的隐层状态就是(batch_size, state_size)，输出就是(batch_size, output_size)。

可以用下面的代码验证一下（注意，以下代码都基于TensorFlow最新的1.2版本）：
```
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
print(cell.state_size) # 128
# 32 是 batch_size
inputs = tf.placeholder(np.float32, shape=(32, 100)) 
# 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
h0 = cell.zero_state(32, np.float32) 
#调用call函数
output, h1 = cell.call(inputs, h0) 

print(h1.shape) # (32, 128)
```
对于BasicLSTMCell，情况有些许不同，因为LSTM可以看做有两个隐状态h和c，对应的隐层就是一个Tuple，每个都是(batch_size, state_size)的形状：
```
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
# 通过zero_state得到一个全0的初始状态
h0 = lstm_cell.zero_state(32, np.float32) 
output, h1 = lstm_cell.call(inputs, h0)

print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)
```
#2. dynamic_rnn：一次执行多步时间序列#

基础的RNNCell有一个很明显的问题：对于单个的RNNCell，我们使用它的call函数进行运算时，只是在序列时间上前进了一步。比如使用x1、h0得到h1，通过x2、h1得到h2等。这样的h话，如果我们的序列长度为10，就要调用10次call函数，比较麻烦。对此，TensorFlow提供了一个tf.nn.dynamic_rnn函数，使用该函数就相当于调用了n次call函数。即通过{h0,x1, x2, …., xn}直接得{h1,h2…,hn}。

具体来说，如果我们的time_major=False的话，设我们输入数据的格式为(batch_size, time_steps, input_size)，其中time_steps表示序列本身的长度，如在Char RNN中，长度为10的句子对应的time_steps就等于10。最后的input_size就表示输入数据单个序列单个时间维度上固有的长度。outputs的格式为(batch_size, time_steps, state_size)。如果time_major=True的话，(time_steps,batch_size , input_size)，输出也发生相应的变化。另外我们已经定义好了一个RNNCell，调用该RNNCell的call函数time_steps次，对应的代码就是：
```
# inputs: shape = (batch_size, time_steps, input_size) 
# cell: RNNCell
# initial_state: shape = (batch_size, cell.state_size)。初始状态。一般可以取零矩阵
outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
```
#3. MultiRNNCell：堆叠RNN#

很多时候，单层RNN的能力有限，我们需要多层的RNN。将x输入第一层RNN的后得到隐层状态h，这个隐层状态就相当于第二层RNN的输入，第二层RNN的隐层状态又相当于第三层RNN的输入，以此类推。在TensorFlow中，可以使用tf.nn.rnn_cell.MultiRNNCell函数对RNNCell进行堆叠，相应的示例程序如下：
```
import tensorflow as tf
import numpy as np

# 每调用一次这个函数就返回一个BasicRNNCell
def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)]) # 3层RNN
# 得到的cell实际也是RNNCell的子类
# 它的state_size是(128, 128, 128)
# (128, 128, 128)并不是128x128x128的意思
# 而是表示共有3个隐层状态，每个隐层状态的大小为128
print(cell.state_size) # (128, 128, 128)
# 使用对应的call函数
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
output, h1 = cell.call(inputs, h0)
print(h1) # tuple中含有3个32x128的向量
```
#4.值得注意#
![这里写图片描述](http://img.blog.csdn.net/20170804110220776)
之前我们讲过，如果dynamic_rnn中time_major=False的话，设我们输出数据的格式为(batch_size, time_steps, state_size)。将上图与TensorFlow的BasicRNNCell对照来看。h就对应了BasicRNNCell的state_size。那么，y是不是就对应了BasicRNNCell的output_size呢？答案是否定的。下面是BasicRNNCell的call函数：
```
def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    output = self._activation(_linear([inputs, state], self._num_units, True))
    return output, output
```
这句“return output, output”说明在BasicRNNCell中，output其实和隐状态的值是一样的。因此，我们还需要额外对输出定义新的变换，才能得到图中真正的输出y。这是因为，输出层只有堆叠rnn的最上方一层才有，其它层不需要，所以只需要在最后对输出定义新的变换即可。另外，由于output和隐状态是一回事，所以在BasicRNNCell中，state_size永远等于output_size。
通过梳理dynamic_rnn的过程，那下面的代码也不难理解了。
```
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
  
    end_points = {}
    # 构建RNN基本单元RNNcell
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    else:
        cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    # 构建堆叠rnn，这里选用两层的rnn
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    # 如果是训练模式，output_data不为None，则初始状态shape为[batch_size * rnn_size]
    # 如果是生成模式，output_data为None，则初始状态shape为[1 * rnn_size]
    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)
    
    # 构建隐层
    with tf.device("/cpu:0"):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    # [batch_size, ?, rnn_size] = [64, ?, 128]
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    output = tf.reshape(outputs, [-1, rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
    # [?, vocab_size+1]

    if output_data is not None:
        # output_data must be one-hot encode
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        # should be [?, vocab_size+1]

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # loss shape should be [?, vocab_size+1]
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points

```
模型训练
--
模型训练部分大部分都不是陌生的代码，该程序将参数保存在checkpoint中，如果中断训练，下次可以直接从中断部分继续。
```
def run_training():
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)
    # 处理数据集
    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    # 生成batch
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    
    # 构建模型
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        # 从上次中断的checkpoint开始训练
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
                if epoch % 6 == 0:
                    saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            # 如果Ctrl+c中断，保存checkpoint，
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))

```
文本生成
--
该部分代码主要包含gen_poem和to_word函数。就是将训练结束后的last_state作为initial_state传入rnn，那么新生成的output即为预测的结果。详细注释在代码中都有，下面我们开一下以“白”开头的诗吧：
```
白首长江上，青山一去年。
不知天地去，不见白云生。
```
这是训练了大概两万多步的结果，读起来还可以哈！！！
```
def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    # 该代码是作者写的，t的长度为vocab_size + 1, 随机生成一个数然后判断能插入第几个位置来取字
    # 个人感觉这使得训练变得毫无意义
    # sample = int(np.searchsorted(t, np.random.rand(1) * s))
    # 而实际上输出的预测向量predict，随着训练过程应该逐渐向one-hot编码靠拢，所以应该取argmax函数
    sample = np.argmax(predict)
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def gen_poem(begin_word):
    batch_size = 1
    print('[INFO] loading corpus from %s' % FLAGS.file_path)
    poems_vector, word_int_map, vocabularies = process_poems(FLAGS.file_path)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        # 如果指定开始的字
        if begin_word:
            word = begin_word
        # 如果不指定开始的字，就按根据start_token生成第一个字
        else:
            word = to_word(predict, vocabularies)
        poem = ''
        while word != end_token:
            poem += word
            x = np.zeros((1, 1))
            # 比如，指定第一个字为“白”，则x就为x[[36]]，即batch_size为1，并且poems_length为1，生成下一个字
            x[0, 0] = word_int_map[word]
            # 传入input_data，此时没有output_data即为生成模式，并且传入初始状态为训练结束的状态
            # state_shape为[1,rnn_size]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            # 根据预测结果生成对应的字
            word = to_word(predict, vocabularies)
        return poem
```

参考文献：
--
1、[TensorFlow中RNN实现的正确打开方式
](https://zhuanlan.zhihu.com/p/28196873)
2、[《安娜卡列尼娜》文本生成——利用TensorFlow构建LSTM模型](https://zhuanlan.zhihu.com/p/27087310)
3、[解读tensorflow之rnn](http://blog.csdn.net/mydear_11000/article/details/52414342)
4、[基于循环神经网络实现基于字符的语言模型（char-level RNN Language Model）-tensorflow实现](http://blog.csdn.net/irving_zhang/article/details/76038710)