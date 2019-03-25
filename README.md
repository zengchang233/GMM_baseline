# GMM_baseline
[未来杯语音赛道说话人识别](https://ai.futurelab.tv/contest_detail/2)的baseline，用的是传统的UBM-GMM模型。

## 更新

-----------------------

**更新1**

~~训练集里面有的语音可能只有`wav`文件头，这样的语音提取特征保存的文件是一个空文件，无法读取。现在我已经在`utils.py`文件里面添加了`remove`函数，并且在`train ubm`之前调用`remove`来删除这些空文件。~~

`aishell2`数据集有一些文件只有`wav`文件头，大小大概为`44k`，请先删除这些文件以免训练出错。

**注意，千万不要用完整的`aishell2`来训练这个模型**，太耗时间了，推荐使用刚开始的那个100说话人的`aishell2`的子集来训练，大概需要一天的时间训练`ubm`。

-----------------------------------

**更新2**

添加了新的特征`plp`，在说话人识别里面非常常用。

---------------------------------

**更新3**

把特征提取部分和训练模型部分分成了两个不同的文件，特征提取在`preprocess.py`里面，模型训练部分还是在`gmm_ubm.py`文件里面。

另外写了一个`shell`脚本，只需要修改里面的一些文件路径和训练参数就可以傻瓜式训练模型了，需要修改的地方我都已经注明了。

修改完成之后`nohup bash run.sh &`就可以在后台训练打分了。

------------------------

## 依赖项

### 软件
`python`版本推荐`3.7.2`

相关的库`pip install -r requirements.txt`

```
sidekit==1.2.2
numpy
pandas
matplotlib
scipy
tqdm
```

### 注意
如果`sidekit`安装了之后无法`import`，需要找到`sidekit`安装的地方，改一下`__init__.py`文件，这个文件里面的38到42行，如果你只训练`gmm-ubm`的话，所有的选项都设置为`False`。

如果你想使用`svm`，请先安装`libsvm`库，然后把`__init__.py`里面的相应选项设置为`True`，最后在`sidekit`的`libsvm`文件夹里面新建一个链接指向`svm`库的`libsvm.so.2`。

最新版的`sidekit`，神经网络部分已经换成了`Pytorch`的`backend`，但是`1.2.2`还是`Theano`，如果你想在`sidekit`里面使用神经网络，请安装最新版。

Ps：最新版的`sidekit`，也就是`1.3.1`我没有测试过，可能有`bug`，因为我使用`1.2.9`版本的时候发现过`bug`，后来回退到了稳定的`1.2.2`版本。

另外，`sidekit`包还提供了生成`DET Curve`的方法，但是可能会失败，具体原因我也没有找到。总之如果大家经过上面的修改之后还是不能导入`sidekit`的话，请修改`bosaris`文件夹里面的`detplot.py`的第39行，`matplotlib.use('PDF')`, 可以改成`matplotlib.use('Qt5Agg')`。

## 主要脚本

`preprocess.py`文件主要是特征提取已经路径的读取，具体用法看`run.sh`脚本。

`gmm_ubm.py`这个脚本包含了训练ubm，自适应得到注册人的gmm，以及计算注册的gmm对所有攻击语音的打分的函数。脚本的用法如下。
测试了512 mixture的GMM分别对于64-dim的fbank和13+delta+double delta的mfcc的拟合，发现mfcc的效果相对较好，但是根据比赛的评分规则，最好也只有0.74。
```
usage: gmm_ubm.py [-h] [--feat_type {mfcc,fb,plp}] [--delta]
                  [--distribNum DISTRIBNUM] [--num_thread NUM_THREAD]
                  [--extract] [--train] [--adaptation] [--score]
                  name

script for GMM-UBM adaptation

positional arguments:
  name                  model name

optional arguments:
  -h, --help            show this help message and exit
  --feat_type {mfcc,fb,plp}
                        feature type (default : "plp")
  --delta               using delta information of feature
  --distribNum DISTRIBNUM
                        distribution number (default : 512)
  --num_thread NUM_THREAD
                        threads number (default : 20)
  --train               train the ubm model
  --adaptation          adaptation for speaker model
  --score               compute the eer
```

`utils.py`这个脚本包含了一些文件的预处理例如idmap，ndx和key，关于这些文件的详细信息，请参考sidekit的官方文档[sidekit](https://projets-lium.univ-lemans.fr/sidekit/_downloads/sidekit.pdf)。另外还包含了一个根据比赛的积分规则打分的函数，`--score`选项用来打分并将打分的结果保存在`result`文件夹里面。
```
python utils.py -h
usage: utils.py [-h] [--enroll ENROLL] [--dev DEV] [--test TEST]
                [--score SCORE]

some utils for preprocess and score

optional arguments:
  -h, --help       show this help message and exit
  --enroll ENROLL  enrollment csv path
  --dev DEV        annotation csv path
  --test TEST      test csv path
  --score SCORE    calculate the score
```

## 说明
这个项目是打算作为本次比赛的baseline。训练的语音是比赛官方提供的aishell2的子集，包含了100人的语音，开发集也是官方提供的数据。
本次比赛需要验证不同信道的说话人的语音，UBM-GMM对于这种条件适应性较差，当然如果用ZT-norm重新整理得分的话效果应该更好，但是总的来说注册语音的信道和测试信道不匹配的问题还是很严重。
这种条件下，我相信i-vector+PLDA是一个更好的选择来解决这个问题。

## 感谢
如果这个repository对你有帮助麻烦请star:-D。

## 联系
Email : <zengchang.elec@gmail.com>
