# GMM_baseline
[未来杯语音赛道说话人识别](https://ai.futurelab.tv/contest_detail/2)的baseline，用的是传统的UBM-GMM模型。

## 依赖项
```
python==3.7.2
sidekit==1.2.2
numpy
pandas
matplotlib
scipy
tqdm
```

### 注意
如果`sidekit`安装了之后无法`import`，需要找到`sidekit`安装的地方，改一下`__init__.py`文件，这个文件里面的38到42行，如果你只训练`gmm-ubm`的话，所有的选项都设置为`False`。

如果你想使用`svm`，请先安装`libsvm`库，然后把`__init__.py`里面的相应选项设置为`True`。

最新版的`sidekit`，神经网络部分已经换成了`Pytorch`的`backend`，但是`1.2.2`还是`Theano`，如果你想在`sidekit`里面使用神经网络，请安装最新版。

Ps：最新版的`sidekit`，也就是`1.3.1`我没有测试过，可能有`bug`，因为我使用`1.2.9`版本的时候发现过`bug`，后来回退到了稳定的`1.2.2`版本。

## 主要脚本
`gmm_ubm.py`这个脚本包含了训练ubm，自适应得到注册人的gmm，以及计算注册的gmm对所有攻击语音的打分的函数。脚本的用法如下。
测试了512 mixture的GMM分别对于64-dim的fbank和13+delta+double delta的mfcc的拟合，发现mfcc的效果相对较好，但是根据比赛的评分规则，最好也只有0.74。
```
python gmm_ubm.py -h
usage: gmm_ubm.py [-h] [--feat_type FEAT_TYPE] [--delta]
                  [--distribNum DISTRIBNUM] [--num_thread NUM_THREAD]
                  [--extract] [--train] [--adaptation] [--score]
                  name

script for GMM-UBM adaptation

positional arguments:
  name                  model name

optional arguments:
  -h, --help            show this help message and exit
  --feat_type FEAT_TYPE
                        feature type
  --delta               using delta information of feature
  --distribNum DISTRIBNUM
                        distribution number
  --num_thread NUM_THREAD
                        threads number
  --extract             extract feature from audio
  --train               train the ubm model
  --adaptation          adaptation for speaker model
  --score               compute the score
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
