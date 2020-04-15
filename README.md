Notes: chinese version at the bottom. ^_^
# iching v0.0.1
Iching is a deep meta reinforcement learning quantitative trading platform.
## Why iching
Iching project is inspired by I ching or The Change of Book - an ancient chinese book to predict the future. It is the first book to use binary sysntem in the world about 5000 years ago. It propose that Tai Chi generate Yin-Yang. Yin-Yang generate Eight Diagrams. Two Eight Diagrams have 64 states. There has a Change Diagrams which has 6 states. In all there has 384 state. Ancient chinese people use this 384 states to predict everything faithfully.
Iching aim to provide a reliable means to deal with financial market. It uses the most advanced AI technology. At the same time it uses the traditional chinese wisdom equally.
## Technical background
We all known that the application of deep learning and machine learing in financial market is not very successful now. Because almost all the algorithms require the test dataset and the training dataset have identical independent distribution. But financial market changes constantly. The macro economy, micro economy, social media, goverment policy all influence the financial market in some way. With the progress of technology and society financial market envolves all the time. Most of the algorithm assume the distribution of the problem to research is normal distribution. Unfortunately most quotation of the financial market is anything but normal distribution. So apply deep learning and machine learning directly to financial market can not have good results.
Iching uses reinforcement learning algorithm to deal with the constant changing characteristic of the financial market. It uses meta learning to accelerate the learning process of reinforcement learning. At the same time it uses psychological few shot learning to deal with the lack of relevent qutotation data. It uses multimodal learning to process the social media and financial news data.
## Installation
Because this project is on the very early stage you can't install it with pip. you should git clone this repository first:
```
git clone https://github.com/yt7589/iching.git
```
You have to email yt7589#qq.com to get the dataset and check point files. After you receive the zip file you should unzip it then copy the data, work, logs folder to iching folder.
Before you start please install python3.7 and pytorch 1.4 first.
Run this program is very easy. Go to the iching folder and run:
```
python app_main.py
```
### Project Structure
#### ann/ds
This folder contains all the datasets.
#### ann/envs
This folder contains all the financial market reinforcement learning environments.
#### ann/fme
This folder contains the base class of the financial market environment. It is a simplified reinforcement learning environment.
#### ann/strategies
This folder contains the strategies used by reinforcement learning algorithm.
#### apps/asdk
This folder contains reinforcement learning application deal with chinese A stock market daily k line dataset. Each sub folder use a different strategy.
#### apps/asml
This folder contains a demo application to use MAML algorithm in chinese A stock daily k line dataset.
#### apps/common
This folder contains common logic for financial market trading, such as commission, tax, transfer fee.
#### apps/ogml
This folder is a demo application to use MAML to omniglot dataset.
#### apps/tp
This folder is a demo application to use trading pair algorithm in chinese A stock daily K line dataset.

易经量化交易系统是基于深度元强化学习技术，并结合心理学小样本学习技术的量化交易平台。  
## 易经量化
易经量化交易平台是受易经启动的量化交易系统，易经是中国古代用于预测的古老的技术。可以说易经是5000多年前第一个使用二进制表示的系统。易经认为太极生阴阳，阴阳生八卦，八卦用三个状态表示，每个状态可以为阴或阳，总共有$2^3$，即8个状态。每次进行预测由上卦和下卦，两个八卦组成，同时有一个六个状态的变卦，根据该值修改上卦或下卦中的某个状态。这样就构成了384种状态，每种状态可以对未来进行预测。易经量化交易系统，试图将易经精神引入金融市场量化交易领域，采用最先进的深度元强化学习技术和心理学小样本学习技术，同时借鉴易经的古老智慧，打造出完美的金融量化交易系统。
## 技术背景
众所周知，采用深度学习和机器学习技术，直接应用于金融市场，目前效果还不是很理想。因为几乎所有的算法均要求训练数据集和测试数据集具有独立、同分布特性，即I.I.D特性，但是金融市场的特点是处于实时变化当中，宏观经济、微观经济、社交媒体、政府政策都会以某种特别的方式影响金融市场行情，并且随着技术和社会的进步，金融市场也在不断进化。同时，多数算法都要求所研究的问题符合正态分布，而金融市场的行情数据，分钟线、日K线等，根本就不符合正态分布。所以直接将深度学习和机器学习算法应用到金融市场，是很难取得理想效果的。易经量化交易平台采用深度强化学习技术来处理不断变化的金融市场特性，同时针对深度强化学习需要大量训练数据，训练速度过慢的问题，采用快速元学习来加快训练过程，并且加入了心理学小样本学习技术，以及利用优秀交易员的交易历史，采用模仿学习技术，同时系统不仅对市场行情数据进行学习，同时将财经新闻、社交媒体、财报、宏观经济数据等通过NLP技术，形成向量化输入源，通过多模态学习，进一步提高模型的泛化能力。
## 安装
本项目需要python3.7和PyTorch 1.4，请先通过pip install安装相关依赖。  
因为项目还处于非常早期的阶段，因此还不能通过pip install来安装，首先需要先克隆本项目：
```bash
git clone https://github.com/yt7589/iching.git
```
然后你需要向yt7589#qq.com发送邮件，请求数据集和工作目录中的check point文件。将获取到的zip文件进行解压，将其中的data、logs、work目录拷贝到iching目录下，然后就可以在iching目录下，通过运行如下命令运行程序：
```bash
python app_main.py
```
## 项目目录结构
ann/ds：项目支持的所有数据集；  
ann/envs：包含所有支持的金融市场强化学习环境；  
ann/fme：金融市场强化学习框架；  
ann/strategies：各种强化学习策略算法；  
apps/asdk：应用于A股市场日K线数据集的应用，每个子目录是一种不同的策略；  
apps/asml：A股日K线数据集上的MAML算法应用演示应用；  
apps/common：金融市场公共方法，如手续费、税费、过户费等规则定义；  
apps/ogml：MAML模型在omniglot数据集上的应用示例；  
apps/tp：在A股日K线数据集上的交易对应用示例；  