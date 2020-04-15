Notes: chinese version at the bottom. ^_^
# iching v0.0.1
Iching is a deep meta reinforcement learning quantitative trading platform.
## Why iching
Iching project is inspired by I ching or The Change of Book - a ancient chinese book to predict the future. It is the first book to use binary sysntem in the world about 5000 years ago. It propose that Tai Chi generate Yin-Yang. Yin-Yang generate Eight Diagrams. Two Eight Diagrams have 64 states. There has a Change Diagrams which has 6 states. In all there has 384 state. Ancient chinese people use this 384 states to predict everything faithfully.
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

易经量化交易系统是基于深度元强化学习技术，并结合心理学小样本学习技术的量化交易平台。\newline
