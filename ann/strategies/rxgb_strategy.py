#
import os
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance

class RxgbStrategy(object):
    def __init__(self):
        self.name = 'apps.rxgb.RxgbStrategy'
        self.model_file = './work/xgb.ckp'
        self.bst = None
        self.xgb_params = {
            'learning_rate': 0.1,  # 步长
            'n_estimators': 10,
            'max_depth': 5,  # 树的最大深度
            'objective': 'multi:softprob',
            'num_class': 3,
            # 决定最小叶子节点样本权重和，如果一个叶子节点的样本权重和小于
            # min_child_weight则拆分过程结束。
            'min_child_weight': 1, 
            # 指定了节点分裂所需的最小损失函数下降值。
            # 这个参数的值越大，算法越保守 
            'gamma': 0,  
            'silent': 0,  # 输出运行信息
            # 每个决策树所用的子样本占总样本的比例（作用于样本）
            'subsample': 0.8,  
            # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
            'colsample_bytree': 0.8,  
            'nthread': 4,
            'seed': 27
        }

    def train(self, X_train, y_train, X_test, y_test):
        '''
        # 当需要控制样本权重时，可用于深度强化学习中
        w = np.random.rand(5,1)
        dtrain = xgb.DMatrix( data, label=label, missing = -999.0, weight=w)
        '''
        # 在学习过程中，见到新样本之后，会生成一个结果，我们用新净值除以老净值的比例作为reward，并将
        # 作为rlw中对应样本的权重。
        rlw = np.ones((X_train.shape[0]))
        print('X_train:{0}'.format(X_train.shape))
        print('y_train:{0}'.format(y_train.shape))
        xg_train = xgb.DMatrix(X_train, label=y_train, weight=rlw)
        xg_test = xgb.DMatrix( X_test, label=y_test)
        watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
        num_round = 300
        print('build xgboost model...')
        bst = xgb.train(self.xgb_params, xg_train, num_round, watchlist )
        # 继续训练是加一个参数：
        # bst = xgb.train(xgb_params, xg_train, num_round, watchlist, xgb_model=self.bst)
        bst.save_model(self.model_file)
        pred = bst.predict( xg_test )
        preds = np.argmax(pred, axis=1)
        delta = preds - y_test
        delta[delta != 0] = 1
        num = np.sum(delta)
        print('preds:\r\n{0};\r\n{1}<=label;'.format(preds, y_test))
        print('精度：{0}; y_test:{1};'.format(1 - num / delta.shape[0], y_test.shape))
        plot_importance(bst)
        plt.show()

    def predict(self, X):
        if self.bst is None:
            if not os.path.exists(self.model_file):
                print('载入模型失败：没有模型文件')
                os.exit(1)
            self.bst = xgb.Booster({})
            self.bst.load_model(self.model_file)
        xg = xgb.DMatrix( X, label=None)
        preds = np.argmax(self.bst.predict(xg), axis=1)
        return preds[0]

    def train_drl(self, X, y, rlw):
        xg_train = xgb.DMatrix(X, label=y, weight=rlw)
        watchlist = [ (xg_train,'train')]
        num_round = 10
        self.bst = xgb.train(self.xgb_params, xg_train, num_round, watchlist, xgb_model=self.bst)
        self.bst.save_model(self.model_file)
        pred = self.bst.predict( xg_train )
        preds = np.argmax(pred, axis=1)
        delta = preds - y
        delta[delta != 0] = 1
        num = np.sum(delta)
        print('精度：{0};'.format(1 - num / delta.shape[0]))

    
    def create_np_dataset(self, dataset_size = 60000):
        a1 = np.random.randn(dataset_size)
        c1 = 1.1
        a2 = np.random.randn(dataset_size)
        c2 = 2.1
        a3 = np.random.randn(dataset_size)
        c3 = 3.2
        a4 = np.random.randn(dataset_size)
        c4 = 4.3
        a5 = np.random.randn(dataset_size)
        c5 = 5.1
        delta = np.random.randn(dataset_size)
        a6_ = a1*c1 + a2*c2 + a3*c3 + a4*c4 + a5*c5
        a6 = a6_ + delta
        y_ = np.zeros((dataset_size), dtype=np.int32)
        for i in range(len(a6)):
            if a6[i] - a6_[i] > 0.3:
                y_[i] = 0
            elif a6[i] - a6_[i] < -0.3:
                y_[i] = 2
            else:
                y_[i] = 1
        X = np.reshape(np.array([a1, a2, a3, a4, a5, a6]).T, (dataset_size, 6))
        y = y_.T
        return X, y