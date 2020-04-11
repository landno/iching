#
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader #, Dataset
from collections import OrderedDict
from apps.asml.asdk_meta_ds import AsdkMetaDs
from apps.asml.asdk_product_ds import AsdkProductDs
from apps.asml.asml_model import AsmlModel
from apps.common.tp_util import TpUtil

class AsmlApp(object):
    def __init__(self):
        self.name = 'apps.asml.AsmlApp'
        self.chpt_file = './work/asml.pkl'
        self.chpt_product_file = './work/asml_product.pkl'
        self.train_data_path = './data/tp/'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def startup(self):
        print('MAML for stock market v0.0.7')
        #self.train_meta()
        #self.evaluate_on_test_ds()
        #self.predict_example()
        #tps = TpUtil.choose_trading_pairs('601006', '2007-01-01', '2007-11-30')
        #print('trading pairs: {0}, {1}'.format(tps[0], tps[1]))
        #print('^_^')
        #self.train_product()
        self.predict_product_example()

    def train_product(self):
        ''' train the model trained by MAML '''
        k_shot = 1
        q_query = 1
        n_way = 3
        meta_lr = 0.005
        max_epoch = 3
        stock_code = '601006'
        start_date = '2007-12-01'
        end_date = '2007-12-31'
        n_way = 3
        batch_size = 4
        eval_batches = 1
        train_loader, train_iter, val_loader, val_iter = \
                        self.load_dataset_product(stock_code, start_date, \
                        end_date, 1)
        model = AsmlModel(1, n_way).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        for epoch in range(max_epoch):
            print("Epoch %d" %(epoch))
            train_loss = []
            train_acc = []
            for step in tqdm(range(len(train_loader) // (batch_size))): # 這裡的 step 是一次 meta-gradinet update step
                x, y, train_iter = self.get_product_batch(
                                batch_size, k_shot, q_query, 
                                train_loader, train_iter)
                loss, acc = self.train_product_batch(model, optimizer, x, y, n_way, k_shot, q_query, loss_fn)
                train_loss.append(loss.item())
                train_acc.append(acc)
            print("  Loss    : ", np.mean(train_loss))
            print("  Accuracy: ", np.mean(train_acc))
            # 每個 epoch 結束後，看看 validation accuracy 如何  
            # 助教並沒有做 early stopping，同學如果覺得有需要是可以做的 
            val_acc = []
            print('##### len={0}; l={1}; e={2};'.format(
                len(val_loader) // (eval_batches),
                len(val_loader), (eval_batches)
            ))
            for eval_step in tqdm(range(len(val_loader) // (eval_batches))):
                x, y, val_iter = self.get_product_batch(eval_batches, k_shot, q_query, val_loader, val_iter)
                _, acc = self.train_product_batch(model, optimizer, x, y, n_way, k_shot, q_query, loss_fn, inner_train_steps = 3, train = False) # testing時，我們更新三次 inner-step
                val_acc.append(acc)
            print("  Validation accuracy: ", np.mean(val_acc))
        torch.save(model.state_dict(), self.chpt_product_file)
        print('^_^ v0.0.6')

    def predict_product_example(self):
        test_data_path = './data/tp/'
        stock_code = '601006'
        start_date = '2008-01-01'
        end_date = '2008-01-31'
        n_way = 3
        k_shot = 1
        q_query = 1
        ds = AsdkProductDs(self.train_data_path, stock_code, start_date, end_date, 0, 0, 1)
        X_mu = ds.X_mu
        X_std = ds.X_std
        k_shot = 1
        q_query = 1
        raw_ds = np.array([
            [6.39, 6.39, 5.49, 5.52, 614436357, 0],
            [5.54, 5.82, 5.52, 5.73, 232650438, 0],
            [5.69, 5.84, 5.66, 5.71, 100498125, 0],
            [5.73, 5.79, 5.64, 5.66, 79173885, 0],
            [5.64, 5.96, 5.59, 5.88, 126376467, 0]
        ])
        X, _, X_raw = AsdkMetaDs.get_ds_by_raw_ds1(raw_ds, k_shot, q_query, X_mu, X_std)
        self.predict_product(X)

    def predict_product(self, X):
        n_way = 3
        lr = 0.005
        X = torch.from_numpy(X.reshape(1, 1, 25)).to(self.device)
        model = AsmlModel(1, n_way).to(self.device)
        model.load_state_dict(torch.load(self.chpt_product_file))
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        fast_weights = OrderedDict(model.named_parameters())
        logits = model.functional_forward(X, fast_weights)
        print(logits)
        labels = torch.argmax(logits, dim=1)
        print('predict label: {0};'.format(labels[0]))

    def train_product_batch(self, model, optimizer, x, y, n_way, k_shot, 
                q_query, loss_fn, inner_train_steps= 1, 
                inner_lr = 0.4, train = True):
        """
        Args:
        x is the input omniglot images for a meta_step, shape = 
                [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
        n_way: 每個分類的 task 要有幾個 class
        k_shot: 每個類別在 training 的時候會有多少張照片
        q_query: 在 testing 時，每個類別會用多少張照片 update
        """
        criterion = loss_fn
        for meta_batch, label in zip(x, y):
            train_set = meta_batch[:n_way*k_shot] # train_set 是我們拿來 update inner loop 參數的 data
            train_label = label[:n_way*k_shot]
            fast_weights = OrderedDict(model.named_parameters()) # 在 inner loop update 參數時，
            logits = model.functional_forward(train_set, fast_weights)
            loss = criterion(logits, train_label)
        model.train()
        optimizer.zero_grad()
        if train:
            loss.backward()
            optimizer.step()
        task_acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == train_label.cpu().numpy()]).mean()
        return loss, task_acc

    def get_product_batch(self, batch_size, k_shot, q_query, data_loader, iterator):
        data = []
        label = []
        for i in range(batch_size):
            try:
                task_data, task_label = iterator.next()  # 一筆 task_data 就是一個 task 裡面的 data，
                #大小是 [n_way, k_shot+q_query, 1, 28, 28]
            except StopIteration:
                iterator = iter(data_loader)
                task_data, task_label = iterator.next()
            train_data = task_data[:, :k_shot].reshape(-1, 1, 25)
            val_data = task_data[:, k_shot:].reshape(-1, 1, 25)
            task_data = torch.cat((train_data, val_data), 0)
            data.append(task_data)
            train_label = task_label[:, :k_shot]
            train_label = train_label.reshape((train_label.shape[0] * train_label.shape[1]))
            val_label = task_label[:, k_shot:]
            val_label = val_label.reshape((val_label.shape[0]*val_label.shape[1]))
            task_label = torch.cat((train_label, val_label), 0)
            label.append(task_label)
        return torch.stack(data).to(self.device), torch.stack(label).to(self.device), iterator






















    def train_meta(self):
        stock_codes = ['601006', '600015', '600585']
        start_date = '2007-01-01'
        end_date = '2007-11-30'
        meta_batch_size = 4
        k_shot = 2
        q_query = 1
        inner_train_steps = 1
        inner_lr = 0.4
        meta_lr = 0.005
        max_epoch = 3
        eval_batches = 1
        n_way = 3
        train_loaders = []
        train_iters = []
        val_loaders = []
        val_iters = []
        max_len = self.get_max_ds_len(stock_codes, start_date, end_date, k_shot, q_query, n_way)
        for stock_code in stock_codes:
            print('stock_code:{0};'.format(stock_code))
            train_loader, train_iter, val_loader, val_iter = \
                        self.load_dataset_meta(stock_code, start_date, \
                        end_date, max_len, k_shot, q_query, n_way)
            train_loaders.append(train_loader)
            train_iters.append(train_iter)
            val_loaders.append(val_loader)
            val_iters.append(val_iter)
        meta_model = AsmlModel(1, n_way).to(self.device)
        num_tasks = len(stock_codes)
        #print('载入已有模型......')
        #meta_model.load_state_dict(torch.load(self.chpt_file))
        optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        for epoch in range(max_epoch):
            print("Epoch %d" %(epoch))

            train_meta_loss = []
            train_acc = []
            for step in tqdm(range(len(train_loader) // (meta_batch_size))): # 這裡的 step 是一次 meta-gradinet update step
                xs = []
                ys = []
                for i in range(num_tasks):
                    x, y, train_iters[i] = self.get_meta_batch(
                                meta_batch_size, k_shot, q_query, 
                                train_loaders[i], train_iters[i])
                    xs.append(x)
                    ys.append(y)
                meta_loss, acc = self.train_meta_batch(meta_model, optimizer, xs, ys, n_way, k_shot, q_query, loss_fn)
                train_meta_loss.append(meta_loss.item())
                train_acc.append(acc)

            print("  Loss    : ", np.mean(train_meta_loss))
            print("  Accuracy: ", np.mean(train_acc))
            # 每個 epoch 結束後，看看 validation accuracy 如何  
            # 助教並沒有做 early stopping，同學如果覺得有需要是可以做的 
            val_acc = []
            print('##### len={0}; l={1}; e={2};'.format(
                len(val_loader) // (eval_batches),
                len(val_loader), (eval_batches)
            ))
            for eval_step in tqdm(range(len(val_loader) // (eval_batches))):
                txs = []
                tys = []
                for i in range(num_tasks):
                    x, y, val_iters[i] = self.get_meta_batch(eval_batches, k_shot, q_query, val_loaders[i], val_iters[i])
                    txs.append(x)
                    tys.append(y)
                _, acc = self.train_meta_batch(meta_model, optimizer, txs, tys, n_way, k_shot, q_query, loss_fn, inner_train_steps = 3, train = False) # testing時，我們更新三次 inner-step
                val_acc.append(acc)
            print("  Validation accuracy: ", np.mean(val_acc))
        torch.save(meta_model.state_dict(), self.chpt_file)
        print('^_^')

    def evaluate_on_test_ds(self):
        test_data_path = './data/tp/'
        stock_code = '601006'
        start_date = '2006-08-01'
        end_date = '2007-04-25'
        n_way = 3
        test_data_path = './data/Omniglot/images_evaluation/'
        k_shot = 1
        q_query = 1
        test_batches = 2
        meta_lr = 0.005
        test_loader = DataLoader(AsdkMetaDs(test_data_path, stock_code, start_date, end_date, k_shot, q_query),
                                batch_size = n_way,
                                num_workers = 1,
                                shuffle = True,
                                drop_last = True)
        test_iter = iter(test_loader)
        test_acc = []

        meta_model = AsmlModel(1, n_way).to(self.device)
        meta_model.load_state_dict(torch.load(self.chpt_file))
        optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        for test_step in tqdm(range(len(test_loader) // (test_batches))):
            x, y, val_iter = self.get_meta_batch(test_batches, k_shot, q_query, test_loader, test_iter)
            _, acc = self.train_batch_meta(meta_model, optimizer, x, y, n_way, k_shot, q_query, loss_fn, inner_train_steps = 3, train = False) # testing時，我們更新三次 inner-step
            test_acc.append(acc)
        print("  Testing accuracy: ", np.mean(test_acc))

    def predict_example(self):
        test_data_path = './data/tp/'
        stock_code = '601006'
        start_date = '2006-08-01'
        end_date = '2007-04-25'
        n_way = 3
        k_shot = 1
        q_query = 1
        ds = AsdkMetaDs(test_data_path, stock_code, start_date, end_date, k_shot, q_query)
        X_mu = ds.X_mu
        X_std = ds.X_std
        k_shot = 1
        q_query = 1
        raw_ds = np.array([
            [6.39, 6.39, 5.49, 5.52, 614436357, 0],
            [5.54, 5.82, 5.52, 5.73, 232650438, 0],
            [5.69, 5.84, 5.66, 5.71, 100498125, 0],
            [5.73, 5.79, 5.64, 5.66, 79173885, 0],
            [5.64, 5.96, 5.59, 5.88, 126376467, 0]
        ])
        X, _, X_raw = AsdkMetaDs.get_ds_by_raw_ds1(raw_ds, k_shot, q_query, X_mu, X_std)
        self.predict(X)

    def predict(self, X):
        test_data_path = './data/tp/'
        stock_code = '601006'
        start_date = '2006-08-01'
        end_date = '2007-04-25'
        n_way = 3
        test_data_path = './data/Omniglot/images_evaluation/'
        k_shot = 1
        q_query = 1
        test_batches = 2
        meta_lr = 0.005
        X = torch.from_numpy(X.reshape(1, 1, 25)).to(self.device)
        meta_model = AsmlModel(1, n_way).to(self.device)
        meta_model.load_state_dict(torch.load(self.chpt_file))
        optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        fast_weights = OrderedDict(meta_model.named_parameters())
        logits = meta_model.functional_forward(X, fast_weights)
        print(logits)
        labels = torch.argmax(logits, dim=1)
        print('predict label: {0};'.format(labels[0]))


    def train_meta_batch(self, model, optimizer, xs, ys, n_way, k_shot, 
                q_query, loss_fn, inner_train_steps= 1, 
                inner_lr = 0.4, train = True):
        """
        Args:
        x is the input omniglot images for a meta_step, shape = 
                [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
        n_way: 每個分類的 task 要有幾個 class
        k_shot: 每個類別在 training 的時候會有多少張照片
        q_query: 在 testing 時，每個類別會用多少張照片 update
        """
        criterion = loss_fn
        task_loss = [] # 這裡面之後會放入每個 task 的 loss 
        task_acc = []  # 這裡面之後會放入每個 task 的 loss 
        task_len = len(xs)
        batch_losses = []
        batch_accs = []
        for i in range(task_len):
            batch_loss = []
            batch_acc = []
            for meta_batch, label in zip(xs[i], ys[i]):
                train_set = meta_batch[:n_way*k_shot] # train_set 是我們拿來 update inner loop 參數的 data
                train_label = label[:n_way*k_shot]
                val_set = meta_batch[n_way*k_shot:]   # val_set 是我們拿來 update outer loop 參數的 data
                val_label = label[n_way*k_shot:]
                fast_weights = OrderedDict(model.named_parameters()) # 在 inner loop update 參數時，
                #print('fast_weights: {0};'.format(fast_weights))
                # 我們不能動到實際參數，因此用 fast_weights 來儲存新的參數 θ'
                for inner_step in range(inner_train_steps): # 這個 for loop 是 Algorithm2 的 line 7~8
                                                    # 實際上我們 inner loop 只有 update 一次 gradients，
                                                    # 不過某些 task 可能會需要多次 update inner loop 的 θ'，
                                                    # 所以我們還是用 for loop 來寫
                    logits = model.functional_forward(train_set, fast_weights)
                    loss = criterion(logits, train_label)
                    grads = torch.autograd.grad(loss, fast_weights.values(), create_graph = True) # 這裡是要計算出 loss 對 θ 的微分 (∇loss)    
                    fast_weights = OrderedDict((name, param - inner_lr * grad)
                                    for ((name, param), grad) in zip(fast_weights.items(), grads)) 
                    # 這裡是用剛剛算出的 ∇loss 來 update θ 變成 θ'
                logits = model.functional_forward(val_set, fast_weights) # 這裡用 val_set 和 θ' 算 logit
                val = criterion(logits, val_label)
                batch_loss.append(val)                      # 這裡用 val_set 和 θ' 算 loss
                batch_acc.append(np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean()) # 算 accuracy
            batch_losses.append(batch_loss)
            batch_accs.append(batch_acc)
        task_loss = self.norm_task_loss(batch_losses)
        task_acc = self.norm_task_acc(batch_accs)
        model.train()
        optimizer.zero_grad()
        meta_batch_loss = torch.stack(task_loss).mean() # 我們要用一整個 batch 的 loss 來 update θ (不是 θ')
        if train:
            meta_batch_loss.backward()
            optimizer.step()
        task_acc = np.mean(task_acc)
        return meta_batch_loss, task_acc



















    def get_max_ds_len(self, stock_codes, start_date, end_date, k_shot, q_query, n_way):
        max_len = 0
        for stock_code in stock_codes:
            ds = AsdkMetaDs(self.train_data_path, stock_code, start_date, end_date, k_shot, q_query)
            if ds.X.shape[0] > max_len:
                max_len = ds.X.shape[0]
        return max_len

    def load_dataset_meta(self, stock_code, start_date, end_date, max_len, k_shot, q_query, n_way):
        ds = AsdkMetaDs(self.train_data_path, stock_code, start_date, end_date, k_shot, q_query)
        ds_num = ds.X.shape[0]
        for i in range(ds_num, max_len):
            ds.padding_last_rec()
        ds_len = len(ds)
        test_size = int(ds_len * 0.1)
        train_set, val_set = torch.utils.data.random_split(ds, [ds_len - test_size, test_size])
        n_way = 3
        train_loader = DataLoader(train_set,
                                batch_size = n_way, # 這裡的 batch size 並不是 meta batch size, 而是一個 task裡面會有多少不同的
                                                    # characters，也就是 few-shot classifiecation 的 n_way
                                num_workers = 1,
                                shuffle = True,
                                drop_last = True)
        train_iter = iter(train_loader)
        val_loader = DataLoader(val_set,
                                batch_size = n_way,
                                num_workers = 1,
                                shuffle = True,
                                drop_last = True)
        val_iter = iter(val_loader)
        return train_loader, train_iter, val_loader, val_iter

        

    def load_dataset_product(self, stock_code, start_date, end_date, n_way):
        ds = AsdkProductDs(self.train_data_path, stock_code, start_date, end_date, 0, 0, 1)
        ds_len = len(ds)
        test_size = int(ds_len * 0.1)
        train_set, val_set = torch.utils.data.random_split(ds, [ds_len - test_size, test_size])
        train_loader = DataLoader(train_set,
                                batch_size = n_way, # 這裡的 batch size 並不是 meta batch size, 而是一個 task裡面會有多少不同的
                                                    # characters，也就是 few-shot classifiecation 的 n_way
                                num_workers = 1,
                                shuffle = True,
                                drop_last = True)
        train_iter = iter(train_loader)
        val_loader = DataLoader(val_set,
                                batch_size = n_way,
                                num_workers = 1,
                                shuffle = True,
                                drop_last = True)
        val_iter = iter(val_loader)
        return train_loader, train_iter, val_loader, val_iter

    def get_meta_batch(self, meta_batch_size, k_shot, q_query, data_loader, iterator):
        data = []
        label = []
        for i in range(meta_batch_size):
            try:
                task_data, task_label = iterator.next()  # 一筆 task_data 就是一個 task 裡面的 data，
                #大小是 [n_way, k_shot+q_query, 1, 28, 28]
            except StopIteration:
                iterator = iter(data_loader)
                task_data, task_label = iterator.next()
            train_data = task_data[:, :k_shot].reshape(-1, 1, 25)
            val_data = task_data[:, k_shot:].reshape(-1, 1, 25)
            task_data = torch.cat((train_data, val_data), 0)
            data.append(task_data)
            train_label = task_label[:, :k_shot]
            train_label = train_label.reshape((train_label.shape[0] * train_label.shape[1]))
            val_label = task_label[:, k_shot:]
            val_label = val_label.reshape((val_label.shape[0]*val_label.shape[1]))
            task_label = torch.cat((train_label, val_label), 0)
            label.append(task_label)
        return torch.stack(data).to(self.device), torch.stack(label).to(self.device), iterator
    
    def norm_task_loss(self, batch_losses):
        task_num = len(batch_losses)
        batch_size = len(batch_losses[0])
        for i in range(1, task_num):
            for j in range(batch_size):
                batch_losses[0][j].add(batch_losses[i][j])
        for j in range(batch_size):
            batch_losses[0][j] *= 1 / task_num
        return batch_losses[0]

    def norm_task_acc(self, batch_accs):
        task_num = len(batch_accs)
        batch_size = len(batch_accs[0])
        for i in range(1, task_num):
            for j in range(batch_size):
                batch_accs[0][j] += batch_accs[i][j]
        for j in range(batch_size):
            batch_accs[0][j] *= 1 / task_num
        return batch_accs[0]
