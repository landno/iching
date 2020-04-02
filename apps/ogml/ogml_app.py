#
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader #, Dataset
from collections import OrderedDict
from apps.ogml.omniglot_ds import OmniglotDs
from apps.ogml.ogml_model import OgmlModel

class OgmlApp(object):
    def __init__(self):
        self.name = 'apps.ogml.OgmlApp'
        self.chpt_file = './work/ogml.pkl'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def startup(self):
        print('Omniglot MAML app startup')
        # self.train()
        self.evaluate_on_test_ds()

    def train(self):
        n_way = 5
        k_shot = 1
        q_query = 1
        inner_train_steps = 1
        inner_lr = 0.4
        meta_lr = 0.001
        meta_batch_size = 32
        max_epoch = 4 #40
        eval_batches = 20
        train_data_path = './data/Omniglot/images_background/'
        dataset = OmniglotDs(train_data_path, k_shot, q_query)

        train_set, val_set = torch.utils.data.random_split(OmniglotDs(train_data_path, k_shot, q_query), [3200,656])
        train_loader = DataLoader(train_set,
                                batch_size = n_way, # 這裡的 batch size 並不是 meta batch size, 而是一個 task裡面會有多少不同的
                                                    # characters，也就是 few-shot classifiecation 的 n_way
                                num_workers = 8,
                                shuffle = True,
                                drop_last = True)
        val_loader = DataLoader(val_set,
                                batch_size = n_way,
                                num_workers = 8,
                                shuffle = True,
                                drop_last = True)
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        #

        meta_model = OgmlModel(1, n_way).to(self.device)
        optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        for epoch in range(max_epoch):
            print("Epoch %d" %(epoch))
            train_meta_loss = []
            train_acc = []
            for step in tqdm(range(len(train_loader) // (meta_batch_size))): # 這裡的 step 是一次 meta-gradinet update step
                x, train_iter = self.get_meta_batch(meta_batch_size, k_shot, q_query, train_loader, train_iter)
                meta_loss, acc = self.train_batch(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn)
                train_meta_loss.append(meta_loss.item())
                train_acc.append(acc)
            print("  Loss    : ", np.mean(train_meta_loss))
            print("  Accuracy: ", np.mean(train_acc))

            # 每個 epoch 結束後，看看 validation accuracy 如何  
            # 助教並沒有做 early stopping，同學如果覺得有需要是可以做的 
            val_acc = []
            for eval_step in tqdm(range(len(val_loader) // (eval_batches))):
                x, val_iter = self.get_meta_batch(eval_batches, k_shot, q_query, val_loader, val_iter)
                _, acc = self.train_batch(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_steps = 3, train = False) # testing時，我們更新三次 inner-step
                val_acc.append(acc)
            print("  Validation accuracy: ", np.mean(val_acc))
        print('train is OK!')
        torch.save(meta_model.state_dict(), self.chpt_file)


    def evaluate_on_test_ds(self):
        n_way = 5
        test_data_path = './data/Omniglot/images_evaluation/'
        k_shot = 1
        q_query = 1
        test_batches = 20
        meta_lr = 0.001
        test_loader = DataLoader(OmniglotDs(test_data_path, k_shot, q_query),
                                batch_size = n_way,
                                num_workers = 8,
                                shuffle = True,
                                drop_last = True)
        test_iter = iter(test_loader)
        test_acc = []

        meta_model = OgmlModel(1, n_way).to(self.device)
        meta_model.load_state_dict(torch.load(self.chpt_file))
        optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        for test_step in tqdm(range(len(test_loader) // (test_batches))):
            x, val_iter = self.get_meta_batch(test_batches, k_shot, q_query, test_loader, test_iter)
            _, acc = self.train_batch(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_steps = 3, train = False) # testing時，我們更新三次 inner-step
            test_acc.append(acc)
        print("  Testing accuracy: ", np.mean(test_acc))

    def create_label(self, n_way, k_shot):
        return torch.arange(n_way).repeat_interleave(k_shot).long()

    def train_batch(self, model, optimizer, x, n_way, k_shot, 
                q_query, loss_fn, inner_train_steps= 1, 
                inner_lr = 0.4, train = True):
        """
        Args:
        x is the input omniglot images for a meta_step, shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
        n_way: 每個分類的 task 要有幾個 class
        k_shot: 每個類別在 training 的時候會有多少張照片
        q_query: 在 testing 時，每個類別會用多少張照片 update
        """
        criterion = loss_fn
        task_loss = [] # 這裡面之後會放入每個 task 的 loss 
        task_acc = []  # 這裡面之後會放入每個 task 的 loss 
        for meta_batch in x:
            train_set = meta_batch[:n_way*k_shot] # train_set 是我們拿來 update inner loop 參數的 data
            val_set = meta_batch[n_way*k_shot:]   # val_set 是我們拿來 update outer loop 參數的 data
            fast_weights = OrderedDict(model.named_parameters()) # 在 inner loop update 參數時，我們不能動到實際參數，因此用 fast_weights 來儲存新的參數 θ'
            for inner_step in range(inner_train_steps): # 這個 for loop 是 Algorithm2 的 line 7~8
                                                # 實際上我們 inner loop 只有 update 一次 gradients，不過某些 task 可能會需要多次 update inner loop 的 θ'，
                                                # 所以我們還是用 for loop 來寫
                train_label = self.create_label(n_way, k_shot).to(self.device)
                logits = model.functional_forward(train_set, fast_weights)
                loss = criterion(logits, train_label)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph = True) # 這裡是要計算出 loss 對 θ 的微分 (∇loss)    
                fast_weights = OrderedDict((name, param - inner_lr * grad)
                                  for ((name, param), grad) in zip(fast_weights.items(), grads)) # 這裡是用剛剛算出的 ∇loss 來 update θ 變成 θ'
            val_label = self.create_label(n_way, q_query).to(self.device)
            logits = model.functional_forward(val_set, fast_weights) # 這裡用 val_set 和 θ' 算 logit
            loss = criterion(logits, val_label)                      # 這裡用 val_set 和 θ' 算 loss
            task_loss.append(loss)                                   # 把這個 task 的 loss 丟進 task_loss 裡面
            acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean() # 算 accuracy
            task_acc.append(acc)
        model.train()
        optimizer.zero_grad()
        meta_batch_loss = torch.stack(task_loss).mean() # 我們要用一整個 batch 的 loss 來 update θ (不是 θ')
        if train:
            meta_batch_loss.backward()
            optimizer.step()
        task_acc = np.mean(task_acc)
        return meta_batch_loss, task_acc

    def get_meta_batch(self, meta_batch_size, k_shot, q_query, data_loader, iterator):
        data = []
        for _ in range(meta_batch_size):
            try:
                task_data = iterator.next()  # 一筆 task_data 就是一個 task 裡面的 data，大小是 [n_way, k_shot+q_query, 1, 28, 28]
            except StopIteration:
                iterator = iter(data_loader)
                task_data = iterator.next()
            train_data = task_data[:, :k_shot].reshape(-1, 1, 28, 28)
            val_data = task_data[:, k_shot:].reshape(-1, 1, 28, 28)
            task_data = torch.cat((train_data, val_data), 0)
            data.append(task_data)
        return torch.stack(data).to(self.device), iterator