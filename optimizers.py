import tqdm
import torch
from torch import nn
from torch import optim

from models import KBCModel
from regularizers import Regularizer
import numpy as np

class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples: torch.LongTensor, weight=None):
        self.model.train()
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        l_list = []
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                # input_batch = actual_examples[
                #     b_begin:b_begin + self.batch_size
                # ]

                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)
                l = l_fit + l_reg
                l.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                l_list.append(l.item())
                # if b_begin >= 5120:
                #     print(l_list)
                bar.set_postfix(loss=f'{np.mean(l_list):.4f}')


class KBCOptimizer_lowrank():
    def __init__(
            self, args, model: KBCModel, regularizer, optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True, regularizer_N3 = None, regularizer_DURA = None, regularizer_DURA_W = None, regularizer_DURA_RESCAL = None
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

        self.regularizer_N3 = regularizer_N3
        self.regularizer_DURA = regularizer_DURA
        self.regularizer_DURA_W = regularizer_DURA_W
        self.regularizer_DURA_RESCAL = regularizer_DURA_RESCAL

        self.use_N3 = args.use_N3
        self.use_DURA = args.use_DURA
        self.use_DURA_W = args.use_DURA_W
        self.use_DURA_RESCAL = args.use_DURA_RESCAL

        self.no_reg = args.no_reg
        self.fully_train = args.fully_train
        self.train_num = args.train_num

        self.curri_learn = args.curri_learn
        self.curri_epochs = args.curri_epochs
        self.epoch_num = 0.0

    def epoch(self, examples: torch.LongTensor, weight = None): # train model for one epoch
        self.epoch_num += 1
        self.model.train()
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean', weight = weight)
        log = {'loss':[], 'reg': []}
        train_num = examples.shape[0] if self.fully_train else self.train_num
        scaler = torch.cuda.amp.GradScaler()
        with tqdm.tqdm(total=train_num, unit='ex', disable=not self.verbose) as bar:
            #bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < train_num:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                
                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                if self.no_reg == False:
                    l_reg, batch_log = self.regularizer.forward(input_batch, self.model.embeddings) # compute \hat{cr} defined in Theorem 4.2 of Sec 4.3
                    if self.curri_learn:
                        l_reg = l_reg * min(self.epoch_num / self.curri_epochs, 1.0)

                    if self.use_N3:
                        l_reg += self.regularizer_N3.forward(factors)
                    elif self.use_DURA:
                        l_reg += self.regularizer_DURA.forward(factors)
                    elif self.use_DURA_W:
                        l_reg += self.regularizer_DURA_W.forward(factors)
                    elif self.use_DURA_RESCAL:
                        l_reg += self.regularizer_DURA_RESCAL.forward(factors)

                    l = l_fit + l_reg # total loss defined in Eq (26) of Sec 4.5
                    for t in batch_log:
                        if t not in log:
                            log[t] = []
                        log[t].append(batch_log[t])
                    log['reg'].append(l_reg.item())
                else:
                    l = l_fit
                l.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                log['loss'].append(l.item())
                desc = []
                for t in log:
                    desc.append('{}:{:.3f}'.format(t,np.mean(log[t])))
                bar.set_description(' '.join(desc))