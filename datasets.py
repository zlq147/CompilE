from pathlib import Path
import pickle
from typing import Dict, Tuple, List

import numpy as np
import torch
from models import KBCModel
import os

DATA_PATH = Path('data')

class Dataset(object):
    def __init__(self, name: str, data_path = None):
        if data_path is None:
            self.root = DATA_PATH / name
            print('data path', DATA_PATH)
        else:
            if name == 'FB237':
                name = 'FB15k-237_pickle'
            elif name == 'WN18RR':
                name = 'WN18RR_pickle'
            self.root = Path(data_path) / name
            print('data path', data_path)

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        #maxis = max(np.max(self.data['train'], axis=0),np.max(self.data['valid'], axis=0),np.max(self.data['test'], axis=0))
        maxis = np.max(self.data['train'], axis=0)
        maxis2 = np.max(self.data['valid'], axis=0)
        maxis3 = np.max(self.data['test'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2], maxis2[0], maxis2[2], maxis3[0], maxis3[2]) + 1)
        self.n_predicates = int(max(maxis[1], maxis2[1], maxis3[1]) + 1)
        self.n_predicates *= 2

        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        '''to_skip_lhs = []
        for hr,ts in self.to_skip['lhs'].items():
            for t in ts:
                to_skip_lhs.append((hr[0],hr[1],t))
        to_skip_rhs = []
        for hr,ts in self.to_skip['rhs'].items():
            for t in ts:
                to_skip_rhs.append((hr[0],hr[1],t))'''

        inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['train'], copy))

    def get_weight(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['train'])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1

        w = appear_list / np.max(appear_list) * 0.9 + 0.1
        return w

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        
        model.eval()
        with torch.no_grad():
            test = self.get_examples(split)
            examples = torch.from_numpy(test.astype('int64')).cuda()
            missing = [missing_eval]
            if missing_eval == 'both':
                missing = ['rhs', 'lhs']

            mean_rank = {}
            mean_reciprocal_rank = {}
            hits_at = {}

            for m in missing:
                q = examples.clone()
                if n_queries > 0:
                    permutation = torch.randperm(len(examples))[:n_queries]
                    q = examples[permutation]
                if m == 'lhs':
                    tmp = torch.clone(q[:, 0])
                    q[:, 0] = q[:, 2]
                    q[:, 2] = tmp
                    q[:, 1] += self.n_predicates // 2
                ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
                mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
                hits_at[m] = torch.FloatTensor((list(map(
                    lambda x: torch.mean((ranks <= x).float()).item(),
                    at
                ))))
                mean_rank[m] = torch.mean(ranks).item()

        return mean_rank, mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities
