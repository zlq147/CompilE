from compute_embs import compute_cp_emb, compute_distmult_emb, compute_complex_emb, compute_RESCAL_emb
from collections import defaultdict
import pickle
import tqdm
import random
import torch
import os


class regularizer_composite():
    def __init__(self, args, name, train_triples, nentity, nrelation):
        self.name = name
        self.train_kg_mode = {'h':{},'t':{},'hr':{},'rt':{},'hrt':{}, 'r':{}, 'ht':{}, 'r-h': {}}
        self.ReadKG(train_triples)

        self.nentity = nentity
        self.n_pos = args.n_pos
        self.forward_model = args.model

        self.nrelation = nrelation
        mode_list = args.mode_list.split(';')

        self.build_lists(args, mode_list)

        self.neighbors = defaultdict(dict)
        self.neighbors_size = defaultdict(dict)
        self.MAX_K = 30
        self.load_cached_neighbors = True
        self.build_neighbors()

    def mode2tuple(self, triple, mode, fill_zero = False):
        h,r,t = triple
        ret = []
        for m in 'hrt':
            if m in mode:
                ret.append(locals()[m])
            elif fill_zero:
                ret.append(0)
        return tuple(ret)

    def build_lists(self, args, mode_list):
        self.mode_list = []
        self.weight_list = []
        weight_dict = {1: args.w1, 2: args.w2, 3: args.w3}
        self.dist_func = args.fact_dist
        for mode in mode_list:
            source_mode, feature_mode, emb_mode = mode.split('-')
            weight = weight_dict[len(emb_mode)]
            if weight != 0:
                self.mode_list.append('-'.join([source_mode, feature_mode, emb_mode]))
                self.weight_list.append(weight)

    def ReadKG(self, train_triples): # parse KG, e.g. get corresponding triples with head h or tail t
        triple_set = set()
        if isinstance(train_triples, torch.Tensor):
            train_triples = train_triples.cpu().detach().numpy()
        for s, p, o in train_triples:
            if tuple([s]) not in self.train_kg_mode['h']:
                self.train_kg_mode['h'][tuple([s])] = []
            self.train_kg_mode['h'][tuple([s])].append((s,p,o))
            if tuple([o]) not in self.train_kg_mode['t']:
                self.train_kg_mode['t'][tuple([o])] = []
            self.train_kg_mode['t'][tuple([o])].append((s,p,o))
            if (s,p) not in self.train_kg_mode['hr']:
                self.train_kg_mode['hr'][(s,p)] = []
            self.train_kg_mode['hr'][(s, p)].append((s,p,o))
            if (p,o) not in self.train_kg_mode['rt']:
                self.train_kg_mode['rt'][(p,o)] = []
            self.train_kg_mode['rt'][(p, o)].append((s,p,o))
            if (s,p,o) not in self.train_kg_mode['hrt']:
                self.train_kg_mode['hrt'][(s,p,o)] = []
            self.train_kg_mode['hrt'][(s, p, o)].append((s,p,o))
            if tuple([p]) not in self.train_kg_mode['r']:
                self.train_kg_mode['r'][tuple([p])] = []
            self.train_kg_mode['r'][tuple([p])].append((s,p,o))
            if (s,o) not in self.train_kg_mode['ht']:
                self.train_kg_mode['ht'][(s,o)] = []
            self.train_kg_mode['ht'][(s, o)].append((s,p,o))

            h_triple = (s,0,0)
            if p not in self.train_kg_mode['r-h']:
                self.train_kg_mode['r-h'][p] = set()
            self.train_kg_mode['r-h'][p].add(h_triple)

            triple_set.add((s,p,o))

        self.train_triple_list = list(triple_set)

    def dist(self, f1, f2, id): # compute distance between two entities
        if self.dist_func == "jaccard":
            return len(f1.intersection(f2)) / len(f1.union(f2))
        else:
            return random.random()

    def fill_zeros(self): # pad for diffeerent numbers of connected(h), defined in Eq (24) of Section 4.3
        for mode in self.neighbors:
            for t1 in self.neighbors[mode]:
                while len(self.neighbors[mode][t1]) < self.n_pos:
                    self.neighbors[mode][t1].append((0,0,0))

    def load_pickle(self): # load existing connected(h), defined in Eq (24) of Section 4.3
        if self.dist_func == "jaccard":
            neighbors_file = self.name + '_neighbors.pkl'
            neighbors_size_file = self.name + '_neighbors_size.pkl'
        else:
            neighbors_file = self.name + '_neighbors_rand.pkl'
            neighbors_size_file = self.name + '_neighbors_size_rand.pkl'
        if os.path.exists(neighbors_file):
            print('load neighbors pickle')
            with open(neighbors_file, "rb") as f:
                self.neighbors = pickle.load(f)
            with open(neighbors_size_file, "rb") as f:
                self.neighbors_size = pickle.load(f)
            print('load neighbors pickle finished')
            print('loaded' + str([t for t in self.neighbors_size.keys()]))

    def dump_pickle(self): # save connected(h), defined in Eq (24) of Section 4.3
        if self.dist_func == "jaccard":
            neighbors_file = self.name + '_neighbors.pkl'
            neighbors_size_file = self.name + '_neighbors_size.pkl'
        else:
            neighbors_file = self.name + '_neighbors_rand.pkl'
            neighbors_size_file = self.name + '_neighbors_size_rand.pkl'
        print('dump neighbors pickle')
        with open(neighbors_file, "wb") as f:
            pickle.dump(self.neighbors, f)
        with open(neighbors_size_file, "wb") as f:
            pickle.dump(self.neighbors_size, f)
        print('dump neighbors pickle finished')

    def build_neighbors(self): # build connected(h), defined in Eq (24) of Section 4.3
        if self.load_cached_neighbors:
            self.load_pickle()
        modify_flag = False

        loaded_modes = [t for t in self.neighbors_size.keys()]
        for mode in loaded_modes:
            if mode.count('-') == 1:
                source_mode, feature_mode = mode.split('-')
                emb_mode = source_mode
                mode_new = '-'.join([source_mode, feature_mode, emb_mode])
                if mode_new not in self.neighbors:
                    self.neighbors[mode_new] = self.neighbors.pop(mode)
                modify_flag = True
        loaded_modes = [t for t in self.neighbors_size.keys()]

        for mode in self.mode_list:
            source_mode, feature_mode, emb_mode = mode.split('-')
            if mode in self.neighbors:
                continue

            modify_flag = True

            self.neighbors[mode] = {}
            tuple_features = {}
            features_tuple = defaultdict(list)
            for triple in tqdm.tqdm(self.train_triple_list,desc='build features (neighbors) {}'.format(mode)):
                this_tuple = self.mode2tuple(triple,source_mode,True)
                this_tuple_key = self.mode2tuple(triple,source_mode)
                if this_tuple in tuple_features:
                    continue
                features = [self.mode2tuple(triple,feature_mode) for triple in self.train_kg_mode[source_mode][this_tuple_key]]
                tuple_features[this_tuple] = set(features)
                for f in features:
                    features_tuple[f].append(this_tuple)
            for t1 in tqdm.tqdm(tuple_features, desc = 'compute neighbors {}'.format(mode)): # [(1174,186,0),(2160,186,0),(4994,186+474//2,0)]:
                neighbors = set()
                for f in tuple_features[t1]:
                    for t2 in features_tuple[f]:
                        neighbors.add(t2)
                neighbors.discard(t1)
                visited = defaultdict(set)
                dist = defaultdict(list)
                for t2 in neighbors:
                    score = round(self.dist(tuple_features[t1],tuple_features[t2],t2[0]),5)
                    for emb_mode in ['hrt','hr','h','rt','t']:
                        mode_this = '-'.join([source_mode,feature_mode,emb_mode])
                        if mode_this in self.mode_list and mode_this not in loaded_modes:
                            t3 = self.mode2tuple(t2,emb_mode,True)
                            if t3 not in visited[mode_this]: #and len(visited[mode_this]) < self.MAX_K:
                                visited[mode_this].add(t3)
                                dist[mode_this].append([t3,score])

                for emb_mode in ['hrt', 'hr', 'h', 'rt', 't']:
                    mode_this = '-'.join([source_mode, feature_mode, emb_mode])
                    if mode_this in self.mode_list and mode_this not in loaded_modes:
                        dist[mode_this] = sorted(dist[mode_this], key=lambda x: x[1], reverse=True)
                        self.neighbors_size[mode_this][t1] = min(self.MAX_K,len(dist[mode_this]))
                        self.neighbors[mode_this][t1] = [t[0] for t in dist[mode_this][:self.neighbors_size[mode_this][t1]]]
                        while len(self.neighbors[mode_this][t1]) < self.MAX_K:
                            self.neighbors[mode_this][t1].append((0,0,0))
        if modify_flag and self.load_cached_neighbors:
            self.dump_pickle()

        self.fill_zeros()

    def least_square_dis(self, x, vecs, sz): # bs * 1 * dim,  bs * n_pos' * dim

        #x = x / x.norm(dim=-1,p=2).unsqueeze(-1)
        ret = []

        sz_gpu = torch.LongTensor(sz).cuda()
        sz_set = set(sz)
        for s in sz_set:
            if s == 0:
                continue
            mask = sz_gpu.eq(s)
            indices = torch.nonzero(mask).squeeze(1) #bs
            x_this = x[indices,0,:] # g_s * dim
            vecs_this = vecs[indices,:s,:] # g_s * s * dim
            basis, r = torch.linalg.qr(vecs_this.transpose(2, 1), mode='reduced')
            basis = basis.transpose(2, 1)  # g_s * s * dim
            d = torch.einsum('gd,gsd->gs', x_this, basis)  # g_s, s
            p = torch.einsum('gs,gsd->gd',d, basis) # g_s * d

            res = (torch.abs(x_this - p) ** 2).sum(-1).sqrt() / (x_this**2).sum(-1).sqrt()

            ret.append(res)  # g_s * 1
        
        if len(ret) == 0:
            return 0*x.sum()
        else:
            return torch.cat(ret, 0).mean()

    def get_pos_emb(self, x, embeddings, source_mode, feature_mode, emb_mode): # get positive and negative embedding for a batch
        mode = '-'.join([source_mode,feature_mode,emb_mode])

        nx = x.cpu().detach().numpy()
        size = []
        idx = []

        empty_triples = []
        while len(empty_triples) < self.n_pos:
            empty_triples.append((0, 0, 0))

        for h, r, t in nx:
            t1 = self.mode2tuple((h,r,t),source_mode,True)
            s = min(self.neighbors_size[mode][t1],self.n_pos)
            if self.neighbors_size[mode][t1] >= self.MAX_K:
                triples = empty_triples
                idx.append(triples)
                size.append(0)
            else:
                triples = random.sample(self.neighbors[mode][t1][:self.neighbors_size[mode][t1]],s)
                while len(triples) < self.n_pos:
                    triples.append((0,0,0))
                idx.append(triples)
                size.append(s)

        idx = torch.LongTensor(idx).cuda()
        h_emb = embeddings[0](idx[:, :, 0])
        r_emb = embeddings[1](idx[:, :, 1])
        t_emb = embeddings[0](idx[:, :, 2]) if self.forward_model != 'CP' else embeddings[2](idx[:, :, 2])

        return h_emb, r_emb, t_emb, idx, size


    def forward(self, x, embeddings, model = None): # compute \hat{cr} defined in Eq (25) of Sec 4.5
        dis_func = self.least_square_dis
        emb_func = {'ComplEx':compute_complex_emb, 'CP':compute_cp_emb,
                    'Distmult':compute_distmult_emb, 'RESCAL':compute_RESCAL_emb}[self.forward_model]

        ret = None
        log = {}

        for i,(mode,weight) in enumerate(zip(self.mode_list,self.weight_list)):
            source_mode, feature_mode, emb_mode = mode.split('-')

            pos_emb_h, pos_emb_r, pos_emb_t, tuples, pos_size = self.get_pos_emb(x, embeddings, source_mode, feature_mode, emb_mode)
  
            pos_emb = emb_func(pos_emb_h, pos_emb_r, pos_emb_t,emb_mode)
            target_emb = emb_func(embeddings[0](x[:, 0]).unsqueeze(1),
                                                embeddings[1](x[:, 1]).unsqueeze(1),
                                                embeddings[0](x[:, 2]).unsqueeze(1), emb_mode)
            
            i_reg = dis_func(target_emb, pos_emb, pos_size) * weight
            i_reg /= len(self.mode_list)

            if ret is None:
                ret = i_reg
            else:
                ret += i_reg
            log[str(i)] = i_reg.item()

        return ret, log