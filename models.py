from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    # print(scores.shape, targets.shape)
                    # assert 1==2
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks



class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in sizes
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        
        # self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        # self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        # self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        # self.lhs.weight.data *= init_size
        # self.rel.weight.data *= init_size
        # self.rhs.weight.data *= init_size

    def score(self, x):
        # lhs = self.lhs(x[:, 0])
        # rel = self.rel(x[:, 1])
        # rhs = self.rhs(x[:, 2])

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[2](x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        # lhs = self.lhs(x[:, 0])
        # rel = self.rel(x[:, 1])
        # rhs = self.rhs(x[:, 2])
        # return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[2](x[:, 2])
        return (lhs * rel) @ self.embeddings[2].weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        # return self.rhs.weight.data[
        #     chunk_begin:chunk_begin + chunk_size
        # ].transpose(0, 1)

        return self.embeddings[2].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.embeddings[0](queries[:, 0]).data * self.embeddings[1](queries[:, 1]).data


class Distmult(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(Distmult, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in sizes[:2]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        
        # self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        # self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        # self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        # self.lhs.weight.data *= init_size
        # self.rel.weight.data *= init_size
        # self.rhs.weight.data *= init_size

    def score(self, x):
        # lhs = self.lhs(x[:, 0])
        # rel = self.rel(x[:, 1])
        # rhs = self.rhs(x[:, 2])

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        # lhs = self.lhs(x[:, 0])
        # rel = self.rel(x[:, 1])
        # rhs = self.rhs(x[:, 2])
        # return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        return (lhs * rel) @ self.embeddings[0].weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        # return self.rhs.weight.data[
        #     chunk_begin:chunk_begin + chunk_size
        # ].transpose(0, 1)

        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.embeddings[0](queries[:, 0]).data * self.embeddings[1](queries[:, 1]).data


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()

        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)


class RESCAL(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(RESCAL, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank),
            nn.Embedding(sizes[1], rank * rank),
        ])

        nn.init.xavier_uniform_(tensor=self.embeddings[0].weight)
        nn.init.xavier_uniform_(tensor=self.embeddings[1].weight)

        # self.lhs = self.embeddings[0]
        # self.rel = self.embeddings[1]
        # self.rhs = self.embeddings[0]

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1]).reshape(-1, self.rank, self.rank)
        rhs = self.embeddings[0](x[:, 2])
        
        # self.rhs = self.embeddings[0]
        return (torch.bmm(lhs.unsqueeze(1), rel)).squeeze() @ self.embeddings[0].weight.t(), (lhs, rel, rhs)

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1]).reshape(-1, self.rank, self.rank)
        rhs = self.embeddings[0](x[:, 2])

        return torch.bmm(torch.bmm(lhs.unsqueeze(1), rel), rhs.unsqueeze(-1)).squeeze(-1) # bs*1

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0]).data
        rel = self.embeddings[1](queries[:, 1]).data.reshape(-1, self.rank, self.rank)
        return (torch.bmm(lhs.unsqueeze(1), rel)).squeeze()