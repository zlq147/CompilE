import math
import torch

def compute_cp_emb(h,r,t, mode ='hrt'):
    #mode = 'hrt'
    if len(mode) == 1:
        return locals()[mode]
    
    lhs, rel, rhs = h, r, t

    if mode == 'ht':
        return torch.cat((lhs, rhs), dim=-1)

    if mode == 'rt' or mode == 'hrt':
        rt = rhs * rel  # bs * 1 * dim
        if mode == 'rt':
            return rt
        else:
            ret = lhs*rt
            return ret
    elif mode == 'hr':
        hr = lhs * rel
        return hr

def compute_distmult_emb(h, r, t, mode='hrt'):
    #mode = 'hrt'
    if len(mode) == 1:
        return locals()[mode]
    
    lhs, rel, rhs = h, r, t

    if mode == 'ht':
        return torch.cat((lhs, rhs), dim=-1)

    if mode == 'rt' or mode == 'hrt':
        rt = rhs * rel  # bs * 1 * dim
        if mode == 'rt':
            return rt
        else:
            ret = lhs*rt
            return ret
    elif mode == 'hr':
        hr = lhs * rel
        return hr

def compute_complex_emb(h,r,t, mode ='hrt'):
    #mode = 'hrt'
    if len(mode) == 1:
        return locals()[mode]
    rank = h.shape[-1]//2

    lhs = h[:, :, :rank], h[:, :, rank:]
    rel = r[:, :, :rank], r[:, :, rank:]
    rhs = t[:, :, :rank], t[:, :, rank:]

    if mode == 'ht':
        ht_re = torch.cat((lhs[0],rhs[0]),dim=-1)  # bs * 1 * dim
        ht_im = torch.cat((lhs[1],-rhs[1]),dim=-1)
        return torch.cat((ht_re, ht_im), dim=-1)

    if mode == 'rt' or mode == 'hrt':
        rt_re = rhs[0] * rel[0] + rhs[1] * rel[1]  # bs * 1 * dim
        rt_im = rhs[0] * rel[1] - rhs[1] * rel[0]
        if mode == 'rt':
            return torch.cat((rt_re,rt_im),dim=-1)
        else:
            ret = torch.cat((lhs[0] * rt_re - lhs[1] * rt_im, lhs[0] * rt_im + lhs[1] * rt_re), dim=-1)
            return ret
    elif mode == 'hr':
        hr_re = lhs[0] * rel[0] - lhs[1] * rel[1]
        hr_im = lhs[0] * rel[1] + lhs[1] * rel[0]
        return torch.cat((hr_re,hr_im),dim=-1)
    assert ValueError('no such mode')

def compute_RESCAL_emb(h,r,t, mode ='hrt'):
    #mode = 'hrt'
    if len(mode) == 1:
        return locals()[mode]

    lhs, rhs = h, t
    bs, num_sample, sq_rank = r.shape
    rank = int(math.sqrt(sq_rank))
    rel = r.reshape(bs, num_sample, rank, rank)

    if mode == 'ht':
        return torch.cat((lhs, rhs), dim=-1)

    if mode == 'rt' or mode == 'hrt':
        rt = torch.einsum('abij,abj->abi', rel, rhs)
        if mode == 'rt':
            return rt
        else:
            ret = lhs*rt
            return ret
    elif mode == 'hr':
        hr = torch.einsum('abi, abij->abj', lhs, rel)
        return hr
    assert ValueError('no such mode')