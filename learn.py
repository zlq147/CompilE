import argparse
import json
import logging
import pickle
from bisect import insort
from typing import Dict
from tqdm import tqdm

import random
import torch
from torch import optim

from datasets import Dataset
from models import CP, ComplEx, Distmult
from regularizers import F2, N3, DURA, DURA_W, DURA_RESCAL
from optimizers import KBCOptimizer, KBCOptimizer_composite
from regularizer_composite import regularizer_composite
import os
import numpy as np

import sys
print(sys.path)
big_datasets = ['WN18RR', 'FB237', 'umls', 'kinship']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets, default = 'umls',
    help="Dataset in {}".format(datasets)
)

models = ['CP', 'ComplEx', 'Distmult']
parser.add_argument(
    '--model', choices=models, default = 'ComplEx',
    help="Model in {}".format(models)
)

regularizers = ['N3', 'F2', 'composite', 'DURA', 'DURA_W', 'DURA_RESCAL']
parser.add_argument(
    '--regularizer', choices=regularizers, default='composite',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=100, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=2, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=2000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)

parser.add_argument(
    '--save_epochs', default=10, type=int
)

parser.add_argument(
    '--save_path', default='models/', type=str
)
parser.add_argument(
    '--init_checkpoint', default=None, type=str
)
parser.add_argument(
    '--data_path', default=None, type=str
)
parser.add_argument(
    '--n_pos', default=10, type=int
)
parser.add_argument(
    '--use_N3', action='store_true'
)
parser.add_argument(
    '--use_N3_weight', default=0, type=float
)

parser.add_argument(
    '--no_reg', action='store_true'
)
parser.add_argument(
    '--mode_list', type=str
)

# weights

parser.add_argument(
    '--w1', type=float, default= 0, help='weight for emb_len=1'
)

parser.add_argument(
    '--w2', type=float, default= 0, help='weight for emb_len=2'
)

parser.add_argument(
    '--w3', type=float, default= 0, help='weight for emb_len=3'
)



# whether fully_train
parser.add_argument(
    '--fully_train', action='store_true'
)

parser.add_argument(
    '--train_num', default=1000, type = int, help = 'num of examples to train'
)

parser.add_argument(
    '--fact_dist', type=str, default= 'rand', choices=['rand','jaccard']
)

# DURA related
parser.add_argument(
    '--use_DURA', action='store_true'
)

parser.add_argument(
    '--use_DURA_W', action='store_true'
)

parser.add_argument(
    '--use_DURA_RESCAL', action='store_true'
)

parser.add_argument(
    '--use_DURA_weight', default=5e-2, type=float
)

parser.add_argument(
    '--use_DURA_W_weight', default=1e-1, type=float
)

parser.add_argument(
    '--use_DURA_RESCAL_weight', default=5e-2, type=float
)

parser.add_argument('-weight', '--do_ce_weight', action='store_true')

parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()
# args.gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
args.gpu_id = 0

print('parse finished.')

def build_base(args): # build base path for checkpoints and logs
    if args.regularizer == 'composite':
        if args.use_N3:
            model_name = f'{args.model}_lN3_{args.use_N3_weight}_w1_{args.w1}_w2_{args.w2}_w3_{args.w3}_lr_{args.learning_rate}_reg_{args.regularizer}'
        else:
            model_name = f'{args.model}_lN3_0_w1_{args.w1}_w2_{args.w2}_w3_{args.w3}_lr_{args.learning_rate}_reg_{args.regularizer}'
    elif args.regularizer != '':
        model_name = f'{args.model}_rank_{args.rank}_decay1_{args.decay1}_decay2_{args.decay2}_reg_{args.regularizer}'
    elif args.no_reg:
        model_name = f'{args.model}_rank_{args.rank}_decay1_{args.decay1}_decay2_{args.decay2}_noreg'
    gpu_str = f'gpu_{args.gpu_id}'
    base_path = os.path.join(args.save_path, gpu_str, model_name)
    return base_path

def set_logger(args): # set log
    base_path = build_base(args)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    log_file = os.path.join(base_path, 'run.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def avg_both(mrs: Dict[str, float], mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]): # average metrics for rhs and lhs
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    mr = (mrs['lhs'] + mrs['rhs']) / 2.
    return {'MR':mr, 'MRR': m, 'hits@[1,3,10]': h}

def save_model(model, optimizer, save_variable_list, args): # save checkpoint
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    base_path = build_base(args)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    argparse_dict = vars(args)
    with open(os.path.join(base_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(base_path, 'checkpoint')
    )

def load_model(args): # load checkpoint
    base_path = build_base(args)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    checkpoint = torch.load(os.path.join(base_path, 'checkpoint'))
    return checkpoint

def main(args): # model training
    set_logger(args)
    logging.info('Logger setting done.')
    random.seed(args.seed)

    dataset = Dataset(args.dataset, args.data_path)
    # print('load data finished.')
    logging.info('load data finished.')
    examples = torch.from_numpy(dataset.get_train().astype('int64'))

    if args.do_ce_weight:
        ce_weight = torch.Tensor(dataset.get_weight()).cuda()
    else:
        ce_weight = None

    device = 'cuda' # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    print(dataset.get_shape())

    # build regularizer
    regularizer = None
    if args.regularizer == 'F2':
        regularizer = F2(args.reg)
    elif args.regularizer == 'N3':
        regularizer = N3(args.reg)
    elif args.regularizer == 'DURA':
        regularizer = DURA(args.reg)
    elif args.regularizer == 'DURA_W':
        regularizer = DURA_W(args.reg)
    elif args.regularizer == 'DURA_RESCAL':
        regularizer = DURA_RESCAL(args.reg)
    elif args.regularizer == 'composite':
        regularizer = regularizer_composite(args, args.dataset, examples, dataset.n_entities, dataset.n_predicates)
        
    assert regularizer is not None, "Invalid regularizer: {}".format(args.regularizer)

    model = {
        'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
        'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
        'Distmult': lambda: Distmult(dataset.get_shape(), args.rank, args.init)
    }[args.model]()

    logging.info(device)
    model = model.to(device)

    optim_method = {
        'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
        'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate),
        'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
    }[args.optimizer]()

    # build optimizer
    if args.regularizer in ['lowrank', 'composite']:
        optimizer = KBCOptimizer_composite(args, model, regularizer, optim_method, args.batch_size, \
            regularizer_N3 = N3(args.use_N3_weight), regularizer_DURA = DURA(args.use_DURA_weight), \
                regularizer_DURA_W = DURA_W(args.use_DURA_W_weight), regularizer_DURA_RESCAL = DURA_RESCAL(args.use_DURA_RESCAL_weight))
    else:
        optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)
        
    # load checkpoint if needed
    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if not args.view_predict:
            optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    
    now = []
    cur_loss = 0
    best_val_acc = 0

    curve = {'train': [], 'valid': [], 'test': []}
    for e in range(args.max_epochs): # train model
        cur_loss = optimizer.epoch(examples, weight=ce_weight) # use this to train the model

        if (e + 1) % args.valid == 0:
            valid, test = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test']
                ]

            curve['valid'].append(valid)
            curve['test'].append(test)

            val_acc = valid['MRR']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                to_save = True
            else:
                to_save = False

            logging.info(f"\t EPOCH: {e} BEST_VALID: {best_val_acc}")
            logging.info(f"\t EPOCH: {e} VALID: {valid}")
            logging.info(f"\t EPOCH: {e} TEST: {test}")

            if to_save:
                save_variable_list = {
                    'step': e+1
                }
                save_model(optimizer.model, optimizer.optimizer, save_variable_list, args)

    checkpoint = load_model(args)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    avg_valid_results = avg_both(*dataset.eval(model, 'valid', -1))
    avg_results = avg_both(*dataset.eval(model, 'test', -1))

    logging.info(f"\n\nVALID : {avg_valid_results}")
    logging.info(f"\n\nTEST : {avg_results}")

    print(avg_results)

if __name__ == '__main__':
    main(args)
