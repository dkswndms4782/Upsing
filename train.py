import os
from args import parse_args
from base_code import trainer
from base_code.dataloader import Preprocess
from base_code.utils import setSeeds
import torch
import wandb
import json
import argparse

def main(args):
    '''
    if args.use_wandb:
        wandb.login()
        wandb.init(project='upsing', config=vars(args), name = args.wandb_name)
    '''
    setSeeds(args.seed) 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    if args.type == 'teq':
        args.train_file_name = 'teq.csv'
    elif args.type == 'voc':
        args.train_file_name = 'voc.csv'
    preprocess.load_data(args.train_file_name,args.sub_file_name)
    data = preprocess.get_data()
    test_data = None

    if args.train_all:
        trainer.run_all(args,data)
    else:
        trainer.run(args,data)
    

    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
