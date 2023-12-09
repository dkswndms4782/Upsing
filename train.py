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
    preprocess.load_data(args.train_file_name,args.test_file_name)
    train_data, test_data = preprocess.get_data()

    if args.train_all:
        trainer.run_all(args,train_data, test_data)
    else:
        trainer.run(args,train_data, test_data)
    

    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
