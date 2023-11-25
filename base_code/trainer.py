import os
import torch
import numpy as np
import json
import copy
import pandas as pd
from sklearn.model_selection import KFold

from .dataloader import get_loaders,get_all_loader
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import AudioClassifier
from datetime import timedelta, timezone, datetime
import wandb

def run_all(args, data_df):
    print(f"# of data : {len(data_df)}")
    print()
    if args.use_wandb:
        wandb.login()
        wandb.init(project='upsing', config=vars(args), name = args.wandb_name + "_run_all")

        train_loader = get_all_loader(args, data_df)

        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
        args.warmup_steps = args.total_steps//10

        print(args)
        model_dir = os.path.join(args.model_dir, args.model_name)
        os.makedirs(model_dir, exist_ok = True)
        json.dump(
            vars(args),
            open(f"{model_dir}/exp_config.json", "w"),
            indent=2,
            ensure_ascii=False
            )

        print(f"\n{model_dir}/exp_config.json is saved!\n")

        model = get_model(args)
        if args.use_finetune:
            load_state = torch.load(args.trained_model)
            model.load_state_dictmodel.load_state_dict(load_state['state_dict'], strict=True)
            print(f"{args.trained_model} is loaded!")

        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        best_auc = -1
        best_acc = -1
        early_stopping_couter = 0
        ### train함수는 한번 학습하는 함수임. 그래서 epoch수만큼 train해야함
        ### train_loader는 torch.utils.data.DataLoader 사용한건데 for문으로 하나씩 부르면 batch데이터가 나옴
        for epoch in range(args.n_epochs):
        
            print(f"Start Training: Epoch {epoch}")
        
            ### TRAIN
            train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)

            ### TODO: model save or early stopping
            if args.use_wandb:
                wandb.log({"train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc})
            
            if epoch == args.n_epochs-1:
                model_to_save = model.module if hasattr(model, 'module') else model
                model_name = 'model_epoch' + str(epoch) +".pt"
                save_checkpoint(
                    {'epoch': epoch, 
                        'state_dict': model_to_save.state_dict(),
                        "train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc,
                        },
                    model_dir, model_name,
                    )

def run(args, data_df):
    print(f"# of data : {len(data_df)}")
    print()
    splits=KFold(n_splits=args.k,shuffle=False)# Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True. //  ,random_state=args.seed)
    total_auc, total_acc = 0,0
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_df)))):

        if args.use_wandb:
            wandb.login()
            wandb.init(project='upsing', config=vars(args), name = args.wandb_name + f"_fold{fold}")

        train_loader, val_loader = get_loaders(args, data_df, train_idx, val_idx)

        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
        args.warmup_steps = args.total_steps//10

        print(args)
        model_dir = os.path.join(args.model_dir, args.model_name)
        os.makedirs(model_dir, exist_ok = True)
        json.dump(
            vars(args),
            open(f"{model_dir}/exp_config_{fold}.json", "w"),
            indent=2,
            ensure_ascii=False
            )

        print(f"\n{model_dir}/exp_config_{fold}.json is saved!\n")

        model = get_model(args)
        if args.use_finetune:
            load_state = torch.load(args.trained_model)
            model.load_state_dict(load_state['state_dict'], strict=True)
            print(f"{args.trained_model} is loaded!")

        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        best_auc = -1
        best_acc = -1
        early_stopping_couter = 0
        ### train함수는 한번 학습하는 함수임. 그래서 epoch수만큼 train해야함
        ### train_loader는 torch.utils.data.DataLoader 사용한건데 for문으로 하나씩 부르면 batch데이터가 나옴
        for epoch in range(args.n_epochs):
        
            print(f"Start Training: Epoch {epoch}")
        
            ### TRAIN
            train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)
            ### VALID
            auc, acc, val_loss = validate(val_loader, model, args)

            ### TODO: model save or early stopping
            if args.use_wandb:
                wandb.log({"train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc,
                       "val_loss": val_loss, "valid_auc": auc, "valid_acc": acc})

            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다. 
                model_to_save = model.module if hasattr(model, 'module') else model
                model_name = 'model_epoch' + str(epoch) + "_fold" + str(fold) + "_auc" + str(round(auc,3)) + "_acc"+ str(round(acc,3)) +  ".pt"
                save_checkpoint(
                    {'epoch': epoch, 
                        'state_dict': model_to_save.state_dict(),
                        'kfold': fold,
                        "train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc,
                        "val_loss": val_loss, "valid_auc": auc, "valid_acc": acc
                        },
                    model_dir, model_name,
                    )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                     print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                     break
            # scheduler
            if args.scheduler == 'plateau':
                scheduler.step(best_auc)
            else:
                scheduler.step()
        total_auc += best_auc
        total_acc += best_acc
    print("="*50)
    print(f"Total AUC: {total_auc / 5}")
    print(f"Total ACC: {total_acc / 5}")
    print("="*50)



def train(train_loader, model, optimizer, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []

    ### batch별로 학습시키고 loss 업데이트
    for step, (data, targets) in enumerate(train_loader):
        preds = model(data)
        targets = targets.type(torch.LongTensor)
        ### 밑에 구현돼있음..
        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        # print("preds shape", preds.shape)
        # print("target shape", targets.shape)
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    '''
    print(total_preds)
    total_preds = F.softmax(torch.from_numpy(total_preds), dim=1)
    print(total_preds)
    '''
    total_targets = np.concatenate(total_targets)
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)

    print(f'TRAIN AUC : {auc} ACC : {acc}')

    return auc, acc, loss_avg

### loss구하는거랑 다 train함수랑 똑같은데 update_params(loss, model, optimizer, args)만 빠짐
def validate(val_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    losses = []
    for step, (data, targets) in enumerate(val_loader):
        preds = model(data)
        targets = targets.type(torch.LongTensor)
        loss = compute_loss(preds, targets)

    
        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Valid AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    
    print(f"Valid Loss: {str(loss_avg)}")
    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, loss_avg

def inference(args, test_data):
    model = load_model(args)
    model.eval ### 꼭 해줘야함 중요
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, (data, targets) in enumerate(test_loader):
        preds = model(data)

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    write_path = os.path.join(args.output_dir, (args.model_name + "_epoch" + str(args.model_epoch) + ".csv"))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))

def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'audio_classifier': model = AudioClassifier(args)
 
    else:
        print("Invalid model!")
        exit()

    model.to(args.device)
    return model


### train안에서 [loss = compute_loss(preds, targets)] 이 코드로 사용됨.
### get_criterion 내 task에 맞춰서 좀 바꿀 필요 있음(너무 단순)
### 여러 값들의 loss값을 구해서 이를 평균 낸 게 compute_loss함수
def compute_loss(preds, targets):
    loss = get_criterion(preds, targets)
    loss = torch.mean(loss)
    return loss

### train안에서 [update_params(loss, model, optimizer, args)]이렇게 사용됨
### compute_loss다음에 사용됨
### 얘 중요!!!!!!!!!!!
def update_params(loss, model, optimizer, args):
    loss.backward() ### pytorch라이브러리 loss라서 backward가 바로 가능한가
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()

def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))

def load_model(args):
    model_dir = os.path.join(args.model_dir, args.model_name)
    model_path = os.path.join(model_dir, ('model_epoch' + str(args.model_epoch) + "_" + str(args.model_fold) + "fold" + ".pt"))
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model= get_model(args)
    model.load_state_dict(load_state['state_dict'], strict = True)
    print("Loading Model from:", model_path, "...Finished.")
    return model
'''
def get_target(datas):
    targets = []
    for data in datas:
        targets.append(data[-1][-1])

    return np.array(targets)

def update_train_data(pseudo_labels, train_data, test_data):
    # pseudo 라벨이 담길 test 데이터 복사본
    pseudo_test_data = copy.deepcopy(test_data)
    
    # pseudo label 테스트 데이터 update
    for p_test_data, pseudo_label in zip(pseudo_test_data, pseudo_labels):
        p_test_data[-1][-1] = pseudo_label

    # train data 업데이트
    # pseudo_train_data = np.concatenate((train_data, pseudo_test_data))
    ### 음??? 왜 test데이터만 사용했지???? 왜 concat안함???
    pseudo_train_data = pseudo_test_data
    print("pseudo_trian is ready!")

    return pseudo_train_data
'''