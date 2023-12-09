import os
import argparse

def parse_args(mode='train'):
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--seed', default=42, type=int, help='seed')
    
    parser.add_argument('--device', default='cuda', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='C:/Users/Ahn/projects/upsing/data/', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='C:/Users/Ahn/projects/upsing/asset/', type=str, help='data directory')
    parser.add_argument('--data_path', default='C:/Users/Ahn/projects/upsing/copy_song_mr_removed/', type=str, help='copy song data path')

    parser.add_argument('--train_file_name', default='voc_train.csv', type=str, help='train file name')
    parser.add_argument('--sub_file_name', default=None, type=str, help='teq sub file name')
    parser.add_argument('--test_file_name', default='voc_test.csv', type=str, help='test file name')
    
    parser.add_argument('--model_dir', default='C:/Users/Ahn/projects/upsing/model/', type=str, help='model directory')

    parser.add_argument('--model_name', default='AudioClassifier', type=str, help='model folder name')
    parser.add_argument('--model_epoch', default=0, type=int, help='epoch')

    parser.add_argument('--output_dir', default='C:/Users/Ahn/projects/upsing/output/', type=str, help='output directory')
    parser.add_argument('--output_file', default='output', type=str, help='output directory')
   
    
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')

   
    # ÈÆ·Ã
    parser.add_argument('--n_epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    parser.add_argument('--patience', default=5, type=int, help='for early stopping')

    
    # Pseudo Labeling
    # parser.add_argument('--use_pseudo', default=False, type=bool, help='Using Pseudo labeling')
    # parser.add_argument('--pseudo_label_file', default='', type=str, help='file path for pseudo labeling')

    # Finetuning
    parser.add_argument('--use_finetune', default=False, type=bool, help='Using Fine Tuning')
    parser.add_argument('--trained_model', default='C:/Users/Ahn/projects/upsing/model/model_epoch0.pt', type=str, help='pretrained model path')

    # log
    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')

    # wandb
    parser.add_argument('--use_wandb', default=False, type=bool, help='if you want to use wandb')

    ### Áß¿ä ###
    parser.add_argument('--model', default='audio_classifier', type=str, help='model type')
    parser.add_argument('--optimizer', default='adamW', type=str, help='optimizer type')
    parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler type')
    
    ### upsing ###
    parser.add_argument('--type', default='voc', type=str, help='model training data type(voc or teq)')
    parser.add_argument('-sr', default=44100, type=int,  help='sample rate')
    parser.add_argument('--channel', default=1, type=int, help='audio channel')
    parser.add_argument('--duration', default=2000, type=int, help='audio duration')
    parser.add_argument('--spec_type', default='Mel', type=str, help='spectrum type')
    parser.add_argument('--spec_aug', default=False, type=bool, help='spectrum augmrntation ')
    parser.add_argument('--n_fft', default=1024, type=int, help='n_fft')
    parser.add_argument('--hop_len', default=None, type=int, help='hop_len')
    parser.add_argument('--n_mels', default=64, type=int, help='n_mels')
    parser.add_argument('--n_mfcc', default=64, type=int, help='n_mfcc')
    parser.add_argument('--n_lfcc', default=64, type=int, help='n_lfcc')
    parser.add_argument('--max_mask_pct', default=0.1, type=float, help='max_mask_pct')
    parser.add_argument('--n_freq_masks', default=1, type=int, help='n_freq_masks')
    parser.add_argument('--n_time_masks', default=1, type=int, help='n_time_masks')
    parser.add_argument('--model_fold', default=5, type=int, help='fold number')
    parser.add_argument('--k', default=5, type=int, help='kfold k')
    parser.add_argument('--wandb_name', default="voc", type=str, help="wandb run name")

    parser.add_argument('--train_all', default=False, type=bool, help="no validation dataset")

    args = parser.parse_args()

    return args