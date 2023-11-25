import os
import torch
import torchaudio
import numpy as np
import math, random
from tqdm import tqdm
import pandas as pd
from torchaudio import transforms
from IPython.display import Audio
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset



class Preprocess:
    def __init__(self, args):
        self.args = args
        self.data = None

    def get_data(self):
        return self.data

    ## label을 ~_classes.npy로 저장하는것
    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        print("*"*100)
        print(encoder.classes_)        
        print("*"*100)
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, main_df, sub_df=None):
        os.makedirs(self.args.asset_dir, exist_ok=True)
        all_df = pd.concat([main_df, sub_df])
        le = LabelEncoder()
        all_df['label'] = all_df['label'].astype(str)
        trans = le.fit_transform(all_df['label'])
        all_df['label'] = trans
        self.__save_labels(le, self.args.type)
        return all_df


    ## data불러오는 함수(main_df만드는 함수)
    def load_data_from_file(self, main_file_name, sub_file_name=None):
        csv_file_path = os.path.join(self.args.data_dir, main_file_name)
        main_df = pd.read_csv(csv_file_path)
        sub_df = None
        if sub_file_name:
            csv_file_path = os.path.join(self.args.data_dir, sub_file_name)
            sub_df = pd.read_csv(csv_file_path)
        main_df = self.__preprocessing(main_df, sub_df)
        return main_df


    def load_data(self, train_file_name, sub_file_name=None):   
        self.data = self.load_data_from_file(train_file_name, sub_file_name)
        print("data is loaded!")
        print()

class AudioUtil():
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    @staticmethod
    def standardize(aud):
        sig, sr = aud
        mean = sig.mean(axis = 1)
        std = sig.std(axis = 1)
        standardized_sig = (sig-mean) /std
        return (standardized_sig, sr)

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            return aud # Nothing to do

        if (new_channel == 1):
            resig = sig[:1,:]
        else:
            resig = torch.cat([sig, sig])
        return (resig, sr)
     
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            return aud

        new_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))
    
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            sig = sig[:, :max_len]

        elif (sig_len < max_len):
            ## 앞과 뒤에 랜덤한 길이만큼 0 패딩 해줌
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

     ##################### SPectrogram transforms #####################
    @staticmethod
    def spectrogram(aud, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
        spec = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_len)(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    @staticmethod
    def mel_scale(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
        spec = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_len)(sig)
        spec = transforms.MelScale(sample_rate=sr,n_mels=n_mels, n_stft=n_fft//2 + 1)(spec)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    @staticmethod
    def mel_spec(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
        spec = transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    @staticmethod
    def mfcc(aud, n_mfcc=64, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
        spec = transforms.MFCC(
            sample_rate=sr, 
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop_len, "n_mels": n_mels}
            )(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    @staticmethod
    def lfcc(aud, n_lfcc=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
        spec = transforms.LFCC(
            sample_rate=sr, 
            n_lfcc=n_lfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop_len}
            )(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
     ##################### SPectrogram transforms #####################

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value) 
        return aug_spec


class UpsingDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, idx):
        audio_file = self.args.data_path + self.data.loc[idx, 'song_name'] + ".wav" 
        label = self.data.loc[idx, 'label']

        ## aud == (sig, sr)
        ## sig.shape == num_channels, num_frames
        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.resample(aud, self.args.sr)
        aud = AudioUtil.rechannel(aud, self.args.channel)
        start, end = map(float, self.data.loc[idx, 'sec'][1:-1].split(','))
        sig, sr = aud
        sig = sig[:,int(start*self.args.sr):int(end*self.args.sr)]
        aud = (sig, sr)
        print()
        aud = AudioUtil.standardize(aud)

        aud = AudioUtil.pad_trunc(aud, self.args.duration)


        if self.args.spec_type:
            if self.args.spec_type == "Spectrogram": 
                aud = AudioUtil.spectrogram(aud, n_fft=self.args.n_fft, hop_len=self.args.hop_len)
            elif self.args.spec_type == "MelScale": 
                aud = AudioUtil.mel_scale(aud, n_mels=self.args.n_mels, n_fft=self.args.n_fft, hop_len=self.args.hop_len)
            elif self.args.spec_type == "Mel":
                aud = AudioUtil.mel_spec(aud, n_mels=self.args.n_mels, n_fft=self.args.n_fft, hop_len=self.args.hop_len)
            elif self.args.spec_type == "MFCC": 
                aud = AudioUtil.mfcc(aud, n_mfcc=self.args.n_mfcc, n_mels=self.args.n_mels, n_fft=self.args.n_fft, hop_len=self.args.hop_len)
            elif self.args.spec_type == "LFCC": 
                 aud = AudioUtil.lfcc(aud, n_lfcc=self.args.n_lfcc, n_fft=self.args.n_fft, hop_len=self.args.hop_len)
        else: self.args.spec_aug = False

        if self.args.spec_aug:
            aud = AudioUtil.spectro_augment(aud, max_mask_pct=self.args.max_mask_pct, n_freq_masks=self.args.n_freq_masks, n_time_masks=self.args.n_time_masks)

        return aud, label

    def __len__(self):
        return len(self.data)


def get_loaders(args, data, train_idx, val_idx):
    train_loader,val_loader = None, None

    if data is not None:
        dataset = UpsingDataset(data, args)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False)
    return train_loader, val_loader

def get_all_loader(args, data):
    train_loader = None

    if data is not None:
        dataset = UpsingDataset(data, args)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader