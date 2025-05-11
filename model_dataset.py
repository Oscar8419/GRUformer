from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
import h5py
import torch
import numpy as np
from utils import set_seed

SNR_START = -10
SNR_STOP = 20
modulation_dict = {'OOK': 0, '4ASK': 1, '8ASK': 2, 'BPSK': 3, 'QPSK': 4, '8PSK': 5, '16PSK': 6, '32PSK': 7, '16APSK': 8, '32APSK': 9, '64APSK': 10, '128APSK': 11,
                   '16QAM': 12, '32QAM': 13, '64QAM': 14, '128QAM': 15, '256QAM': 16, 'AM-SSB-WC': 17, 'AM-SSB-SC': 18, 'AM-DSB-WC': 19, 'AM-DSB-SC': 20, 'FM': 21, 'GMSK': 22, 'OQPSK': 23, }
modulation_list = list(modulation_dict.keys())


class Signal_Dataset(Dataset):
    def __init__(self,
                 path_data: str = "/root/autodl-tmp/GOLD_XYZ_OSC.0001_1024.hdf5"):
        super().__init__()
        self.path_data = path_data
        hdf5_file = h5py.File(path_data,  'r')
        self.data = hdf5_file['X']
        self.modulation_onehot = hdf5_file['Y']
        self.SNR = hdf5_file['Z']

    def __getitem__(self, idx):
        idx_data = idx
        self.snr_target = self.SNR[idx_data].item()
        # get the index of modulation
        self.idx_modulation = self.modulation_onehot[idx_data].argmax().item()
        data_target = torch.from_numpy(self.data[idx_data])
        # normalize signal power
        data_target /= torch.sqrt(self.sig_power(data_target))
        # self.snr_target = (self.snr_target + 20) // 2  # get the index of snr
        return data_target, torch.tensor(self.snr_target, dtype=torch.long), torch.tensor(self.idx_modulation, dtype=torch.long)

    @staticmethod
    def sig_power(signal_data):
        iq_signal = signal_data[:, 0] + 1j*signal_data[:, 1]
        signal_average_power = torch.mean(torch.abs(iq_signal)**2)
        return signal_average_power


class MySampler(Sampler):
    '''here we create a sampler to sample data from different SNR and modulation
    '''

    def __init__(self, data_source=None, type='train',
                 snr_start: int = SNR_START, snr_stop: int = SNR_STOP):
        '''type should in ['train', 'validation', 'test'], showing the type of dataset
            snr_start: the start SNR of the dataset
            snr_stop: the stop SNR of the dataset
        '''
        super().__init__(data_source)
        set_seed(42)  # MUST fix random seed
        assert (type in ['train', 'validation', 'test'])
        self.num_mods = 24  # 24 Modulations
        self.mods = modulation_list  #
        self.frames_per_mod_snr = 4096
        # 4096 is divided into 3496/300/300   Train set/Validation set/Test set
        indices = np.random.permutation(4096)
        train_indices = indices[:3496]
        val_indices = indices[3496:3496+300]
        test_indices = indices[3496+300:]
        self.snrs = np.arange(-20, 31, 2)
        assert snr_start in self.snrs and snr_stop in self.snrs
        assert snr_start <= snr_stop
        # here we just calculate the index of snr_start and snr_stop
        self.snr_start_idx = (snr_start + 20) // 2
        self.snr_stop_idx = (snr_stop + 20) // 2
        self.num_snrs = self.snr_stop_idx - self.snr_start_idx + 1
        self.mod_snr_groups = []
        for mod in self.mods:
            mod_idx = modulation_dict[mod]
            for snr_idx in range(self.snr_start_idx, self.snr_stop_idx+1):
                start = mod_idx * (len(self.snrs) * self.frames_per_mod_snr) + \
                    (snr_idx * self.frames_per_mod_snr)
                if type == 'train':
                    idx = list(start + train_indices)
                elif type == 'validation':
                    idx = list(start + val_indices)
                elif type == 'test':
                    idx = list(start + test_indices)
                self.mod_snr_groups.append(idx)
        self.total_samples = sum(len(g) for g in self.mod_snr_groups)

    def __iter__(self):
        np.random.shuffle(self.mod_snr_groups)
        all_indices = [idx for group in self.mod_snr_groups for idx in group]
        np.random.shuffle(all_indices)
        return iter(all_indices)

    def __len__(self):
        return self.total_samples
