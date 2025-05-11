from typing import Iterable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torch.utils.data
from utils import CfgNode, get_linear_schedule_with_warmup_hold, get_confuse_matrix
import logging
from datetime import datetime
from model_dataset import MySampler, modulation_list, SNR_START, SNR_STOP
import os

mysampler_train = MySampler(type='train')
mysampler_valid = MySampler(type='validation')
mysampler_test = MySampler(type='test')


class Trainer():
    @staticmethod
    def get_default_config():
        '''return default training config'''
        config = CfgNode()
        config.learning_rate = 1e-3
        config.num_workers = 8
        config.batch_size = 512
        config.device = 'auto'
        config.betas = (0.9, 0.999)
        config.num_epoch = 50
        return config

    def __init__(self,
                 config,
                 model: nn.Module,
                 train_dataset: torch.utils.data.Dataset,
                 valid_dataset: torch.utils.data.Dataset):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename="./train.log", level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        torch.set_float32_matmul_precision('high')
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.current_time = datetime.now().strftime("%y-%m-%d-%H-%M")
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        # speed up model forward processing
        self.model = torch.compile(self.model)
        self.loss_fn1 = nn.CrossEntropyLoss()
        self.loss_fn2 = nn.MSELoss()
        self.logger.info("==================================================")
        self.logger.critical(
            f"model is {model.__class__.__name__},SNR range:{SNR_START}, {SNR_STOP}")
        self.logger.critical(f"running in device: {self.device}")

    def run(self, checkpoint_path=None):
        '''checkpoint_path is the path of the checkpoint to load'''
        self.logger.critical(f"start training")
        self.current_time = datetime.now().strftime("%y-%m-%d-%H-%M")
        config = self.config
        self.model.train()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=1e-2)
        self.scheduler = get_linear_schedule_with_warmup_hold(
            self.optimizer, total_steps=config.num_epoch)
        start_epoch = 0
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.current_time = checkpoint['current_time']
            self.logger.critical(
                f"load checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")

        os.makedirs(f"model_parameter/{self.current_time}", exist_ok=True)
        # here we use sampler to get data from dataset
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, sampler=mysampler_train, pin_memory=True, num_workers=config.num_workers, persistent_workers=True)
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=self.config.batch_size, sampler=mysampler_valid, pin_memory=True, num_workers=self.config.num_workers, persistent_workers=True)

        for epoch in range(start_epoch, config.num_epoch):
            loss_epoch = self.do_train(self.train_loader)
            self.scheduler.step()

            self.logger.info(f"Train: epoch {epoch} loss: {loss_epoch}")
            if epoch % 5 == 0 or epoch == config.num_epoch - 1:
                filename = f'checkpoint_{self.current_time}_epoch_{epoch}_loss_{loss_epoch:.2f}.pt'
                model_path = f"model_parameter/{self.current_time}/{filename}"
                self.save_model(epoch, loss_epoch, path=model_path)
            self.validate()

        self.logger.info(f"finish training , model saved")

    def do_train(self,
                 loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.model.train()
        loss_item_content = 0
        for data_X, snr, modulation_cls in loader:
            data_X, modulation_cls, snr = (
                t.to(self.device) for t in (data_X, modulation_cls, snr))
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                cls_Y1_hat = self.model(data_X)
                loss = self.loss_fn1(cls_Y1_hat, modulation_cls)
            batch_size = data_X.shape[0]
            loss_item_content += round((loss.item() * batch_size), 0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_item_content

    def save_model(self, epoch, loss, path="./model_param.pt"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'loss': loss,
            'epoch': epoch,
            'current_time': self.current_time,
        }, path)
        self.logger.info(f"save model successfully to {path}")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        loss_epoch = self.do_validate(self.valid_loader)
        self.logger.critical(f"Validate: validate loss: {loss_epoch:.2f} ")

    @torch.no_grad()
    def do_validate(self,
                    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        loss_valid = 0
        for data_X, snr, modulation_cls in data_loader:
            data_X, modulation_cls, snr = (
                t.to(self.device) for t in (data_X, modulation_cls, snr))
            cls_Y1_hat = self.model(data_X)
            # cls_Y_hat.shape = (batch_size, num_class), modulatisn_cls.shape = (batch_size, )
            loss = self.loss_fn1(cls_Y1_hat, modulation_cls)
            batch_size = data_X.shape[0]
            loss_valid += (loss.item() * batch_size)
        return loss_valid

    def test(self, model_path: str, dataset_test, ):
        '''model_path: the model checkpoint path which you want to test
        '''
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.logger.info(f"In test,load model successfully from {model_path}")
        self.y_true, self.y_pred = {str(
            key): [] for key in range(SNR_START, SNR_STOP+1, 2)}, {str(key): [] for key in range(SNR_START, SNR_STOP+1, 2)}

        data_loader = DataLoader(
            dataset_test, batch_size=self.config.batch_size, sampler=mysampler_test, num_workers=2)

        loss_epoch = self.do_test(data_loader)

        get_confuse_matrix(model_path.split('/')[-1], self.y_true, self.y_pred,
                           timestamp=self.current_time, labels=modulation_list, normalize='true')
        self.logger.critical(f"Test: test loss: {loss_epoch:.2f} ")
        self.logger.info(
            f'Testing finished. Result saved in /result_data/confusionMatrix_{self.current_time}.pkl')

    def do_test(self,
                data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        loss = 0
        for data_X, snr, modulation_cls in data_loader:
            data_X = data_X.to(self.device)
            modulation_cls = modulation_cls.to(self.device)
            cls_Y1_hat = self.model(data_X)
            snr_str = [str(x.item()) for x in snr]
            modulation_cls_list = modulation_cls.tolist()
            cls_Y1_hat_list = cls_Y1_hat.argmax(dim=1, keepdim=False).tolist()
            for snr_now, y_true, y_pred in zip(snr_str, modulation_cls_list, cls_Y1_hat_list):
                self.y_true[snr_now].append(y_true)
                self.y_pred[snr_now].append(y_pred)
            loss += (self.loss_fn1(cls_Y1_hat, modulation_cls).item()
                     * data_X.shape[0])
        return loss
