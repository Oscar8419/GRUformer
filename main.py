from model_dataset import Signal_Dataset
from train import Trainer
from model import GRUformer
dataset_train = Signal_Dataset()
dataset_valid = Signal_Dataset()
dataset_test = Signal_Dataset()

model_checkpoint_path = None
model = GRUformer()

config = Trainer.get_default_config()
config.batch_size = 1024
config.num_epoch = 50
config.num_workers = 9
config.learning_rate = 1e-3
trainer = Trainer(config, model, dataset_train, dataset_valid, )


def main():
    trainer.run(checkpoint_path=model_checkpoint_path, )
    # trainer.test(model_path, dataset_test, ) #when test, uncomment this line and modify model_path ,comment the line above


if __name__ == '__main__':
    main()
