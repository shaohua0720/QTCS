from .config import Config
import argparse
class Option:
    def __init__(self) -> None:
        self.config = Config()

    def parse(self):
        parser = argparse.ArgumentParser(description='QCSI Compression')  
        parser.add_argument('--batch_size', type = int, metavar = 'B',
                            help = 'Batch size of the training')
        parser.add_argument('--lr', type = float, metavar = 'L',
                            help = 'Learning rate')
        parser.add_argument('--epochs', type = int, default = 500, metavar = 'E',
                            help = 'numer of Epochs')
        parser.add_argument('--device', type = str,  metavar = 'DV',
                            help = 'GPU')
        parser.add_argument('--DDP', type = int, metavar = 'DV',
                            help = 'Distributed Data parallels enabled')
        parser.add_argument('--train_data', type = str, metavar = 'DV',
                            help = 'train data file path')
        parser.add_argument('--val_data', type = str, metavar = 'DV',
                            help = 'val data file path')
        parser.add_argument('--test_data', type = str, metavar = 'DV',
                            help = 'test data file path')
        parser.add_argument('--ratio', type = str, metavar = 'DV',
                            help = 'CS ratio parameter')
        parser.add_argument('--save_every', type = int, metavar = 'DV',
                            help = 'save model snapshot')
        args = parser.parse_args()

        """Update the default Config with the command params"""
        self.config.device = args.device or self.config.device
        self.config.epochs = args.epochs or self.config.epochs
        self.config.batch_size = args.batch_size or self.config.batch_size
        self.config.lr = args.lr or self.config.lr
        self.config.DDP = args.DDP or self.config.DDP
        self.config.train_data = args.train_data or self.config.train_data
        self.config.val_data = args.val_data or self.config.val_data
        self.config.test_data = args.test_data or self.config.test_data
        self.config.ratio = args.ratio or self.config.ratio

        # self.config.check()
        # self.config.show()
        return self.config