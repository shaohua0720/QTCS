import os
import torch
import datetime

class Config:
    def __init__(self, ratio=0.1, device="cuda:0"):
        self.ratio = ratio
        self.epochs = 200
        self.name = r'QCSMIMO'
        self.comments = r'Quantized Compressed Sensing for CSI feedback'
        self.batch_size = 25
        self.train_data = r'E:\\Code_shaohua\\datasets\\qdg_umi5g_7p68\\train.mat'
        self.val_data = r'E:\\Code_shaohua\\datasets\\qdg_umi5g_7p68\\val.mat'
        self.test_data = r'E:\\Code_shaohua\\datasets\\qdg_umi5g_7p68\\test.mat'
        self.DDP = False # Distributed Data Parallel enabled/Disable
        now = datetime.datetime.now()
        self.start_time = now.strftime("%Y-%m-%d %H-%M-%S")
        self.results = os.path.join(r'results', self.name)
        # self.channel = 1

        self.block_size = 96
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Paths
        # self.results = "./results"
        self.log = os.path.join(self.results, str(int(self.ratio * 100)), "log.txt")

        self.folder = os.path.join(self.results, str(int(self.ratio * 100)), "models")
        self.model = os.path.join(self.folder, "model.pth")
        self.optimizer = os.path.join(self.folder, "optimizer.pth")
        self.info = os.path.join(self.folder, "info.pth")

    def check(self):
        if not os.path.isdir(self.results):
            os.mkdir(self.results)

        sub_path = os.path.join(self.results, str(int(self.ratio * 100)))
        if not os.path.isdir(sub_path):
            os.mkdir(sub_path)
            print("Mkdir: " + sub_path)

        models_path = os.path.join(sub_path, "models")
        if not os.path.isdir(models_path):
            os.mkdir(models_path)
            print("Mkdir: " + models_path)

    def show(self):
        print("\n=> Your configs are:")
        print("=" * 70)
        for item in self.__dict__:
            print("{:<20s}".format(item + ":") + "{:<30s}".format(str(self.__dict__[item])))
            print("-" * 70)
        print("\n")


if __name__ == "__main__":
    config = Config()
    config.check()
    config.show()
