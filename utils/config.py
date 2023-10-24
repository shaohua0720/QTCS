import os
import torch
import datetime
import pickle

class Config:
    def __init__(self, ratio=0.1, device="cuda:0"):
        self.ratio = ratio
        self.epochs = 150
        self.name = r'QCSMIMO'
        self.comments = r'Quantized Compressed Sensing for CSI feedback'
        self.batch_size = 200
        self.train_data = r'E:\\Code_shaohua\\datasets\\qdg_umi5g_3p84\\val.h5'
        self.val_data = r'E:\\Code_shaohua\\datasets\\qdg_umi5g_3p84\\test.h5'
        self.test_data = r'E:\\Code_shaohua\\datasets\\qdg_umi5g_3p84\\test.h5'
        self.DDP = False # Distributed Data Parallel enabled/Disable
        now = datetime.datetime.now()
        self.start_time = now.strftime("%Y-%m-%d %H-%M-%S")
        self.results = os.path.join(r'results', self.name)
        self.lr = 1e-4 # learning rate

        self.block_size = 32
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_every = 5
        self.resume = True # resume training

        self.n_embed = 8 # number of embedding -> log2(self.n_embed) bits quantization
        self.embed_d = 1 # embedding dimenstion

        # Paths
        # self.results = "./results"
        self.log = os.path.join(self.results, str(int(self.ratio * 100)), "log.txt")

        self.folder = os.path.join(self.results, str(int(self.ratio * 100)), "models")
        self.model = os.path.join(self.folder, "model.pth")
        self.snapshot_path = os.path.join(self.folder, "model_snp.pth")
        self.optimizer = os.path.join(self.folder, "optimizer.pth")
        self.info = os.path.join(self.folder, "info.pth")

    def check(self):
        if not os.path.isdir(self.results):
            os.makedirs(self.results)

        sub_path = os.path.join(self.results, str(int(self.ratio * 100)))
        if not os.path.isdir(sub_path):
            os.makedirs(sub_path)
            print("Mkdir: " + sub_path)

        models_path = os.path.join(sub_path, "models")
        if not os.path.isdir(models_path):
            os.makedirs(models_path)
            print("Mkdir: " + models_path)
        
        with open(os.path.join(models_path, 'config.txt'), 'w+') as f:
            for item in self.__dict__:
                print("{:<20s}".format(item + ":") + "{:<30s}".format(str(self.__dict__[item])),file= f)
        with open(os.path.join(models_path, 'config.pkl'), 'wb') as f:
            pickle.dump(self, f)

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
