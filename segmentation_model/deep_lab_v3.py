import torch
import numpy as np

from torchvision import models
from torch import optim
from torch.nn import Conv2d
from torch.nn.functional import softmax
from segmentation_model.soft_iou_loss import SoftIOULoss
from tqdm import tqdm
from utils.utils import plot_hist, get_metrics, save_predictions
#from prettytable import PrettyTable


def count_parameters(model):
    #table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        #table.add_row([name, param])
        total_params += param
    #print(table)
    #print(f"Total Trainable Params: {total_params}")

    return total_params


class DeepLabV3:
    def __init__(self, train_loader=None, test_loader=None,  model_path=None, seed=0, output_channels=1,
                 backbone="RESNET101", optimizer="SGD", lr=0.001):
        torch.manual_seed(seed)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_path = model_path

        if backbone not in ["RESNET50", "RESNET101"]:
            raise ValueError(f"{backbone} is not supported")
        else:
            self.backbone = backbone
            self.output_channels = output_channels

        self.model = self.create_model()

        if optimizer == "SGD":
            self.optimizer = optim.SGD(params=self.model.parameters(), lr=lr)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        elif optimizer == "LBFGS":
            self.optimizer = optim.LBFGS(params=self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"{optimizer} is not supported.")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = SoftIOULoss(device=self.device)

    def create_model(self):
        model = None
        if self.backbone == "RESNET50":
            model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        elif self.backbone == "RESNET101":
            model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)

        model.classifier[4] = Conv2d(256, self.output_channels, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = Conv2d(256, self.output_channels, kernel_size=(1, 1), stride=(1, 1))

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

        for param in model.aux_classifier.parameters():
            param.requires_grad = True

        count_parameters(model)

        return model

    def train_epoch(self):
        print("Training...")
        self.model.to(self.device)
        self.model.train()

        all_y_true, all_y_hat = list(), list()
        epoch_loss = 0

        for sample in tqdm(self.train_loader):
            x = sample["image"].to(self.device)
            y = sample["mask"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)

            loss = self.criterion(output['out'], y)

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.data.item() / len(x)

            y_true = y.data.cpu().numpy()
            y_hat = np.nanargmax(softmax(output['out'].data.cpu(), dim=1).numpy(), axis=1)

            for y_true_, y_hat_ in zip(y_true, y_hat):
                all_y_true.append(y_true_)
                all_y_hat.append(y_hat_)

        epoch_acc, epoch_acc_cls, epoch_mean_iou, epoch_freq_weighted_avg_acc = get_metrics(y_true=all_y_true,
                                                                                            y_hat=all_y_hat,
                                                                                            n_class=self.output_channels)

        return epoch_loss, epoch_acc, epoch_acc_cls, epoch_mean_iou, epoch_freq_weighted_avg_acc

    def test_epoch(self):
        print("Validating...")
        self.model.to(self.device)
        self.model.eval()

        all_y_true, all_y_hat = list(), list()
        epoch_loss = 0

        for sample in tqdm(self.test_loader):
            x = sample["image"].to(self.device)
            y = sample["mask"].to(self.device)

            with torch.no_grad():
                output = self.model(x)

            loss = self.criterion(output['out'], y)
            epoch_loss += loss.data.item() / len(x)

            y_true = y.data.cpu().numpy()
            y_hat = np.nanargmax(softmax(output['out'].data.cpu(), dim=1).numpy(), axis=1)

            for y_true_, y_hat_ in zip(y_true, y_hat):
                all_y_true.append(y_true_)
                all_y_hat.append(y_hat_)

        epoch_acc, epoch_acc_cls, epoch_mean_iou, epoch_freq_weighted_avg_acc = get_metrics(y_true=all_y_true,
                                                                                            y_hat=all_y_hat,
                                                                                            n_class=self.output_channels)

        return epoch_loss, epoch_acc, epoch_acc_cls, epoch_mean_iou, epoch_freq_weighted_avg_acc

    def train(self, epochs=100):
        best_iou = 0

        train_loss_hist, train_iou_hist = list(), list()
        val_loss_hist, val_iou_hist = list(), list()

        for epoch in range(1, epochs + 1):
            print("-"*50)
            print(f"Epoch {epoch}/{epochs}:")

            train_loss, train_acc, train_acc_cls, train_mean_iou, train_freq_weighted_avg_acc = self.train_epoch()
            print(f"Train results: Loss: {train_loss} | Acc: {train_acc} | Mean IOU: {train_mean_iou} | Freq Weighted"
                  f" Avg Acc: {train_freq_weighted_avg_acc}")
            train_loss_hist.append(train_loss)
            train_iou_hist.append(train_mean_iou)

            val_loss, val_acc, val_acc_cls, val_mean_iou, val_freq_weighted_avg_acc = self.test_epoch()
            print(f"Val results: Loss: {val_loss} | Acc: {val_acc} | Mean IOU: {val_mean_iou} | Freq Weighted"
                  f" Avg Acc: {val_freq_weighted_avg_acc}")
            val_loss_hist.append(val_loss)
            val_iou_hist.append(val_mean_iou)

            if val_mean_iou > best_iou:
                best_iou = val_mean_iou
                state = self.model.state_dict()
                torch.save(state, self.model_path)

        plot_hist((train_loss_hist, train_iou_hist), labels=["train_loss", "train_mean_iou"])
        plot_hist((val_loss_hist, val_iou_hist), labels=["val_loss", "val_mean_iou"])

        return self.model.state_dict()

    def load_model(self):
        self.model = self.create_model()
        state = torch.load(self.model_path)
        self.model.load_state_dict(state)
        self.model.eval()

        return self.model.state_dict()

    def model_inference(self, loader, output_path):
        self.model.eval()
        self.model.to(self.device)

        all_y_true, all_y_hat = list(), list()
        loss_ = 0

        for sample in tqdm(loader):
            x = sample["image"].to(self.device)
            y = sample["mask"].to(self.device)
            tags = sample["tag"]

            with torch.no_grad():
                output = self.model(x)

            loss = self.criterion(output['out'], y)
            loss_ += loss.data.item() / len(x)

            x_batch = x.data.cpu()
            y_true = y.data.cpu().numpy()
            y_hat = np.nanargmax(softmax(output['out'].data.cpu(), dim=1).numpy(), axis=1)

            for x_, y_true_, y_hat_, tag in zip(x_batch, y_true, y_hat, tags):
                save_predictions(loader=loader, x=x_, y_hat=y_hat_, tag=tag, path=output_path)
                all_y_true.append(y_true_)
                all_y_hat.append(y_hat_)

        acc, acc_cls, mean_iou, freq_weighted_avg_acc = get_metrics(y_true=all_y_true, y_hat=all_y_hat,
                                                                    n_class=self.output_channels)

        print(f"Inference results: Loss: {loss_} | Acc: {acc} | Mean IOU: {mean_iou} | Freq Weighted Avg Acc: "
              f"{freq_weighted_avg_acc}")

        return acc, acc_cls, mean_iou, freq_weighted_avg_acc
