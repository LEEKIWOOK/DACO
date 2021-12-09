import os
from scipy.stats import spearmanr, pearsonr
import numpy as np

import torch
import torch.nn as nn

from utils import *
from modeling.utils import *


class Train:
    def __init__(self, Runner):

        self.framework = Runner.framework
        self.optimizers = Runner.optimizers
        self.scheduler = Runner.scheduler
        self.earlystop = Runner.earlystop

        self.criterion_reg = nn.MSELoss(reduction="mean")
        self.criterion_cls = nn.CrossEntropyLoss()
        self.EPOCH = Runner.EPOCH
        self.batch_size = Runner.batch_size

        self.train_target_iter = ForeverDataIterator(Runner.train_loader)
        self.val_target_iter = ForeverDataIterator(Runner.val_loader)
        self.test_target_iter = ForeverDataIterator(Runner.test_loader)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run_step(self, logger):

        best_model = os.path.join(logger.get_checkpoint_path("latest"))
        torch.save(self.framework.state_dict(), os.path.join(best_model + "_net.pth"))
        plot_fig = logger.get_image_path("latest.png")
        # early stopping patience; how long to wait after last time validation loss improved.
        early_stopping = EarlyStopping(
            patience=self.earlystop, verbose=True, path=best_model
        )
        avg_train_losses = []
        avg_valid_losses = []

        for epoch in range(self.EPOCH):

            train_losses = self.train(epoch)
            valid_losses = self.validate(epoch)
            self.scheduler.step()

            avg_train_losses.append(np.mean(train_losses))
            avg_valid_losses.append(np.mean(valid_losses))

            early_stopping(np.median(avg_valid_losses), self.framework)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        loss_plot(
            train_loss=avg_train_losses, valid_loss=avg_valid_losses, file=plot_fig
        )

        # load the last checkpoint with the best model
        self.framework.load_state_dict(torch.load(best_model + "_net.pth"))
        corrs, corrp = self.test()
        print(f"Spearman Correlation.\t{corrs}")
        print(f"Pearson Correlation.\t{corrp}")

    def to_cls(seklf, x):
        ret = x
        ret[np.where(x>=0.75)] = 1
        ret[np.where(x<0.25)] = 0
        ret[np.where((x>=0.25) & (x < 0.75))] = 2
        ret = ret.to(torch.long)
        return ret

    def train(self, epoch):

        train_losses = list()
        eval = {"predicted_value": list(), "real_value": list()}
        self.framework.train()
        for i in range(len(self.train_target_iter)):
            self.optimizers.zero_grad()

            E, y = next(self.train_target_iter)
            E = E.to(self.device)
            y = y.to(self.device)    

            oreg = self.framework(E)
            loss = self.criterion_reg(oreg, y)
            loss.backward()

            self.optimizers.step()

            eval["predicted_value"] += oreg.cpu().detach().numpy().tolist()
            eval["real_value"] += y.cpu().detach().numpy().tolist()
            train_losses.append(loss.item())

            if i == 0:
                print(
                    f"Training step : Epoch : [{epoch}/{self.EPOCH}] [{i}/{len(self.train_target_iter)}], Loss_reg : {loss}"
                )

        corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
        print(f"Training Spearman correlation = {corrs}")
        return train_losses

    def validate(self, epoch):
        valid_losses = []
        eval = {"predicted_value": list(), "real_value": list()}
        self.framework.eval()
        with torch.no_grad():
            for i in range(len(self.val_target_iter)):

                E, y = next(self.val_target_iter)
                E = E.to(self.device)
                y = y.to(self.device)

                oreg = self.framework(E)
                loss = self.criterion_reg(oreg, y)
                
                eval["predicted_value"] += oreg.cpu().detach().numpy().tolist()
                eval["real_value"] += y.cpu().detach().numpy().tolist()
                valid_losses.append(loss.item())

                if i == 0:
                    print(
                        f"Validation step : Epoch : [{epoch}/{self.EPOCH}] [{i}/{len(self.train_target_iter)}], Loss_reg : {loss}"
                    )
        corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
        print(f"Validation Spearman correlation = {corrs}")
        return valid_losses

    def test(self):

        eval = {"predicted_value": list(), "real_value": list()}
        self.framework.eval()
        with torch.no_grad():
            for i in range(len(self.test_target_iter)):
                
                E, y = next(self.test_target_iter)

                E = E.to(self.device)
                y = y.to(self.device)

                oreg = self.framework(E)

                eval["predicted_value"] += oreg.cpu().detach().numpy().tolist()
                eval["real_value"] += y.cpu().detach().numpy().tolist()

        corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
        corrp = pearsonr(eval["real_value"], eval["predicted_value"])[0]
        return corrs, corrp
