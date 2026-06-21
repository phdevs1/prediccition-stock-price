# -*-Encoding: utf-8 -*-
from data_load.data_loader import Dataset_Custom
from model.model import denoise_net, pred_net
from gluonts.torch.util import copy_parameters
from utils.tools import EarlyStopping, adjust_learning_rate
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader

import os
import warnings


warnings.filterwarnings("ignore")


class Exp_Model(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

        # denoise_net se entrena; pred_net se usa en inferencia (recibe los pesos via
        # copy_parameters). Se eliminaron self.gen_net y self.embedding del original:
        # eran codigo muerto (nunca se usaban en train/vali/test).
        self.denoise_net = denoise_net(args).to(self.device)
        self.diff_step = args.diff_steps
        self.pred_net = pred_net(args).to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self, flag):
        args = self.args
        Data = Dataset_Custom
        if flag == "test" or flag == "val":
            shuffle_flag = False
            # drop_last=True: Res12_Quadratic asume batches multiplo de 8 (view -1, 8192).
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            # En entrenamiento drop_last=True es obligatorio: torch.randint(..., (batch_size,))
            # usa el batch_size del arg, no el del batch real.
            drop_last = True
            batch_size = args.batch_size
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.sequence_length, args.prediction_length],
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader

    def _get_combined_data(self, flag, csv_files):
        args = self.args
        datasets = [
            Dataset_Custom(
                root_path=args.root_path,
                data_path=f,
                flag=flag,
                size=[args.sequence_length, args.prediction_length],
            )
            for f in csv_files
        ]
        combined = ConcatDataset(datasets)
        print(flag, len(combined))
        loader = DataLoader(
            combined,
            batch_size=args.batch_size,
            shuffle=(flag == "train"),
            num_workers=args.num_workers,
            drop_last=True,
        )
        return combined, loader

    def _select_optimizer(self):
        denoise_optim = optim.Adam(
            self.denoise_net.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        return denoise_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        copy_parameters(self.denoise_net, self.pred_net)
        self.pred_net.eval()
        total_mse = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y[..., -self.args.target_dim :].float().to(self.device)
                noisy_out, out, _ = self.pred_net(batch_x, batch_x_mark)
                mse = criterion(out.squeeze(1), batch_y)
                total_mse.append(mse.item())

        self.pred_net.train()
        total_mse = np.average(total_mse)
        return total_mse

    def train(self, setting, csv_files=None):
        if csv_files:
            train_data, train_loader = self._get_combined_data("train", csv_files)
            vali_data, vali_loader = self._get_combined_data("val", csv_files)
        else:
            train_data, train_loader = self._get_data(flag="train")
            vali_data, vali_loader = self._get_data(flag="val")
        train_steps = len(train_loader)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        denoise_optim = self._select_optimizer()
        criterion = self._select_criterion()
        kl_anneal_start = self.args.kl_anneal_start
        kl_anneal_end = self.args.kl_anneal_end

        for epoch in range(self.args.train_epochs):
            # KL annealing: peso constante durante toda la epoca (solo depende de epoch).
            if kl_anneal_end > kl_anneal_start:
                if epoch < kl_anneal_start:
                    kl_w = 0.0
                elif epoch < kl_anneal_end:
                    kl_w = self.args.kl_latent_weight * (epoch - kl_anneal_start) / (kl_anneal_end - kl_anneal_start)
                else:
                    kl_w = self.args.kl_latent_weight
            else:
                kl_w = self.args.kl_latent_weight

            mse = []
            kl = []
            dsm = []
            latent = []
            all_loss = []
            self.denoise_net.train()
            for i, (batch_x, batch_y, x_mark, _) in enumerate(train_loader):
                t = (
                    torch.randint(0, self.diff_step, (self.args.batch_size,))
                    .long()
                    .to(self.device)
                )
                batch_x = batch_x.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                batch_y = batch_y[..., -self.args.target_dim :].float().to(self.device)
                denoise_optim.zero_grad()
                output, y_noisy, dsm_loss = self.denoise_net(
                    batch_x, x_mark, batch_y, t
                )
                recon = output.log_prob(y_noisy)
                mse_loss = criterion(output.sample(), y_noisy)
                kl_loss = -torch.mean(torch.sum(recon, dim=[1, 2, 3]))
                latent_kl = torch.mean(
                    self.denoise_net.diffusion_gen.generative.kl_loss
                )
                loss = (
                    mse_loss
                    + self.args.zeta * kl_loss
                    + self.args.eta * dsm_loss
                    + kl_w * latent_kl
                )
                latent.append(latent_kl.item())
                mse.append(mse_loss.item())
                kl.append(kl_loss.item() * self.args.zeta)
                dsm.append(dsm_loss.item() * self.args.eta)
                all_loss.append(loss.item())
                loss.backward()
                denoise_optim.step()
                if i % 40 == 0:
                    print(loss)
            all_loss = np.average(all_loss)
            kl = np.average(kl)
            dsm = np.average(dsm)
            mse = np.average(mse)
            latent = np.average(latent)
            vali_mse = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | MSE: {2:.7f} KL: {3:.7f} DSM: {4:.7f} LatentKL: {5:.7f} KL_w: {6:.5f} Loss:{7:.7f}".format(
                    epoch + 1, train_steps, mse, kl, dsm, latent, kl_w, all_loss
                )
            )
            early_stopping(vali_mse, self.denoise_net, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(denoise_optim, epoch + 1, self.args)
        best_model_path = path + "/" + "checkpoint.pth"
        self.denoise_net.load_state_dict(torch.load(best_model_path))

    def test(self, setting):
        copy_parameters(self.denoise_net, self.pred_net)
        self.pred_net.eval()
        test_data, test_loader = self._get_data(flag="test")
        preds = []
        trues = []
        noisy = []
        sigmas = []
        inputs = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, _ in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[..., -self.args.target_dim :].float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                noisy_out, out, sigma_out = self.pred_net(batch_x, batch_x_mark)
                noisy.append(noisy_out.squeeze(1).detach().cpu().numpy())
                preds.append(out.squeeze(1).detach().cpu().numpy())
                sigmas.append(sigma_out.squeeze(1).detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                inputs.append(batch_x[..., -1:].detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        noisy = np.concatenate(noisy, axis=0)
        sigmas = np.concatenate(sigmas, axis=0)
        inp = np.concatenate(inputs, axis=0)
        # MSE en espacio estandarizado (comparable con la Tabla 2 del paper).
        mse = np.mean((preds - trues) ** 2)
        print("mse:{}".format(mse))

        model_dir = 'bi-mamba' if getattr(self.args, 'use_bimamba', False) else 'dva'
        folder_path = './results/' + model_dir + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'noisy.npy', noisy)
        np.save(folder_path + 'pred_sigma.npy', sigmas)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'input.npy', inp)
        # Estadisticos del scaler del target (train) para invertir a retornos reales
        # en el analisis de portafolio / metricas.
        np.save(folder_path + 'target_mean.npy', np.asarray(test_data.target_mean))
        np.save(folder_path + 'target_std.npy', np.asarray(test_data.target_std))
        return mse
