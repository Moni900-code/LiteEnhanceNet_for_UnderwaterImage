import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.nn import Module
import torchvision
from torchvision import transforms
import argparse
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange

from dataloader import myDataSet
from metrics_calculation import *
from model import *
from combined_loss import *

__all__ = [
    "Trainer",
    "setup",
    "training",
]

@dataclass
class Trainer:
    model: Module
    opt: torch.optim.Optimizer
    loss: Module

    @torch.enable_grad()
    def train(self, train_dataloader, config, test_dataloader=None):
        device = config.device
        primary_loss_lst = []
        vgg_loss_lst = []
        total_loss_lst = []

        # Initial evaluation before training
        if config.test and test_dataloader is not None:
            UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
            print(f"Epoch [0] - UIQM: {np.mean(UIQM):.4f}, SSIM: {np.mean(SSIM):.4f}, PSNR: {np.mean(PSNR):.4f}")

        for epoch in trange(0, config.num_epochs, desc="[Full Loop]", leave=False):
            primary_loss_tmp, vgg_loss_tmp, total_loss_tmp = 0, 0, 0

            # Learning rate decay
            if epoch > 1 and epoch % config.step_size == 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] *= 0.7

            for inp, label, _ in tqdm(train_dataloader, desc=f"[Train Epoch {epoch}]", leave=False):
                inp, label = inp.to(device), label.to(device)

                self.model.train()
                self.opt.zero_grad()
                out = self.model(inp)
                loss, mse_loss, vgg_loss = self.loss(out, label)

                loss.backward()
                self.opt.step()

                primary_loss_tmp += mse_loss.item()
                vgg_loss_tmp += vgg_loss.item()
                total_loss_tmp += loss.item()

            total_loss_lst.append(total_loss_tmp / len(train_dataloader))
            vgg_loss_lst.append(vgg_loss_tmp / len(train_dataloader))
            primary_loss_lst.append(primary_loss_tmp / len(train_dataloader))

            # Print loss
            if epoch % config.print_freq == 0:
                print(f"Epoch [{epoch}/{config.num_epochs}] - Total Loss: {total_loss_lst[-1]:.4f}, Primary Loss: {primary_loss_lst[-1]:.4f}, VGG Loss: {vgg_loss_lst[-1]:.4f}")

            # Evaluation
            if config.test and epoch % config.eval_steps == 0 and test_dataloader is not None:
                UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
                print(f"Evaluation at Epoch [{epoch+1}] - UIQM: {np.mean(UIQM):.4f}, SSIM: {np.mean(SSIM):.4f}, PSNR: {np.mean(PSNR):.4f}")

            # Saving model
            if epoch % config.snapshot_freq == 0:
                os.makedirs(config.snapshots_folder, exist_ok=True)
                save_path = os.path.join(config.snapshots_folder, f'model_epoch_{epoch}.ckpt')
                torch.save(self.model, save_path)

    @torch.no_grad()
    def eval(self, config, test_dataloader, test_model):
        test_model.eval()
        for img, _, name in test_dataloader:
            img = img.to(config.device)
            output = test_model(img)
            torchvision.utils.save_image(output, os.path.join(config.output_images_path, name[0]))

        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path, config.GTr_test_images_path)
        UIQM_measures = calculate_UIQM(config.output_images_path)
        return UIQM_measures, SSIM_measures, PSNR_measures

def setup(config):
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Mynet().to(config.device)
    transform = transforms.Compose([
        transforms.Resize((config.resize, config.resize)),
        transforms.ToTensor()
    ])

    train_dataset = myDataSet(config.raw_images_path, config.label_images_path, transform, train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False)
    print("Train Dataset Reading Completed.")
    print(model)

    loss = combinedloss(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    trainer = Trainer(model, opt, loss)

    if config.test:
        test_dataset = myDataSet(config.test_images_path, None, transform, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
        print("Test Dataset Reading Completed.")
        return train_dataloader, test_dataloader, model, trainer

    return train_dataloader, None, model, trainer

def training(config):
    train_loader, test_loader, model, trainer = setup(config)
    trainer.train(train_loader, config, test_loader)
    print("==================")
    print("Training complete!")
    print("==================")

def main(config):
    training(config)

if __name__ == '__main__':
    from argparse import Namespace
    config = Namespace(
        raw_images_path = "/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA/",
        label_images_path = "/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB/",
        test_images_path = "/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA/",
        GTr_test_images_path = "/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/validation/",
        test = True,
        lr = 0.0002,
        step_size = 50,
        num_epochs = 50,
        train_batch_size = 16,
        test_batch_size = 16,
        resize = 256,
        cuda_id = 0,
        print_freq = 1,
        snapshot_freq = 2,
        snapshots_folder = "./snapshots/",
        output_images_path = "./data/output/",
        eval_steps = 1
    )
    main(config)


