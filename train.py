import argparse
import torch
from dconv_model import DistillNet
from ImageLoaders import PairedImageSet
from loss import PerceptualLossModule
from torch.optim.lr_scheduler import MultiStepLR  
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import analyze_image_pair_lab, compute_shadow_mask_otsu
import os  
import gc
from PIL import Image
from torchvision import transforms
import numpy as np
import wandb

os.environ['TORCH_HOME'] = "C:/Users/Raytrack/Desktop/mxq-2.29/"

if __name__ == '__main__':
    # parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=60, help="number of epochs of training")
    parser.add_argument("--resume_epoch", type=int, default=1, help="epoch to resume training")  # 重载训练，从之前中断处接着
    parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")

    parser.add_argument("--optimizer", type=str, default="adam", help="['adam']adam ['sgd']sgd")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient") # GPT说常默认0.9
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient") # GPT说常默认0.999
    parser.add_argument("--gamma", type=float, default=0.2, help="adam: 学习率衰减的乘数")

    parser.add_argument("--decay_epoch", type=int, default=20, help="epoch from which to start lr decay")
    parser.add_argument("--decay_steps", type=int, default=5, help="number of step decays")

    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")  # 也要改一下，租的GPU
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")

    parser.add_argument("--pixelwise_weight", type=float, default=1.0, help="Pixelwise loss weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.1, help="Perceptual loss weight")
    parser.add_argument("--mask_weight", type=float, default=0.02, help="mask loss weight")

    parser.add_argument("--val_checkpoint", type=int, default=1, help="checkpoint for validation")
    parser.add_argument("--save_checkpoint", type=int, default=8, help="checkpoint for visual inspection") # valdataset中每个几个保存一下图片，尽量减少计算
    opt = parser.parse_args()

    wandb.init(project="DNSR-MaterialData-sweep1", config=vars(opt))
    wandb.config.update(opt)

    print('CUDA: ', torch.cuda.is_available(), torch.cuda.device_count())

    criterion_pixelwise = torch.nn.MSELoss() 
    pl = PerceptualLossModule()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    translator = DistillNet(num_iblocks=6, num_ops=4)
    # translator = torch.nn.DataParallel(translator.cuda(), device_ids=[0,1])  # 用了现有结构的.pth, 不能改6和4，除非UNet
    translator.load_state_dict(torch.load("C:/Users/Raytrack/Desktop/mxq-2.29/gen_sh2f_mapped.pth"))
    translator = translator.to(device)
      
    print("USING CUDA FOR MODEL TRAINING")
    criterion_pixelwise.cuda()

    if opt.optimizer == "adam":
        optimizer_G = torch.optim.Adam(translator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    elif opt.optimizer == "sgd":
        optimizer_G = torch.optim.SGD(translator.parameters(), lr=opt.lr, momentum=0.9)

    decay_step = (opt.n_epochs - opt.decay_epoch) // opt.decay_steps
    milestones = [me for me in range(opt.decay_epoch, opt.n_epochs, decay_step)] 
    scheduler = MultiStepLR(optimizer_G, milestones=milestones, gamma=opt.gamma)
   
    Tensor = torch.cuda.FloatTensor

    train_set = PairedImageSet('C:/Users/Raytrack/Desktop/mxq-2.29/MaterialData-sweep', 'train', use_mask=False, aug=False)
    val_set = PairedImageSet('C:/Users/Raytrack/Desktop/mxq-2.29/MaterialData-sweep', 'validation', use_mask=False, aug=False)

    dataloader = DataLoader(
        train_set,  
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,  
        drop_last=False
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=True,
        num_workers=opt.n_cpu
    )

    # num_samples = len(dataloader)
    # val_samples = len(val_dataloader)
    
    wandb.define_metric("Epoch")
    wandb.define_metric("loss/*", step_metric="Epoch")
    wandb.define_metric("main/*", step_metric="Epoch")
    wandb.define_metric("main/val_rmse", summary="min")
    # best_rmse = 600
        
    for epoch in range(opt.resume_epoch, opt.n_epochs):
        train_loss = 0
        train_pix_loss = 0
        train_perc_loss = 0
        train_mask_loss = 0

        val_loss = 0
        val_mask_loss = 0
        val_perc_loss = 0
        val_pix_loss = 0

        err_rmse = 0
        err_psnr = 0

        translator = translator.cuda()
        translator = translator.train()

        for i, (B_img, AB_mask, A_img) in enumerate(dataloader):
            B_img = B_img.to(device)
            AB_mask = AB_mask.to(device)
            A_img = A_img.to(device)

            # 将图像分割为16个512x512块
            for m in range(4):
                for n in range(4):
                    left = n * 512
                    upper = m * 512
                    right = (n + 1) * 512
                    lower = (m + 1) * 512

                    gt = B_img[:, :, upper:lower, left:right]
                    mask = AB_mask[:, :, upper:lower, left:right]
                    inp = A_img[:, :, upper:lower, left:right]

                    # 将每个块送入网络模型进行训练,输出结果      
                    optimizer_G.zero_grad()
                    out = translator(inp, mask) # mask计算中使用的otsu方法计算阴影遮罩mask不太靠谱吧。。。。
                        
                    # 模仿源文件，设计一系列loss计算
                    synthetic_mask = compute_shadow_mask_otsu(inp, out.clone().detach())
                    mask_loss = criterion_pixelwise(synthetic_mask, mask)
                    loss_pixel = criterion_pixelwise(out, gt)
                    perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())
                    loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss + opt.mask_weight * mask_loss
                    loss_G.backward()  # 计算/累积梯度
                        
                    # 计算每一块的tile_loss之和，遍历所有pic的所有16 tiles
                    train_loss += loss_G.detach().item()
                    train_pix_loss += loss_pixel.detach().item()
                    train_perc_loss += perceptual_loss.detach().item()
                    train_mask_loss += mask_loss.detach().item()

            # 一个batch后更新模型参数
            optimizer_G.step()
        
        wandb.log({
             "main/train_loss": train_loss,
             "loss/train_mask_loss": train_mask_loss,
             "loss/train_pix_loss": train_pix_loss,
             "loss/train_perc_loss": train_perc_loss,
             "Epoch": epoch
         })

        scheduler.step() # 训练结束，根据设定更新学习率

# 在评估阶段，模型的目标是衡量其在未见过的数据上的性能，而不再进行参数的更新。这通常发生在训练完成后，用于验证集或测试集上。
# 在评估阶段，模型应该是固定的，不再进行参数更新。为了获得一致的结果，一些层（如 Batch Normalization）可能会使用固定的统计数据而不是每个批次的统计数据。
        if epoch % opt.val_checkpoint == 0:
            with torch.no_grad():
                translator = translator.eval()

                for idx, (B_img, AB_mask, A_img) in enumerate(val_dataloader):
                    B_img = B_img.to(device)
                    AB_mask = AB_mask.to(device)
                    A_img = A_img.to(device)                    
                    all_tiles = [] # if epoch % opt.save_checkpoint == 0:

                    for m in range(4):
                        for n in range(4):
                            left = n * 512
                            upper = m * 512
                            right = (n + 1) * 512
                            lower = (m + 1) * 512

                            gt = B_img[:, :, upper:lower, left:right]
                            mask = AB_mask[:, :, upper:lower, left:right]
                            inp = A_img[:, :, upper:lower, left:right]

                            optimizer_G.zero_grad()
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                out = translator(inp, mask)

                            if idx % opt.save_checkpoint == 0 and idx > 0:
                                out_img = transforms.ToPILImage()(out[0])
                                all_tiles.append(out_img)
                                if m == 3 and n == 3:
                                    fullout = Image.new('RGB', (2048, 2048)) # 创建一个空白的完整图片
                                    for i, tile in enumerate(all_tiles): # 遍历all_tiles中的小块，将它们拼接到完整图片上
                                        row = i // 4
                                        col = i % 4
                                        left = col * 512
                                        upper = row * 512
                                        fullout.paste(tile, (left, upper)) # 此处更改为接缝处理算法！！
                                        wandb.log({"prediction_epoch{}_{}".format(epoch, idx): [wandb.Image(fullout)]})
                                    
                            synthetic_mask = compute_shadow_mask_otsu(inp, out.clone().detach())
                            mask_loss = criterion_pixelwise(synthetic_mask, mask)
                            loss_pixel = criterion_pixelwise(out, gt)
                            perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())
                            loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss + opt.mask_weight * mask_loss
                            rmse, psnr = analyze_image_pair_lab(out.squeeze(0), gt.squeeze(0))

                            # 计算每一块的tile_loss之和，遍历所有val_pic的所有16 tiles
                            val_loss += loss_G.detach().item()
                            val_mask_loss += mask_loss.detach().item()
                            val_pix_loss += loss_pixel.detach().item()
                            val_perc_loss += perceptual_loss.detach().item()
                            err_rmse += rmse
                            err_psnr += psnr

        # table_data.append([epoch, train_loss, val_loss])
        # table = wandb.Table(data=table_data, columns=["Epoch", "train_loss", "val_loss"])
        # table_line = wandb.plot.line(table, x='Epoch', y1='train_loss',  y2='val_loss', title='Train vs. Val Loss')
        wandb.log({
             "main/val_loss": val_loss,
             "loss/val_mask_loss": val_mask_loss,
             "loss/val_pix_loss": val_pix_loss,
             "loss/val_perc_loss": val_perc_loss,
             "main/val_rmse": err_rmse,
             "main/val_psnr": err_psnr,
             # "main/Train vs. Val Loss": table_line
             "Epoch": epoch
        })

        print("EPOCH{}  -  LOSS: {:.3f} | {:.3f}  -  RMSE {:.3f}  -  PSNR {:.3f}  -  MskLoss: {:.3f} | {:.3f} ".format(
                                                                                    epoch, train_loss, val_loss, err_rmse, err_psnr,
                                                                                    train_mask_loss, val_mask_loss))
        
        # if _rmse < best_rmse and epoch > 1:
        #     best_rmse = _rmse
        #     wandb.config.update({"best_rmse": best_rmse}, allow_val_change=True)
        #     print("Saving checkpoint for epoch {} and RMSE {}".format(epoch, best_rmse))
        #     torch.save(translator.cpu().state_dict(), "./best_rmse_model/distillnet_epoch{}.pth".format(epoch))
        #     torch.save(optimizer_G.state_dict(), "./best_rmse_model/optimizer_epoch{}.pth".format(epoch))
