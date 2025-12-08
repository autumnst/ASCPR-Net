import os
import torch
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter
from utils.distributed import get_rank, reduce_loss_dict
from utils.misc import requires_grad, sample_data, postprocess
from criteria.loss import generator_loss_func, discriminator_loss_func
import utils.utils_image as cal

from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image


# ---------------------------------------------
# LPIPS
# ---------------------------------------------
def calculate_lpips(img1, img2, lpips_criterion):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)

    img1 = torch.autograd.Variable(img1, requires_grad=False).cuda()
    img2 = torch.autograd.Variable(img2, requires_grad=False).cuda()

    lpips_value = lpips_criterion(img1, img2).item()
    return lpips_value



# ============================================================
#                      TRAIN FUNCTION (16 参数版本)
# ============================================================
def train(
    opts,
    image_data_loader,
    generator,
    discriminator,
    extractor,
    generator_optim,
    discriminator_optim,
    is_cuda,
    test_opts,
    image_data_loader_test,
    lpips_criterion,
    aai_extractor,
    reconstructor,            # 13
    rec_discriminator,        # 14
    reconstructor_optim,      # 15
    rec_discriminator_optim   # 16
):

    # ------------------------
    # dataloader
    # ------------------------
    image_data_loader = sample_data(image_data_loader)
    image_data_loader_test = sample_data(image_data_loader_test)

    pbar = range(opts.train_iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opts.start_iter, dynamic_ncols=True, smoothing=0.01)

    # ------------------------
    # unwrap DDP
    # ------------------------
    if opts.distributed:
        generator_module = generator.module
        discriminator_module = discriminator.module
        reconstructor_module = reconstructor.module
        rec_discriminator_module = rec_discriminator.module
    else:
        generator_module = generator
        discriminator_module = discriminator
        reconstructor_module = reconstructor
        rec_discriminator_module = rec_discriminator

    writer = SummaryWriter(opts.log_dir)

    # ================================================================
    #                           TRAIN LOOP
    # ================================================================
    for index in pbar:

        i = index + opts.start_iter
        if i > opts.train_iter:
            print("Done.")
            break

        # ------------------------------
        # Get data
        # ------------------------------
        ground_truth, mask, edge, gray_image = next(image_data_loader)
        if is_cuda:
            ground_truth = ground_truth.cuda()
            mask = mask.cuda()
            edge = edge.cuda()
            gray_image = gray_image.cuda()

        input_image = ground_truth * mask
        input_edge = edge * mask
        input_gray_image = gray_image * mask

        # ============================================================
        #             Stage-1: Train Reconstructor + Rec_D
        # ============================================================
        if i <= opts.rec_pretrain_iter:

            requires_grad(reconstructor, True)
            requires_grad(rec_discriminator, True)

            # rec_out = (struct_img, proj, latent, priors_dict)
            rec_out = reconstructor(input_image, mask)
            struct_img = rec_out[0]

            # --- Train Rec Discriminator ---
            real_pred = rec_discriminator(ground_truth)
            fake_pred = rec_discriminator(struct_img.detach())

            # BCE loss
            rec_d_loss = -(torch.log(real_pred + 1e-8) +
                           torch.log(1 - fake_pred + 1e-8)).mean()

            rec_discriminator_optim.zero_grad()
            rec_d_loss.backward()
            rec_discriminator_optim.step()

            # --- Train Reconstructor G ---
            fake_pred = rec_discriminator(struct_img)
            rec_g_loss_adv = -torch.log(fake_pred + 1e-8).mean()

            # Reconstruction L1
            rec_l1 = torch.mean(torch.abs(struct_img - ground_truth))

            rec_total = (
                rec_g_loss_adv +
                rec_l1 * opts.REC_CODE_LOSS
            )

            reconstructor_optim.zero_grad()
            rec_total.backward()
            reconstructor_optim.step()

            # For generator usage later (detach)
            priors_dict = rec_out[3] if len(rec_out) == 4 else None
            struct_priors = {k: v.detach() for k, v in priors_dict.items()} if priors_dict else None

            if get_rank() == 0:
                writer.add_scalar("rec/l1", rec_l1.item(), i)
                writer.add_scalar("rec/adv", rec_g_loss_adv.item(), i)
                writer.add_scalar("rec/total", rec_total.item(), i)

        # ============================================================
        #             Stage-2: Freeze Reconstructor → only priors
        # ============================================================
        else:
            requires_grad(reconstructor, False)
            requires_grad(rec_discriminator, False)

            with torch.no_grad():
                rec_out = reconstructor(input_image, mask)
            if len(rec_out) == 4:
                struct_priors = rec_out[3]
            else:
                struct_priors = None

        # ============================================================
        #                  Train Generator
        # ============================================================
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        # sync generator → aai extractor
        with torch.no_grad():
            update_dict = {
                k: v for k, v in generator_module.state_dict().items()
                if k in aai_extractor.state_dict()
            }
            aai_extractor.load_state_dict(update_dict)
            aai_extractor.eval()
            aai_gt = aai_extractor(ground_truth, torch.ones_like(ground_truth))

        output, proj, aai_gt_dummy, aai_out = generator(
            input_image,
            torch.cat((input_edge, input_gray_image), dim=1),
            mask,
            ground_truth,
            is_train=True,
            struct_priors=struct_priors
        )

        comp = ground_truth * mask + output * (1 - mask)

        vgg_comp = extractor(comp)
        vgg_output = extractor(output)
        vgg_gt = extractor(ground_truth)

        output_pred = discriminator(output, gray_image, is_real=False)

        g_loss_dict = generator_loss_func(
            mask, output, ground_truth, output_pred,
            vgg_comp, vgg_output, vgg_gt,
            aai_gt, aai_out
        )

        g_total = (
            g_loss_dict["loss_hole"] * opts.HOLE_LOSS +
            g_loss_dict["loss_valid"] * opts.VALID_LOSS +
            g_loss_dict["loss_perceptual"] * opts.PERCEPTUAL_LOSS +
            g_loss_dict["loss_style"] * opts.STYLE_LOSS +
            g_loss_dict["loss_adversarial"] * opts.ADVERSARIAL_LOSS +
            g_loss_dict["loss_intermediate"] * opts.INTERMEDIATE_LOSS +
            g_loss_dict["loss_aai"] * opts.AAI_LOSS
        )
        g_loss_dict["loss_joint"] = g_total

        generator_optim.zero_grad()
        g_total.backward()
        generator_optim.step()

        # ============================================================
        #                  Train Main Discriminator
        # ============================================================
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        real_pred = discriminator(ground_truth, gray_image, is_real=True)
        fake_pred = discriminator(output.detach(), gray_image, is_real=False)

        d_loss_dict = discriminator_loss_func(real_pred, fake_pred, edge)
        d_total = d_loss_dict["loss_adversarial"]
        d_loss_dict["loss_joint"] = d_total

        discriminator_optim.zero_grad()
        d_total.backward()
        discriminator_optim.step()

        # ============================================================
        #                   LOGGING
        # ============================================================
        g_loss_dict_red = reduce_loss_dict(g_loss_dict)
        d_loss_dict_red = reduce_loss_dict(d_loss_dict)

        if get_rank() == 0:
            pbar.set_description(
                f"Iter {i} | G={g_loss_dict_red['loss_joint']:.4f}  "
                f"D={d_loss_dict_red['loss_joint']:.4f}"
            )

            for k, v in g_loss_dict_red.items():
                writer.add_scalar(f"g/{k}", v, i)
            for k, v in d_loss_dict_red.items():
                writer.add_scalar(f"d/{k}", v, i)

        # ============================================================
        #                       Save & Validation
        # ============================================================
        if get_rank() == 0 and (i % opts.save_interval == 0):

            torch.save(
                {
                    "n_iter": i,
                    "generator": generator_module.state_dict(),
                    "discriminator": discriminator_module.state_dict(),
                    "reconstructor": reconstructor_module.state_dict(),
                    "rec_discriminator": rec_discriminator_module.state_dict(),
                },
                f"{opts.save_dir}/{str(i).zfill(6)}.pt"
            )

            print("Start validation ...")
            validate(
                generator_module, aai_extractor, reconstructor,
                test_opts, image_data_loader_test,
                lpips_criterion, is_cuda
            )



# ============================================================
#                   Validation Function
# ============================================================
def validate(generator, aai_extractor, reconstructor,
             test_opts, loader, lpips_criterion, is_cuda):

    generator.eval()
    aai_extractor.eval()

    # 让 loader 循环使用（避免 StopIteration）
    loader_loop = sample_data(loader)

    img_psnr = 0
    img_ssim = 0
    img_lpips = 0

    for idx in range(test_opts.number_eval):

        # -------------------------
        #       读取测试 batch
        # -------------------------
        ground_truth, mask, edge, gray_image = next(loader_loop)

        if is_cuda:
            ground_truth = ground_truth.cuda()
            mask = mask.cuda()
            edge = edge.cuda()
            gray_image = gray_image.cuda()

        input_image = ground_truth * mask
        input_edge = edge * mask
        input_gray = gray_image * mask

        # -------------------------
        #    Structure Priors
        # -------------------------
        with torch.no_grad():
            rec_out = reconstructor(input_image, mask)
            priors_dict = rec_out[3] if len(rec_out) == 4 else None

        # -------------------------
        #    AAI Extractor
        # -------------------------
        with torch.no_grad():
            aai = aai_extractor(
                ground_truth * mask,
                mask.expand_as(ground_truth)
            ) * mask

        # -------------------------
        #    Forward Inference
        # -------------------------
        output, _, _, _ = generator(
            input_image,
            torch.cat((input_edge, input_gray), dim=1),
            mask,
            aai,
            is_train=False,
            struct_priors=priors_dict
        )

        comp = ground_truth * mask + output * (1 - mask)
        comp = postprocess(comp)

        # -------------------------
        #        保存图像
        # -------------------------
        save_image(comp,
                   f"{test_opts.result_root}/generate/{idx:05d}.png")
        save_image(postprocess(ground_truth * mask),
                   f"{test_opts.result_root}/corrupt/{idx:05d}.png")
        save_image(postprocess(ground_truth),
                   f"{test_opts.result_root}/ground_truth/{idx:05d}.png")

        # -------------------------
        #      指标计算
        # -------------------------
        comp_np = comp.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = postprocess(ground_truth).squeeze(0).permute(1, 2, 0).cpu().numpy()

        psnr = cal.calculate_psnr(comp_np, gt_np)
        ssim = cal.calculate_ssim(comp_np, gt_np)
        lpips_value = calculate_lpips(
            Image.fromarray(comp_np.astype(np.uint8)),
            Image.fromarray(gt_np.astype(np.uint8)),
            lpips_criterion
        )

        img_psnr += psnr
        img_ssim += ssim
        img_lpips += lpips_value

    # -------------------------
    #         平均结果
    # -------------------------
    img_psnr /= test_opts.number_eval
    img_ssim /= test_opts.number_eval
    img_lpips /= test_opts.number_eval

    # -------------------------
    #      输出 + 记录日志
    # -------------------------
    print(f"[Val] PSNR={img_psnr:.2f}  SSIM={img_ssim:.3f}  LPIPS={img_lpips:.3f}")

    with open("validation.txt", "a") as f:
        f.write(f"avg_psnr={img_psnr:.2f} "
                f"avg_ssim={img_ssim:.3f} "
                f"avg_lpips={img_lpips:.3f}\n")

