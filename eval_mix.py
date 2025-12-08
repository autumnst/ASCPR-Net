import os
import cv2
import torch
import numpy as np
import lpips
import pandas as pd
from torch.autograd import Variable
from torchvision.models import inception_v3
from torchvision.models.inception import Inception_V3_Weights
import torchvision.transforms as transforms
from scipy import linalg
from skimage.metrics import structural_similarity as compare_ssim


class calpackage(object):
    def __init__(self, mode="all"):
        self.mode = mode

    def call(self, img1, img2):
        loss_fn_alex = lpips.LPIPS(net='alex', verbose=False)
        if torch.cuda.is_available():
            loss_fn_alex = loss_fn_alex.cuda()
            img1 = img1.cuda()
            img2 = img2.cuda()
        lpips_value_alex = loss_fn_alex(img1, img2, normalize=True)
        return lpips_value_alex


def calculate_fid(act_pred, act_real):
    assert len(act_pred.shape) == 2 and len(act_real.shape) == 2, \
        f"FID 输入必须是二维数组，实际维度: pred={act_pred.shape}, real={act_real.shape}"
    
    mu_pred, sigma_pred = act_pred.mean(axis=0), np.cov(act_pred, rowvar=False)
    mu_real, sigma_real = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    
    ssdiff = np.sum((mu_pred - mu_real) **2.0)
    covmean = linalg.sqrtm(sigma_pred.dot(sigma_real))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    covmean = np.maximum(covmean, 0)
    fid = ssdiff + np.trace(sigma_pred + sigma_real - 2.0 * covmean)
    return fid


def get_inception_feature(img_tensor, model):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        img = preprocess(img_tensor)
        img = img.unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
        feat = model(img)
    return feat.detach().cpu().numpy().ravel()


def get_metrics(img, referimg, mask=None, size=512):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    referimg = cv2.resize(referimg, (size, size), interpolation=cv2.INTER_AREA)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    referimg_rgb = cv2.cvtColor(referimg, cv2.COLOR_BGR2RGB)
    (score, diff) = compare_ssim(
        referimg_rgb, img_rgb, 
        win_size=21, 
        full=True, 
        channel_axis=2,
        data_range=255
    )
    ssim = score
    
    img = np.array(img, dtype=np.float64).transpose((2, 0, 1))
    referimg = np.array(referimg, dtype=np.float64).transpose((2, 0, 1))
    
    img = torch.from_numpy(img)
    referimg = torch.from_numpy(referimg)
    
    img1 = Variable(img / 255).unsqueeze(0)
    img2 = Variable(referimg / 255).unsqueeze(0)
    
    caltool = calpackage()
    lpips_value_alexnetbase = caltool.call(
        img1.type(torch.FloatTensor), 
        img2.type(torch.FloatTensor)
    )
    lpips_value_alex = lpips_value_alexnetbase.detach().cpu().numpy().mean()
    
    loss2 = torch.mean((img / 255. - referimg / 255.)** 2)
    
    psnr = 10 * torch.log10(1 / loss2) if loss2 != 0 else torch.tensor(float('inf'))
    
    if mask is not None:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
        proportion = np.array(mask, dtype=np.float64).mean() / 255
        if proportion < 1e-6:
            proportion = 1e-6
        ssim = ssim - (1 - ssim) * (1 - proportion) / proportion
        loss2 = loss2 / proportion
        lpips_value_alex = lpips_value_alex / proportion
        psnr = 10 * torch.log10(1 / loss2) if loss2 != 0 else torch.tensor(float('inf'))
    
    return {
        "ssim": ssim,
        "l2": loss2.item(),
        "psnr": psnr.item(),
        "LPIPS_value_alex": lpips_value_alex
    }


# 初始化 InceptionV3 模型
inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
inception_model.fc = torch.nn.Identity()
inception_model.eval()
if torch.cuda.is_available():
    inception_model = inception_model.cuda()


# 配置参数
test_methods = ["generate"]
test_root = "/data/lqq/tql/APLRL_3/results/dh_bs=2/"
referpath = "/data/lqq/tql/APLRL_3/results/dh_bs=2/ground_truth"
maskpath = "/data/lqq/tql/APLRL_data/test_dh/masks/"
print(f"掩码目录路径：{maskpath}（请确认该路径下存在掩码文件）")

# 输出CSV路径
output_csv = os.path.join(test_root, "metrics_results.csv")
output_avg_csv = os.path.join(test_root, "metrics_average.csv")

referfiles = sorted(os.listdir(referpath))
# 预获取掩码文件列表（用于快速检查）
mask_files_exist = set(os.listdir(maskpath)) if os.path.exists(maskpath) else set()

all_results = []
out_dicts = {method: {"ssim": 0, "l2": 0, "psnr": 0, "LPIPS_value_alex": 0} for method in test_methods}
fid_features = {method: {"pred": [], "real": []} for method in test_methods}
valid_count = {method: 0 for method in test_methods}


# 遍历图像计算指标
for i, filename in enumerate(referfiles):
    # 处理参考图像
    refer_img_path = os.path.join(referpath, filename)
    referimg = cv2.imread(refer_img_path)
    if referimg is None:
        print(f"警告：参考图像 {refer_img_path} 无法读取（可能损坏），跳过")
        continue
    
    # 处理掩码文件（核心优化）
    mask = None
    mask_filename = filename
    # 先检查掩码文件是否存在于目录中
    if mask_filename not in mask_files_exist:
        print(f"警告：掩码文件 {mask_filename} 不存在于 {maskpath}，不使用掩码")
    else:
        # 存在则尝试读取
        mask_path = os.path.join(maskpath, mask_filename)
        mask = cv2.imread(mask_path)
        if mask is None:
            print(f"警告：掩码文件 {mask_path} 存在但无法读取（可能损坏），不使用掩码")
        # else:  # 可选：取消注释可查看成功加载的掩码
        #     print(f"已加载掩码：{mask_filename}")
    
    # 处理预测图像并计算指标
    referimg_rgb = cv2.cvtColor(referimg, cv2.COLOR_BGR2RGB)
    referimg_tensor = torch.from_numpy(referimg_rgb).permute(2, 0, 1).float() / 255.0
    
    for method in test_methods:
        pred_img_path = os.path.join(test_root, method, filename)
        img = cv2.imread(pred_img_path)
        if img is None:
            print(f"警告：预测图像 {pred_img_path} 无法读取，跳过")
            continue
        
        values = get_metrics(img, referimg, mask, size=256)
        single_result = {
            "filename": filename,
            "method": method,
            **values
        }
        all_results.append(single_result)
        
        for key in values.keys():
            out_dicts[method][key] += values[key]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        pred_feat = get_inception_feature(img_tensor, inception_model)
        real_feat = get_inception_feature(referimg_tensor, inception_model)
        fid_features[method]["pred"].append(pred_feat)
        fid_features[method]["real"].append(real_feat)
        valid_count[method] += 1
    
    if (i + 1) % 10 == 0:
        print(f"已处理 {i + 1}/{len(referfiles)} 张图像")


# 保存单张结果
df = pd.DataFrame(all_results)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"单张图像指标已保存至：{output_csv}")


# 计算并保存平均结果
avg_results = []
for method in test_methods:
    count = valid_count[method]
    if count == 0:
        print(f"方法 {method} 无有效样本，跳过平均计算")
        continue
    
    ssim_avg = out_dicts[method]["ssim"] / count
    l2_avg = out_dicts[method]["l2"] / count
    psnr_avg = out_dicts[method]["psnr"] / count
    lpips_avg = out_dicts[method]["LPIPS_value_alex"] / count
    
    act_pred = np.array(fid_features[method]["pred"])
    act_real = np.array(fid_features[method]["real"])
    fid_score = calculate_fid(act_pred, act_real)
    
    avg_result = {
        "method": method,
        "valid_samples": count,
        "ssim_avg": ssim_avg,
        "l2_avg": l2_avg,
        "psnr_avg": psnr_avg,
        "LPIPS_alex_avg": lpips_avg,
        "fid_score": fid_score
    }
    avg_results.append(avg_result)
    
    print("\n" + "=" * 80)
    print(f"方法: {method}")
    print(f"有效样本数: {count}")
    print(f"L2损失平均值: {l2_avg:.4f}")
    print(f"PSNR平均值: {psnr_avg:.4f}")
    print(f"SSIM平均值: {ssim_avg:.4f}")
    print(f"LPIPS_alex平均值: {lpips_avg:.4f}")
    print(f"FID值: {fid_score:.4f}")
    print("=" * 80)


df_avg = pd.DataFrame(avg_results)
df_avg.to_csv(output_avg_csv, index=False, encoding="utf-8-sig")
print(f"平均指标已保存至：{output_avg_csv}")