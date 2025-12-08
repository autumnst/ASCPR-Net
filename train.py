import warnings
warnings.filterwarnings("ignore")

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils import data

# ------------------------------
# Model components
# ------------------------------
from models.discriminator.discriminator import Discriminator
from models.rec_discriminator.discriminator import Discriminator as RecDiscriminator
from models.generator.generator import Generator, Extractor
from models.generator.vgg16 import VGG16FeatureExtractor
from models.reconstructor.reconstructor import Reconstructor

# ------------------------------
# Options
# ------------------------------
from options.train_options import TrainOptions
from options.test_options import TestOptions

# ------------------------------
# Dataset & util
# ------------------------------
from datasets.dataset import create_image_dataset
from utils.distributed import synchronize
from utils.ddp import data_sampler
from trainer import train
from lpips import LPIPS


# ==========================================================
# 1) Load opts
# ==========================================================
opt_loader = TrainOptions()
opts = opt_loader.parse         # ← 注意要加括号

test_opt_loader = TestOptions()
test_opts = test_opt_loader.parse

os.makedirs(opts.save_dir, exist_ok=True)


# ==========================================================
# 2) CUDA setup
# ==========================================================
is_cuda = torch.cuda.is_available()
if is_cuda:
    print("Cuda is available")
    cudnn.enabled = True
    cudnn.benchmark = True

    n_gpu = int(os.environ.get("WORLD_SIZE", "1"))
    print("GPU number:", n_gpu)
    opts.distributed = n_gpu > 1

    if opts.distributed:
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        synchronize()
else:
    opts.distributed = False


# ==========================================================
# 3) Build models
# ==========================================================
generator = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
discriminator = Discriminator(image_in_channels=3, edge_in_channels=2)

extractor = VGG16FeatureExtractor()
aai_extractor = Extractor()

# Structure generator & discriminator (MAPs 结构先验分支)
reconstructor = Reconstructor()
rec_discriminator = RecDiscriminator(image_in_channels=3)


# ==========================================================
# 4) Move to GPU
# ==========================================================
if is_cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    extractor = extractor.cuda()
    aai_extractor = aai_extractor.cuda()
    reconstructor = reconstructor.cuda()
    rec_discriminator = rec_discriminator.cuda()


# 5) 先加载主 ckpt，恢复 start_iter
if opts.pre_trained != "":
    print(f"[Main] Loading checkpoint from: {opts.pre_trained}")
    ckpt = torch.load(opts.pre_trained, map_location="cpu")

    if "n_iter" in ckpt:
        opts.start_iter = ckpt["n_iter"]

    if "generator" in ckpt:
        generator.load_state_dict(ckpt["generator"])
        print("[Main] Loaded generator.")
    if "discriminator" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator"])
        print("[Main] Loaded discriminator.")

    # （结构这两行建议 Phase-1 用，Phase-2 用 partial_extractor 覆盖）
    if "reconstructor" in ckpt:
        reconstructor.load_state_dict(ckpt["reconstructor"], strict=False)
        print("[Main] Loaded reconstructor from main ckpt.")
    if "rec_discriminator" in ckpt:
        rec_discriminator.load_state_dict(ckpt["rec_discriminator"], strict=False)
        print("[Main] Loaded rec_discriminator from main ckpt.")

    print("[Main] Starting from iter", opts.start_iter)
else:
    print("[Main] Starting from iter", opts.start_iter)

# 6) 再根据“更新后的 start_iter”判断 Phase-1 / Phase-2
pretrain_iter = opts.rec_pretrain_iter

if opts.start_iter >= pretrain_iter:
    # ===== Phase-2: 结构先验 + 冻结 =====
    print(f">>> Entering Phase-2 (MAPs stage). start_iter={opts.start_iter}, pretrain_iter={pretrain_iter}")

    recon_ckpt_path = f"/data/lqq/tql/APLRL/snapshots/dh1/partial_extractor/{pretrain_iter}.pt"

    if os.path.exists(recon_ckpt_path):
        ck = torch.load(recon_ckpt_path, map_location='cpu')

        if "reconstructor" in ck:
            reconstructor.load_state_dict(ck["reconstructor"], strict=False)
            print("[Phase-2] reconstructor loaded from structure pretrain.")

        if "rec_discriminator" in ck:
            rec_discriminator.load_state_dict(ck["rec_discriminator"], strict=False)
            print("[Phase-2] rec_discriminator loaded from structure pretrain.")
    else:
        print(f"[Phase-2] Structure pretrained file not found: {recon_ckpt_path}")
        exit()

    # 冻结结构分支
    print(">>> Freezing structure prior network (reconstructor & rec_discriminator).")
    for p in reconstructor.parameters():
        p.requires_grad = False
    for p in rec_discriminator.parameters():
        p.requires_grad = False
    print("    - reconstructor: frozen")
    print("    - rec_discriminator: frozen")

else:
    # ===== Phase-1: 结构训练阶段 =====
    print(f">>> Phase-1: Structure training stage (iter {opts.start_iter}/{pretrain_iter})")
    print("    Structure NOT loaded from partial_extractor, NOT frozen.")


# ==========================================================
# 6) Optimizers
#    注意：虽然创建了结构分支的优化器，但由于 requires_grad=False，
#    它们在第二阶段不会产生任何更新，只是为了兼容 trainer 的参数签名。
# ==========================================================
if opts.finetune:
    print("Fine tune mode")
    lr = opts.lr_finetune
    generator.freeze_ec_bn = True
else:
    lr = opts.gen_lr

# Generator + AAI
generator_optim = optim.Adam(
    filter(lambda p: p.requires_grad, generator.parameters()),
    lr=lr
)

discriminator_optim = optim.Adam(
    discriminator.parameters(),
    lr=lr * opts.D2G_lr,
)

# Structure branch (已冻结，只是占位优化器，方便 trainer 接口统一)
reconstructor_optim = optim.Adam(reconstructor.parameters(), lr=lr)
rec_discriminator_optim = optim.Adam(rec_discriminator.parameters(), lr=lr)


# ==========================================================
# 7) Load main pretrained checkpoint (整体模型继续训练用)
# ==========================================================
if opts.pre_trained != "":
    print(f"[Main] Loading checkpoint from: {opts.pre_trained}")
    ckpt = torch.load(opts.pre_trained, map_location="cpu")

    if "n_iter" in ckpt:
        opts.start_iter = ckpt["n_iter"]

    if "generator" in ckpt:
        generator.load_state_dict(ckpt["generator"])
        print("[Main] Loaded generator.")
    if "discriminator" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator"])
        print("[Main] Loaded discriminator.")
    if "reconstructor" in ckpt:
        # **注意**：如果不希望主 ckpt 覆盖结构先验，可以注释掉这一行
        reconstructor.load_state_dict(ckpt["reconstructor"], strict=False)
        print("[Main] (Optional) Loaded reconstructor from main ckpt.")
    if "rec_discriminator" in ckpt:
        rec_discriminator.load_state_dict(ckpt["rec_discriminator"], strict=False)
        print("[Main] (Optional) Loaded rec_discriminator from main ckpt.")

    print("[Main] Starting from iter", opts.start_iter)
else:
    print("[Main] Starting from iter", opts.start_iter)


# ==========================================================
# 8) DDP wrapping
# ==========================================================
if opts.distributed:
    generator = nn.parallel.DistributedDataParallel(
        generator,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        broadcast_buffers=False,
    )
    discriminator = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        broadcast_buffers=False,
    )
    # 结构分支一般单卡即可，这里不包 DDP：reconstructor / rec_discriminator


# ==========================================================
# 9) Train dataset
# ==========================================================
train_set = create_image_dataset(opts)
print("Train dataset size:", len(train_set))

train_loader = data.DataLoader(
    train_set,
    batch_size=opts.batch_size,
    sampler=data_sampler(train_set, shuffle=True, distributed=opts.distributed),
    drop_last=True
)


# ==========================================================
# 10) Validation loader
# ==========================================================
lpips_criterion = LPIPS(net="vgg")
if is_cuda:
    lpips_criterion = lpips_criterion.cuda()

os.makedirs(os.path.join(test_opts.result_root, "ground_truth"), exist_ok=True)
os.makedirs(os.path.join(test_opts.result_root, "generate"), exist_ok=True)
os.makedirs(os.path.join(test_opts.result_root, "corrupt"), exist_ok=True)

test_set = create_image_dataset(test_opts)
print("Validation dataset size:", len(test_set))

test_loader = data.DataLoader(
    test_set,
    batch_size=test_opts.batch_size,
    shuffle=False,
    num_workers=test_opts.num_workers,
    drop_last=False
)


# ==========================================================
# 11) Call Trainer  (16 parameters)
#      —— 第二阶段：使用冻结的结构先验进行 MAPs 训练
# ==========================================================
train(
    opts,
    train_loader,
    generator,
    discriminator,
    extractor,
    generator_optim,
    discriminator_optim,
    is_cuda,
    test_opts,
    test_loader,
    lpips_criterion,
    aai_extractor,
    reconstructor,              # 冻结后的结构先验
    rec_discriminator,          # 冻结后的结构判别器
    reconstructor_optim,        # 仅占位，内部不会真正更新
    rec_discriminator_optim     # 同上
)
