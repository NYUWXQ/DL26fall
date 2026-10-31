# NYU Torch (HPC) 运行本项目的完整流程

> 目标：把本仓库的训练搬到 NYU Torch 的 GPU 上跑，产物（checkpoint / submission.csv）能拉回本机提交 Kaggle。
> 本文每条命令都对应 `services.rt.nyu.edu/docs/hpc/` 的真实文档；标注 **[已核实]** 的来自文档原文，标注 **[待本机核实]** 的需要在能连上 Torch 的机器上跑一次确认。

## 0. 前置条件（本机现状）

**[已核实] 本机目前连不上 Torch**：`login.torch.hpc.nyu.edu` DNS 解析失败（`pypi.org` / `kaggle.com` 正常），说明本机不在 NYU 网络内。需要先连 **NYU VPN** 或校园网。

`~/.ssh/config` 里已经有可用配置：`User qs2283`，`Host torch / login.torch.hpc.nyu.edu`，compute 节点走 `ProxyCommand ssh -W %h:%p torch-login`。

连通性自测：

```bash
ssh -o BatchMode=yes torch "hostname; my_slurm_accounts"
```

- 返回主机名 + 一个 `torch_pr_xxxxxx` 形式的 account 名 → 可以投作业。
- `Permission denied` → SSH key 没登记到 Torch（把 `~/.ssh/id_ed25519.pub` 加到 HPC portal 的 SSH keys）。
- 超时 → VPN 没连。

**[已核实]** 投作业必须有 **active allocation**，且每条 `sbatch` 必须带 `--account=`；可用 `my_slurm_accounts` 列出自己有权限的 account。

## 1. 上传代码与数据

Torch 的**算力节点没有外网**（`[已核实]` 文档明确写 "compute nodes do not have internet"），所以数据必须先传上去。

```bash
# 1) 在 Torch 上建工作目录（/scratch 是快盘，home 有配额）
ssh torch "mkdir -p /scratch/\$USER/dl26fall/{data,runs,code}"

# 2) 传代码（不含 .venv / runs / 图片）
rsync -av --exclude '.venv' --exclude 'runs' --exclude '.cache' \
      --exclude 'train_images' --exclude 'test_images' --exclude '.git' \
      ./ torch:/scratch/\$USER/dl26fall/code/

# 3) 传数据（50k+2k 张 PNG 约 150 MB，一次即可）
rsync -av train_images test_images *.csv torch:/scratch/\$USER/dl26fall/data/
```

大数据集时用 Globus 更稳（文档有 Globus 端点），但本项目 ~150 MB，`rsync` 足够。
传完在 Torch 上做一次校验：

```bash
ssh torch "cd /scratch/\$USER/dl26fall/data && ls train_images | wc -l && ls test_images | wc -l"
# 期望 50000 和 2000
```

## 2. 准备 PyTorch 运行环境（overlay + SIF）

**[已核实]** 文档规定的组合是 Apptainer + conda overlay：

- 镜像：`/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`
- overlay：`/scratch/$USER/pytorch-example/my_pytorch.ext3`（按 Apptainer with Conda 页面创建）
- 用法：`singularity exec --nv --overlay <overlay>:ro <sif> /bin/bash -c "source /ext3/env.sh; python ..."`

租一个 CPU 交互节点来建环境（**不要在登录节点 pip install**）：

```bash
srun --pty -c 2 --mem=8GB --time=1:00:00 /bin/bash
singularity exec --overlay /scratch/$USER/dl26fall/pytorch.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
# 容器内：
source /ext3/env.sh
pip install --no-cache-dir torch torchvision numpy pillow
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
exit; exit
```

> 若已有别人建好的 overlay，`--overlay ...:ro` 只读挂载即可，不必重建。
> `[待本机核实]` overlay 首次创建耗时 10–20 分钟，建议一次做好复用。

## 3. 提交训练作业

仓库里给了可直接改的模板：`slurm/train.sbatch`。核心参数（**[已核实]**）：

| 项 | 用法 |
|---|---|
| 账号 | `#SBATCH --account=torch_pr_XXXXXX`（必填，否则拒投） |
| GPU | `#SBATCH --gres=gpu:1` |
| GPU 型号 | `#SBATCH --constraint='h200\|l40s'`（可选；不写则由调度器分配） |
| 分区 | **不要手写 `--partition`**（文档明确要求，资源由 QoS 分配） |
| 抢占 | `#SBATCH --comment="preemption=yes;requeue=true"` 允许用别人的空闲资源，被抢占后自动重新排队 |
| GPU 利用率 | 低于阈值会被**自动 cancel**（gl*/ga*/gr* 50%、gh* 60%、默认 10%）→ 数据加载必须够快 |
| 用户 GPU 配额 | 48 小时内最多 24 张卡 |

提交与观察：

```bash
cd /scratch/$USER/dl26fall/code
sbatch slurm/train.sbatch
squeue -u $USER
tail -f runs/hpc_*.log
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,MaxVMSize
```

断点续训：`train.py --resume` 会从 `last.pt` 继续（实现见 `slurm/train.sbatch` 注释里的两种方案；本项目 300 epoch ≈ 3–4 小时，放在 `< 48h` 窗口内一次跑完最省事）。

## 4. 取回结果并提交 Kaggle

```bash
# 拉回 checkpoint 与提交 CSV
rsync -av torch:/scratch/\$USER/dl26fall/runs/wrn22x4_s0/final.pt ./runs/hpc/
rsync -av torch:/scratch/\$USER/dl26fall/submissions/ ./submissions/
kaggle competitions submit -c cifar-10-res-net-architecture-challenge-dl \
    -f submissions/submission.csv -m "hpc ensemble tta"
```

## 5. 常见坑

1. **算力节点无外网**：`torchvision.datasets.CIFAR10(download=True)` 在节点上会失败；本项目数据已随仓库上传，不依赖下载。
2. **登录节点不要跑训练**：会被杀 + 影响别人；一律 `sbatch` 或 `srun --pty`。
3. **被抢占**：`requeue=true` 会自动重投，但要从 checkpoint 续训，所以 `train.py` 必须每个 epoch 存 `last.pt`（本项目已实现）。
4. **GPU 空转被杀**：`num_workers` 至少 4–8，数据要先 load 成 tensor 缓存（本项目 `src/data.py` 已做 `.cache/*.npy`）。
5. **`--constraint` 写错型号**：可用 `sinfo -o "%N %G %P %t"` 看真实 GPU 型号；文档只保证 `h200` / `l40s`。
6. **本机 kaggle 凭据**：`~/.kaggle/access_token` 已存在且 `kaggle` CLI 2.2.5 可用；不要在 HPC 上提交，拉回来本机提交即可。

## 6. 我做不到的部分（需要你或 VPN）

- 本机未连 NYU VPN → `ssh torch` 不通，所以我**没有**真实提交过任何 HPC 作业；上面所有命令都是按官方文档写好的，连通后可直接执行。
- 若 `qs2283` 不是你的 NetID，或 SSH key 未登记，需要你在 HPC portal 里改/加。
