# 论文代码使用说明

论文：**High-Capacity Robust Diffusion Steganography via Quantization Index Modulation**。

本目录可作为一个独立 GitHub 仓库。完整安装步骤、参数表、图像指标与 DCT 直方图命令见英文 README。代码包含主方法、提取、鲁棒性实验入口、视觉质量评估和消融分析；未包含模型权重、实验图片、个人服务器配置及后续探索分支。

## 最常用的运行方式

在 Linux 服务器安装 README 所列依赖后，在仓库根目录执行：

```bash
# 先检查展开的实验指令，不加载模型
python run_paper.py --model /path/to/stable-diffusion-2-1-base --images 5 --dry_run

# 415 bit、172张、无攻击
CUDA_VISIBLE_DEVICES=0 python -u run_paper.py --model /path/to/stable-diffusion-2-1-base --preset E5 --attack none

# 415 bit、JPEG75
CUDA_VISIBLE_DEVICES=0 python -u run_paper.py --model /path/to/stable-diffusion-2-1-base --preset E5 --attack jpeg75

# 65 bit、172张、无攻击
CUDA_VISIBLE_DEVICES=0 python -u run_paper.py --model /path/to/stable-diffusion-2-1-base --preset k65 --attack none
```

四模型对比只更换 `--model` 和 `--model_tag`，保持提示词、容量、生成种子一致。主实验提示词为 `prompts/paper172.txt`；五张示例可使用 `--prompt_file prompts/five_styles.txt --images 5`。

## 代码来源与校验

本次整理发现本地代码与服务器代码的字节哈希不同，因此主生成脚本、QIM 和 LDPC 使用与正式实验 manifest 哈希一致的服务器版本。扩散管线依赖一并补齐。代码来源和哈希记录在 `SOURCE_MANIFEST.json` 中。该记录不包含个人挂载盘或服务器绝对路径。

```bash
python verify_release.py
python test_codec.py
```

第一条检查文件完整性和 Python 语法；第二条在 CPU 上检查 LDPC 编解码，并在安装 PyTorch 时检查 DCT/QIM 的无噪声回环。GPU 生成和完整指标计算仍需在实验环境运行。依赖文件是依据旧版接口整理的兼容约束，不是服务器完整环境的导出。

## 上传到 GitHub

在 GitHub 创建一个空仓库后，从**本目录内部**执行（替换仓库地址）：

```bash
git init
git add .
git status --short
git commit -m "Prepare DCT-QIM paper implementation"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/dct-qim-steganography.git
git push -u origin main
```

不要从上级研究目录执行 `git add .`。`.gitignore` 已忽略模型权重、输出目录、环境配置、缓存及日志。原始新增代码的整体开源许可证尚未指定，上游代码的原有许可和署名已保留。
