# Auto Research

一套基于 Claude Code Harness 的算法论文全自动研究工作流，输入一篇论文（PDF 或 arXiv 链接），自动完成：

1. **论文解析** — 提取全文结构，按标准章节归类
2. **创新点梳理** — 结构化分析问题定义、模型架构、Loss 设计、训练策略、实验结果
3. **代码复现** — 生成完整可运行的 PyTorch 工程
4. **HTML 报告** — 输出自包含的深色主题可视化摘要页面

整个流程通过 Claude Code 的 Skill（斜杠命令）驱动，无需任何外部服务，本地即可运行。

---

## 效果预览

对任意一篇论文运行 `/auto-research` 后，得到以下四类产物：

```
papers/
└── attention-is-all-you-need.pdf          # 缓存的原始 PDF

analyses/attention-is-all-you-need/
├── raw.md                                  # 提取的原文结构（8 个标准章节）
└── innovations.md                          # 创新点深度分析（问题/贡献/架构/Loss/训练/结果）

reproductions/attention-is-all-you-need/
├── config.py                               # 论文中所有超参数
├── model.py                                # Transformer 架构（PyTorch）
├── loss.py                                 # Label Smoothing Cross-Entropy
├── dataset.py                              # WMT 数据加载与预处理
├── train.py                                # 完整训练循环（含 warmup、checkpointing）
└── README.md                               # 复现说明与预期指标

outputs/attention-is-all-you-need/
├── summary.html                            # 深色主题 HTML 报告，浏览器直接打开
└── model.py                                # 零外部依赖的独立模型文件
```

---

## 安装

**依赖：Claude Code CLI**（需已安装并登录）

```bash
# Python 依赖（PDF 解析）
pip install pdfplumber requests

# 克隆 / 进入项目
cd auto-research

# Claude Code 会自动读取 .claude/settings.json 中的权限配置
```

---

## 快速开始

### 一键全流程

```bash
# 从 arXiv URL
/auto-research https://arxiv.org/abs/1706.03762

# 从 arXiv ID
/auto-research 2310.06825

# 从本地 PDF，指定论文名
/auto-research ./papers/my-paper.pdf mamba-ssm
```

运行结束后，打开 HTML 报告浏览结果：

```bash
open outputs/attention-is-all-you-need/summary.html
```

---

### 分步执行

每个阶段可单独运行，便于调试或从中间结果继续：

```bash
# Stage 1：解析论文 → analyses/{name}/raw.md
/parse-paper https://arxiv.org/abs/1706.03762 attention-is-all-you-need

# Stage 2：分析创新点 → analyses/{name}/innovations.md
/analyze-innovations attention-is-all-you-need

# Stage 3：生成 PyTorch 复现代码 → reproductions/{name}/
/reproduce-code attention-is-all-you-need

# Stage 4：生成 HTML 报告 → outputs/{name}/
/generate-report attention-is-all-you-need
```

---

## 项目结构

```
auto-research/
│
├── .claude/
│   ├── settings.json              # Harness 配置：权限白名单 + 会话日志 Hook
│   └── commands/                  # Skills（斜杠命令定义）
│       ├── parse-paper.md         # Stage 1：论文解析
│       ├── analyze-innovations.md # Stage 2：创新点提取
│       ├── reproduce-code.md      # Stage 3：代码复现（PyTorch）
│       ├── generate-report.md     # Stage 4：HTML 报告生成
│       └── auto-research.md       # 四阶段编排器
│
├── prompts/                       # 各阶段的系统提示词模板
│   ├── parse_system.md            # 章节提取规则
│   ├── innovations_system.md      # 创新点分析精度要求
│   ├── reproduce_system.md        # 代码生成规范（形状注释、命名等）
│   └── html_report_system.md      # HTML 设计规则（配色、自包含要求）
│
├── scripts/                       # Skill 调用的 Python 辅助脚本
│   ├── fetch_paper.py             # 从 arXiv 下载或复制本地 PDF
│   ├── parse_pdf.py               # PDF → 结构化 JSON（章节 + 摘要 + 图表）
│   └── utils.py                   # slugify、状态检查、路径工具
│
├── papers/                        # 缓存的 PDF 文件
│
├── analyses/
│   ├── _template/                 # 输出格式参考模板
│   │   ├── raw.md                 # 原文提取的 8 节标准结构
│   │   └── innovations.md         # 创新点分析的 8 节标准结构
│   └── {paper_name}/              # 每篇论文一个目录
│       ├── raw.md
│       └── innovations.md
│
├── reproductions/
│   ├── _template/                 # PyTorch 代码模板
│   └── {paper_name}/
│       ├── config.py              # 超参数 DataClass（全部来自论文）
│       ├── model.py               # 模型架构
│       ├── loss.py                # Loss 函数（返回 dict，便于逐项消融）
│       ├── dataset.py             # 数据加载与预处理
│       ├── train.py               # 训练循环（optimizer、scheduler、checkpoint）
│       └── README.md              # 复现说明、预期指标、已知差距
│
└── outputs/                       # HTML 报告（主要交付物）
    ├── _template/
    │   ├── summary.html           # HTML 报告模板（深色主题）
    │   └── model.py               # 独立模型文件模板
    └── {paper_name}/
        ├── summary.html           # 完整创新点报告，浏览器直接打开
        └── model.py               # 零本地依赖的独立 PyTorch 模型
```

---

## 各阶段说明

### Stage 1 — 论文解析 (`/parse-paper`)

- 接受 arXiv URL / arXiv ID / 本地 PDF 路径
- `scripts/fetch_paper.py`：从 arXiv API 获取元数据并下载 PDF，或复制本地文件
- `scripts/parse_pdf.py`：用 `pdfplumber` 提取全文，按章节正则切分
- 输出 `analyses/{name}/raw.md`：包含 YAML frontmatter（标题/作者/年份/arXiv ID）+ 8 个标准章节

**8 个标准章节**：Abstract / Introduction / Related Work / Method / Model Architecture / Loss Functions / Experiments / Ablations

### Stage 2 — 创新点分析 (`/analyze-innovations`)

读取 `raw.md`，参照 `prompts/innovations_system.md` 的精度要求，输出 `innovations.md`：

| 节 | 内容 |
|----|------|
| 1. Problem Statement | 核心问题、先验工作局限、一句话总结 |
| 2. Core Contributions | 每条贡献的定义、意义、创新性 |
| 3. Model Architecture | 高层设计、关键组件（含 tensor shape）、文字架构图 |
| 4. Loss Design | 完整 Loss 公式、各项解释、权重策略 |
| 5. Training Strategy | 数据集、优化器、LR schedule、batch size、训练技巧 |
| 6. Key Results | 对比 SOTA 的指标表、Ablation 洞察 |
| 7. Implementation Notes | 容易遗漏的细节、复现 Checklist |
| 8. Paper Significance | 影响力、局限性、后续方向 |

### Stage 3 — 代码复现 (`/reproduce-code`)

基于 `innovations.md` 生成完整 PyTorch（≥ 2.0）工程：

- **`config.py`**：所有超参数以 `@dataclass` 封装，值直接来自论文，推测值有注释标注
- **`model.py`**：每个子模块独立 `nn.Module`，forward 有 tensor shape 注释
- **`loss.py`**：返回 `{"total": ..., "primary": ..., "aux_*": ...}` dict，便于逐项分析
- **`dataset.py`**：匹配论文实验设置的数据加载流水线
- **`train.py`**：含 seed 固定、梯度裁剪、LR warmup、best/latest checkpoint

### Stage 4 — HTML 报告 (`/generate-report`)

输出两个文件到 `outputs/{name}/`：

**`summary.html`**：自包含深色主题报告页，无需联网即可浏览
- TL;DR 一句话摘要
- 可跳转的目录导航
- 贡献卡片（绿色左边框）、架构组件块（tensor shape 标签）、Loss 公式块（等宽字体、琥珀色边框）、训练参数表、指标对比表、复现 Checklist

**`model.py`**：从 `reproductions/{name}/model.py` 复制而来，删除所有本地 import，自包含运行

---

## Harness 设计说明

本项目基于 Claude Code Harness 工程化设计，核心机制：

| 机制 | 用途 |
|------|------|
| `.claude/commands/*.md` | 定义斜杠命令，每个文件是一条 Skill 的完整执行指令 |
| `.claude/settings.json` | 声明 Bash 权限白名单，避免每次工具调用都需要手动确认 |
| `prompts/*.md` | 每个阶段的系统提示词，Skill 在执行前读取，保证输出格式一致 |
| `_template/` 目录 | 各阶段输出的 Schema 参考，也是 Claude 生成内容时对齐的目标格式 |
| `session.log` | PreToolUse Hook 自动记录每次 Bash 调用，便于回溯研究过程 |

数据流是单向的：

```
papers/ → analyses/raw.md → analyses/innovations.md → reproductions/ → outputs/
```

`innovations.md` 是整个流水线的核心契约：Stage 3 和 Stage 4 都以它为唯一输入源。

---

## 命名约定

- **论文名（`{name}`）**：全小写、连字符分隔，最长 60 字符，例如 `attention-is-all-you-need`、`denoising-diffusion-probabilistic-models`
- 未指定 `{name}` 时，arXiv 论文自动从标题生成，本地 PDF 从文件名生成
- 同一 `{name}` 再次运行时，PDF 不重复下载，已有文件直接复用

---

## 新增目录说明

### `datasets/` — 数据集目录

每个数据集一个子目录，结构固定：

```
datasets/{dataset_name}/
├── raw/          原始下载文件（不进版本控制）
├── processed/    dataset.py 直接读取的预处理文件
└── splits/       train.txt / val.txt / test.txt（每行一个样本路径/ID）
```

`reproductions/{name}/dataset.py` 中的 `data_dir` 默认指向 `datasets/{dataset_name}/processed/`。

---

### `src/` — 基础代码库

所有论文复现共享的公共组件，通过 `sys.path` 导入：

```python
# reproductions/any-paper/train.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.logger import MetricLogger
from src.metrics.classification import ClassificationMetrics
```

| 模块 | 内容 |
|------|------|
| `src/base/base_model.py` | `BaseModel(nn.Module)` — `save/load/freeze/num_parameters` |
| `src/base/base_trainer.py` | `BaseTrainer` — 抽象 `train_step/eval_step`，具体训练循环 |
| `src/base/base_evaluator.py` | `BaseEvaluator` — 测试时推理 + 指标计算 + 结果保存 |
| `src/metrics/classification.py` | `ClassificationMetrics.compute_all()` → AUC, Accuracy, F1, Precision, Recall, AUC-PR |
| `src/metrics/regression.py` | `RegressionMetrics.compute_all()` → MSE, RMSE, MAE, R², MAPE |
| `src/utils/logger.py` | `MetricLogger` — 写 `metrics.jsonl` + `train.log` + TensorBoard（可选）+ CSV 导出 |
| `src/utils/checkpoint.py` | `CheckpointManager` — 保存 `ckpt_best.pt` / `ckpt_latest.pt`，按指标自动选 best |
| `src/utils/seed.py` | `set_seed(seed, deterministic=False)` |

---

### `logs/` — 日志与指标目录

每次训练运行自动创建子目录，命名为 `run_YYYYMMDD_HHMMSS`（或 `--run-name` 指定）：

```
logs/{paper_name}/{run_name}/
├── config.json         训练开始时保存的配置快照
├── metrics.jsonl       逐步指标，每行一条 JSON：
│                       {"step":100,"epoch":1,"train_loss":0.42,"val_auc":0.85}
├── train.log           带时间戳的人类可读文本日志
├── metrics.csv         训练结束时从 JSONL 导出（便于 Excel / pandas 分析）
├── ckpt_best.pt        验证集最优指标对应的 checkpoint
├── ckpt_latest.pt      最近一次 epoch 的 checkpoint
└── test_results.json   运行 test.py 后填写
```

**读取日志示例：**

```python
import json
import pandas as pd

# 方式 1：pandas 读 CSV
df = pd.read_csv("logs/my-paper/run_0/metrics.csv")
df.plot(x="step", y=["train_loss", "val_auc"])

# 方式 2：逐行读 JSONL
with open("logs/my-paper/run_0/metrics.jsonl") as f:
    rows = [json.loads(l) for l in f]
```

---

### `reproductions/{name}/train.py` + `test.py`

每个复现目录包含两个可运行脚本：

**train.py：**
```bash
python train.py                        # 默认配置
python train.py --lr 1e-3 --epochs 50 --run-name exp_warmup
```
- 日志自动写入 `logs/{name}/{run_name}/`
- 每 50 步记录 `train_loss`；每 epoch 记录 `val_loss`、`val_auc` 等完整指标
- 保存 `ckpt_best.pt`（val_loss 最低）和 `ckpt_latest.pt`

**test.py：**
```bash
python test.py --run-name exp_warmup       # 使用该 run 的 ckpt_best.pt
python test.py --checkpoint path/to.pt     # 指定 checkpoint 文件
python test.py --split val                 # 在验证集上评估
```
- 自动选取最新 run（不指定 `--run-name` 时）
- 输出格式化指标表，保存 `test_results.json`

---

## 依赖

| 依赖 | 用途 | 是否必需 |
|------|------|----------|
| `pdfplumber` | PDF 文本提取 | 推荐（fallback 使用系统 `pdftotext`） |
| `requests` | HTTP 下载 | 是（arXiv PDF 下载） |
| `numpy` | 指标计算 | 是 |
| `sklearn` | AUC-PR 等额外指标 | 可选（不安装时自动 fallback） |
| Claude Code CLI | 执行所有 Skill | 是 |
| PyTorch ≥ 2.0 | 运行复现代码 | 运行代码时需要 |

```bash
pip install pdfplumber requests numpy
pip install scikit-learn   # 可选，提升指标计算覆盖率
```
