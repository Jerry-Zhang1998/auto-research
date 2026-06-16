# Auto Research

一套基于 Claude Code Harness 的算法论文全自动研究工作流，输入一篇论文（PDF 或 arXiv 链接），自动完成：

1. **论文解析** — 提取全文结构，识别章节，提取图片（含架构图）
2. **创新点梳理** — 结构化分析问题、架构、Loss、训练策略、实验结果，并自动查找官方 GitHub 仓库
3. **代码复现** — 优先使用官方 GitHub 代码适配，或从头生成完整可运行 PyTorch 工程
4. **HTML 报告** — 输出自包含深色主题摘要 + 训练曲线 + 评估指标可视化
5. **自动修复** — 训练出错时自动读取日志、定位问题代码、对照论文打补丁、验证通过

整个流程通过 Claude Code 的 Skill（斜杠命令）驱动，无需任何外部服务，本地即可运行。

---

## 效果预览

对任意一篇论文运行 `/auto-research` 后，得到以下产物：

```
papers/
└── attention-is-all-you-need.pdf

analyses/attention-is-all-you-need/
├── raw.md                        # 原文结构（8 个标准章节 + YAML frontmatter）
├── innovations.md                # 创新点深度分析（Section 0 含 GitHub URL）
├── figures/                      # 从 PDF 提取的图片
│   ├── manifest.json             # 图片索引（位置、标题、评分）
│   ├── fig_001.png               # 最高分 = 架构图（自动识别）
│   └── fig_002_page3.png         # 其余提取图片
└── _official_repo/               # 官方 GitHub 仓库（如论文中提供链接则自动克隆）
    └── ...                       # 原始代码，model.py/loss.py 作为复现主要来源

outputs/attention-is-all-you-need/reproduction/
├── config.py                     # 所有超参数（值直接来自论文）
├── model.py                      # 完整模型架构（PyTorch）
├── loss.py                       # Loss 函数（返回 dict，便于消融）
├── dataset.py                    # 数据加载与预处理
├── train.py                      # PaperTrainer(BaseTrainer) + CLI 入口
├── test.py                       # BaseEvaluator 推理 + 指标输出
└── README.md                     # 复现说明、预期指标、已知差距

logs/attention-is-all-you-need/run_20260615_143022/
├── config.json                   # 运行配置快照
├── metrics.jsonl                 # 逐步指标（每行一条 JSON）
├── train.log                     # 带时间戳的文本日志
├── metrics.csv                   # 训练结束后导出
├── ckpt_best.pt                  # 最优验证指标 checkpoint
├── ckpt_latest.pt                # 最近 epoch checkpoint
└── test_results.json             # test.py 运行后填写

outputs/attention-is-all-you-need/
├── html/
│   ├── summary.html              # 完整创新点报告（含架构图，浏览器直接打开）
│   ├── train.html                # 训练曲线可视化（loss / AUC / LR）
│   └── evaluate.html             # 评估报告（ROC、PR 曲线、混淆矩阵）
└── reproduction/
    ├── config.py                 # 所有超参数（值直接来自论文）
    ├── model.py                  # 完整模型架构（PyTorch）
    ├── loss.py                   # Loss 函数
    ├── dataset.py                # 数据加载
    ├── train.py                  # PaperTrainer(BaseTrainer) + CLI 入口
    ├── test.py                   # BaseEvaluator 推理 + 指标输出
    └── README.md
```

---

## 安装

**依赖：Claude Code CLI**（需已安装并登录）

```bash
# Python 依赖
pip install pdfplumber requests arxiv pymupdf numpy

# 可选：提升分类指标计算覆盖率
pip install scikit-learn

# PyTorch（运行复现代码时需要）
pip install torch torchvision
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

完成后打开报告：

```bash
open outputs/attention-is-all-you-need/html/summary.html
```

### 分步执行

```bash
# Stage 1：解析论文 + 提取架构图
/parse-paper https://arxiv.org/abs/1706.03762 attention-is-all-you-need

# Stage 2：分析创新点（自动查找 GitHub URL）
/analyze-innovations attention-is-all-you-need

# Stage 3：生成 PyTorch 复现代码（优先使用官方 GitHub 代码）
/reproduce-code attention-is-all-you-need

# Stage 4：生成 HTML 报告 + 可视化
/generate-report attention-is-all-you-need
```

### 运行复现代码

```bash
cd outputs/attention-is-all-you-need/reproduction

# 训练
python train.py                            # 默认配置
python train.py --lr 1e-3 --epochs 50 --run-name exp_01

# 测试（自动使用最新 run 的 ckpt_best.pt）
python test.py --run-name exp_01

# 生成训练曲线 + 评估可视化
python3 ../../../scripts/generate_viz.py \
    --log-dir ../../../logs/attention-is-all-you-need/exp_01 \
    --output-dir ../../../outputs/attention-is-all-you-need/html
```

### 训练出错时自动修复

```bash
# 自动读取日志 → 定位错误 → 对照论文修复 → 验证，最多重试 5 次
/fix-reproduction attention-is-all-you-need

# 指定 run 和最大重试次数
/fix-reproduction attention-is-all-you-need exp_01 3
```

---

## Skill 速查

| Skill | 输入 | 输出 |
|-------|------|------|
| `/parse-paper` | PDF 路径 或 arXiv URL | `analyses/{name}/raw.md` + `figures/` |
| `/analyze-innovations` | 论文名 | `analyses/{name}/innovations.md`（含 GitHub URL） |
| `/reproduce-code` | 论文名 | `outputs/{name}/reproduction/`（优先用官方代码） |
| `/generate-report` | 论文名 | `outputs/{name}/html/{summary,train,evaluate}.html` + `outputs/{name}/reproduction/model.py` |
| `/auto-research` | PDF 路径 或 arXiv URL | 以上四个阶段全部执行 |
| `/fix-reproduction` | 论文名 \[run\] \[次数\] | 自动打补丁直到代码可运行 |

---

## 项目结构

```
auto-research/
│
├── .claude/
│   ├── settings.json                  # Harness 配置：权限白名单 + 会话日志 Hook
│   └── commands/                      # Skills（斜杠命令定义）
│       ├── parse-paper.md             # Stage 1：论文解析 + 图片提取
│       ├── analyze-innovations.md     # Stage 2：创新点分析 + GitHub 查找
│       ├── reproduce-code.md          # Stage 3：代码复现（GitHub 优先）
│       ├── generate-report.md         # Stage 4：HTML 报告生成
│       ├── fix-reproduction.md        # 运行时错误自动修复循环
│       └── auto-research.md           # 四阶段编排器
│
├── prompts/
│   ├── parse_system.md                # 章节提取规则
│   ├── innovations_system.md          # 创新点分析精度要求（Round 2 系统分析师指导）
│   ├── math_specialist_system.md      # Round 1 数学专项提取指导（符号保真 + 推导逻辑）
│   ├── reproduce_system.md            # 代码生成规范（形状注释、命名等）
│   └── html_report_system.md          # HTML 设计规则（配色、架构图嵌入）
│
├── scripts/
│   ├── fetch_paper.py                 # arXiv 下载 或 本地 PDF 复制
│   ├── parse_pdf.py                   # PDF → 结构化 JSON（章节 + 图表）
│   ├── extract_figures.py             # PyMuPDF 图片提取 + 架构图评分
│   ├── generate_viz.py                # metrics.jsonl + test_results.json → HTML
│   ├── fetch_repo.py                  # GitHub 仓库克隆 + 结构分析
│   ├── parse_errors.py                # Python traceback 解析 → JSON
│   └── utils.py                       # 共享工具函数
│
├── papers/                            # 缓存的 PDF 文件
│
├── analyses/
│   ├── _template/
│   │   ├── raw.md                     # 原文提取的 8 节标准结构（参考模板）
│   │   └── innovations.md             # 创新点分析结构（含 Section 0: Repository）
│   └── {paper_name}/
│       ├── raw.md
│       ├── innovations.md
│       ├── figures/                   # 提取的图片 + manifest.json
│       └── _official_repo/            # 官方 GitHub 代码（/reproduce-code 克隆）
│
├── outputs/
│   ├── _template/
│   │   ├── html/                      # HTML 报告模板（Skills 生成时对齐）
│   │   │   ├── summary.html
│   │   │   ├── train.html
│   │   │   └── evaluate.html
│   │   └── reproduction/              # 代码模板（Skills 生成时对齐）
│   │       ├── config.py
│   │       ├── model.py
│   │       ├── loss.py
│   │       ├── dataset.py
│   │       ├── train.py               # 薄 BaseTrainer 子类示例
│   │       └── test.py                # BaseEvaluator 用法示例
│   └── {paper_name}/
│       ├── html/                      # 所有 HTML 报告
│       │   ├── summary.html           # 创新点报告（含架构图 base64 内嵌）
│       │   ├── train.html             # 训练曲线（loss / 指标 / LR）
│       │   └── evaluate.html          # ROC、PR 曲线、混淆矩阵
│       └── reproduction/              # 完整可运行复现代码
│           ├── config.py              # 所有超参数（来自论文）
│           ├── model.py               # 模型架构
│           ├── loss.py                # Loss 函数
│           ├── dataset.py             # 数据加载
│           ├── train.py               # PaperTrainer(BaseTrainer)
│           ├── test.py                # BaseEvaluator 推理
│           └── README.md
│
├── datasets/
│   └── {dataset_name}/
│       ├── raw/                       # 原始下载文件
│       ├── processed/                 # dataset.py 直接读取的预处理文件
│       └── splits/                    # train.txt / val.txt / test.txt
│
├── src/                               # 所有复现共享的基础代码
│   ├── base/
│   │   ├── base_model.py              # BaseModel(nn.Module) — save/load/freeze
│   │   ├── base_trainer.py            # BaseTrainer — 训练循环骨架
│   │   └── base_evaluator.py          # BaseEvaluator — 推理 + 指标 + 结果保存
│   ├── metrics/
│   │   ├── classification.py          # AUC, AUC-PR, Accuracy, F1, Precision, Recall
│   │   └── regression.py              # MSE, RMSE, MAE, R², MAPE
│   └── utils/
│       ├── logger.py                  # MetricLogger → metrics.jsonl + train.log + CSV
│       ├── checkpoint.py              # CheckpointManager → ckpt_best / ckpt_latest
│       └── seed.py                    # set_seed(seed, deterministic=False)
│
└── logs/
    └── {paper_name}/
        └── {run_name}/
            ├── config.json
            ├── metrics.jsonl
            ├── train.log
            ├── metrics.csv
            ├── ckpt_best.pt
            ├── ckpt_latest.pt
            └── test_results.json
```

---

## 各阶段详解

### Stage 1 — 论文解析 (`/parse-paper`)

- 接受 arXiv URL / arXiv ID / 本地 PDF 路径
- `fetch_paper.py`：从 arXiv API 获取元数据并下载 PDF，或复制本地文件
- `parse_pdf.py`：用 `pdfplumber` 提取全文，按章节正则切分为 8 个标准节
- 输出 `analyses/{name}/raw.md`：YAML frontmatter（标题/作者/年份/arXiv ID）+ 正文

**架构图提取（`extract_figures.py` / PyMuPDF）：**

- 提取 PDF 中所有嵌入式光栅图片（PNG/JPEG xref）
- 对只有矢量绘图的页面（ML 论文常见）进行整页光栅化渲染
- 按 Caption 关键词（architecture / framework / overview / pipeline…）、图编号、宽高比自动评分
- 得分最高者认定为架构图。其 base64 写入 sidecar 文件 `figures/arch_b64.txt`，**不写入 `raw.md`**（base64 blob 高达 150K+ tokens，会污染所有下游 skill 的上下文）；`raw.md` frontmatter 只记录图片路径与 caption。`summary.html` 生成时由脚本直接从 PNG 读取并内嵌

---

### Stage 2 — 创新点分析 (`/analyze-innovations`)

读取 `raw.md`，扫描全文中所有 `github.com` URL，输出 `innovations.md`。

分析分为两轮：**Round 1（数学专项）** 和 **Round 2（系统分析）**。

**检测扫描（Step 1c）**：在两轮分析前，先快速扫描论文，判断以下各维度是否存在：

| 维度 | 判断依据 |
|------|---------|
| `theoretical_analysis` | 包含 theorem/proof/lemma/convergence/bound/"we prove" 等 |
| `model_architecture` | 提出具名组件、有架构图/伪代码、描述 tensor 维度 |
| `loss_design` | 提出超出标准 CE/MSE 的自定义损失函数 |
| `training_strategy` | 非标准训练过程、多阶段、训练 trick 是贡献的一部分 |
| `efficiency_analysis` | 报告 FLOPs/参数量对比、"X× faster"、复杂度分析 |
| `ablation_study` | 包含消融实验表、"ablation"/"effect of" 等 |
| `implementation_notes` | 独立的实现细节章节、初始化策略、工程级细节 |

检测结果写入 `innovations.md` YAML frontmatter 和 `## Paper Profile` 节，**仅输出存在的章节**。

**输出 sections（仅存在的才写入）：**

| 节 | 条件 |
|----|------|
| **0. Repository** | 始终 |
| **1. Problem Statement** | 始终 |
| **Theoretical Analysis** | `theoretical_analysis = true` — 理论框架、核心推导步骤、定理/命题、假设条件、理论→设计连接 |
| **2. Core Contributions** | 始终 |
| **3. Model Architecture** | `model_architecture = true` |
| **4. Loss Design** | `loss_design = true` |
| **5. Training Strategy** | `training_strategy = true` |
| **Efficiency Analysis** | `efficiency_analysis = true` — 复杂度、参数量、吞吐量、显存、可扩展性 |
| **6. Key Results** | 始终 |
| **7. Implementation Notes** | `implementation_notes = true` |
| **8. Paper Significance** | 始终 |

**两轮分析流程（`theoretical_analysis = true` 时）：**

```
Round 1 — Math Specialist（prompts/math_specialist_system.md）
  目标：数学内容高保真提取
  ├── 只读论文的数学密集段落（定理/推导/附录）
  ├── 符号与论文完全一致，禁止改写或简化
  ├── 提取 T.1–T.4，生成 {THEORY_DRAFT}
  └── T.5 强制留空（placeholder）— 禁止在此 pass 填写

Round 2 — Systems Analyst（prompts/innovations_system.md）
  目标：架构、Loss、训练的系统级分析 + 跨 pass 综合
  ├── 分析完 Section 3（架构）和 Section 4（Loss）之后
  ├── T.1–T.4：从 THEORY_DRAFT 校验复制
  └── T.5：在此写成 —— 具体设计决策 + 对应定理 + 逻辑连接
```

T.5 是两轮分析的核心产出：它要回答"这个设计选择背后的理论依据是什么，改掉它会破坏什么"。

**Theoretical Analysis 节的核心子节：**
- `T.1 Theoretical Framework` — 论文依赖的理论框架（变分推断、信息论、PAC 学习等）
- `T.2 Core Claim / Main Result` — 主定理或核心命题（直接引用论文符号）
- `T.3 Key Derivation Steps` — 2-4 个最关键推导步骤（命名每步数学技法）
- `T.4 Assumptions & Conditions` — 理论成立的条件（标注 standard / restrictive / implicit）
- `T.5 Theory → Design Connection` — 理论结果如何直接驱动模型设计或 Loss 选择（Round 2 综合产出）

Section 0 的 GitHub URL 会传递到 Stage 3，驱动官方代码优先策略。

---

### Stage 3 — 代码复现 (`/reproduce-code`)

**Step 0 — GitHub 优先策略：**

先读取 `innovations.md` Section 0 的 GitHub URL。若存在：

```bash
python3 scripts/fetch_repo.py https://github.com/author/repo analyses/{name}/
# → 克隆到 analyses/{name}/_official_repo/（shallow clone）
# → 自动识别 model.py / loss.py / train.py 等关键文件
# → 返回 JSON，包含各文件前 80 行预览
```

- **`model.py`、`loss.py`**：以官方实现为主要来源，数学逻辑不变，只适配 config dataclass 和类型注解
- **`train.py`、`test.py`**：始终使用 BaseTrainer / BaseEvaluator 包装，不直接使用官方训练脚本
- **无 GitHub URL**：完全从 `innovations.md` 生成

**生成的 7 个文件：**

| 文件 | 内容 |
|------|------|
| `config.py` | 所有超参数以 `@dataclass` 封装，值直接来自论文 |
| `model.py` | 每个子模块独立 `nn.Module`，forward 含 tensor shape 注释 |
| `loss.py` | 返回 `{"total": ..., "primary": ..., "aux_*": ...}` dict |
| `dataset.py` | 匹配论文实验设置的数据加载流水线 |
| `train.py` | `PaperTrainer(BaseTrainer)` 子类，只覆写 `train_step` / `eval_step` |
| `test.py` | `BaseEvaluator` 推理，自动选最新 checkpoint，保存 `test_results.json` |
| `README.md` | 快速启动、文件说明、预期指标、已知差距 |

**生成后校验（Step 4.5，不可跳过）：** 代码生成后立即 `python3 -m py_compile *.py` 检查语法，再用一个极小 config + dummy tensor 跑一次 model forward + loss（断言无 NaN）。语法/shape bug 在生成阶段就被抓住，而非拖到 `/fix-reproduction`。仅依赖真实磁盘数据的 dataset/IO 代码豁免此校验。

`train.py` 的使用方式：

```bash
python train.py                        # 默认配置
python train.py --run-name exp_01      # 命名 run（日志写入 logs/{name}/exp_01/）
python train.py --lr 5e-4 --epochs 100
```

`test.py` 的使用方式：

```bash
python test.py --run-name exp_01       # 使用该 run 的 ckpt_best.pt
python test.py --checkpoint /path/to.pt
python test.py --split val             # 在验证集上评估
```

---

### Stage 4 — HTML 报告 (`/generate-report`)

输出到 `outputs/{name}/html/` 和 `outputs/{name}/reproduction/`：

**`html/summary.html`** — 自包含深色主题报告页，无需联网即可浏览
- Paper Profile 栏：论文类型、领域，以及各节存在/缺失的彩色标签（仅存在的节可点击跳转）
- TL;DR 摘要、动态目录（只列存在的节）
- 架构图（从 PDF 提取的 base64 PNG 直接内嵌）
- **Theoretical Analysis 节**（条件渲染）：理论框架卡片、推导步骤块（双色边框、等宽字体）、假设列表、Theory→Design 连接
- **Efficiency Analysis 节**（条件渲染）：四格指标卡（复杂度、参数量、吞吐量、显存）
- 贡献卡片、架构组件块（tensor shape 标签）、Loss 公式块、训练参数表、指标对比表、复现 Checklist

**`reproduction/`** — 完整可运行复现代码（由 Stage 3 生成），包含 config.py、model.py、loss.py、dataset.py、train.py、test.py、README.md

**`html/train.html`** — 训练过程可视化（由 `generate_viz.py` 从 `metrics.jsonl` 生成）
- Loss 曲线（train / val）
- 验证指标曲线（AUC / Accuracy / F1）
- 学习率 schedule 曲线

**`html/evaluate.html`** — 测试集评估报告（由 `generate_viz.py` 从 `test_results.json` 生成）
- 指标卡片（AUC / Accuracy / F1 / Precision / Recall）
- ROC 曲线、PR 曲线
- 混淆矩阵热图

> 所有图表使用原生 Canvas API 绘制，无 CDN 依赖，离线可用。

---

### 自动修复 (`/fix-reproduction`)

训练脚本出错时，运行：

```bash
/fix-reproduction {name}
/fix-reproduction {name} exp_01       # 指定 run
/fix-reproduction {name} exp_01 3     # 最多重试 3 次
```

**内部流程（最多 N 次循环）：**

```
1. 定位最新 train.log（或先触发一次运行拿到报错）
2. parse_errors.py → 提取最后一个 traceback，定位用户代码帧
3. 读取报错文件 + innovations.md 对应节（model→Section 3 / loss→Section 4 / …）
4. 按错误类型打最小补丁（ImportError / 形状不匹配 / TypeError / CUDA 设备 / …）
   原则：只修复实现 bug，不改动算法数学逻辑
5. timeout 90s 验证运行
6. 无错误 → 报告成功；有错误 → 进入下一轮
```

同一位置重复出错 2 次以上时，提示根因在更上游并停止，给出 debug 建议。

---

## `src/` 基础代码库

所有复现工程通过 `sys.path.insert` 共享这些组件：

```python
# outputs/{paper_name}/reproduction/train.py 首行
import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.base.base_trainer import BaseTrainer
from src.utils.logger import MetricLogger
from src.metrics.classification import ClassificationMetrics
```

| 模块 | 关键接口 |
|------|---------|
| `src/base/base_trainer.py` | `BaseTrainer.fit(train_loader, val_loader)` — 抽象 `train_step` / `eval_step` |
| `src/base/base_evaluator.py` | `BaseEvaluator.evaluate(loader)` → `{metrics, curves, confusion}` |
| `src/base/base_model.py` | `BaseModel.save/load/freeze/num_parameters` |
| `src/metrics/classification.py` | `ClassificationMetrics.compute_all(targets, logits)` → AUC, AUC-PR, Accuracy, F1, Precision, Recall |
| `src/metrics/regression.py` | `RegressionMetrics.compute_all(targets, preds)` → MSE, RMSE, MAE, R², MAPE |
| `src/utils/logger.py` | `MetricLogger.log_metrics(step, epoch, **metrics)` — 写 JSONL + 文本 + TensorBoard |
| `src/utils/checkpoint.py` | `CheckpointManager(monitor="val_loss", mode="min")` — 自动保存 best / latest |
| `src/utils/seed.py` | `set_seed(42, deterministic=False)` |

**`BaseTrainer` 接口约定：**

`train_step` 和 `eval_step` 必须返回包含以下键的 dict：

```python
{
    "loss":    torch.Tensor,   # 必须，用于反向传播和日志
    "logits":  torch.Tensor,   # 可选，分类任务的 logit 输出
    "targets": torch.Tensor,   # 可选，与 logits 配套，用于自动计算指标
}
```

---

## `logs/` 日志格式

每次训练运行生成一个子目录，命名为 `run_YYYYMMDD_HHMMSS`（或 `--run-name` 指定）：

```
logs/{paper_name}/{run_name}/
├── config.json          训练开始时的配置快照
├── metrics.jsonl        逐步指标，每行一条：
│                        {"step":100,"epoch":1,"train_loss":0.42,"val_auc":0.85}
├── train.log            带时间戳的人类可读文本日志
├── metrics.csv          训练结束时从 JSONL 导出
├── ckpt_best.pt         验证集最优指标对应的 checkpoint
├── ckpt_latest.pt       最近一次 epoch 的 checkpoint
└── test_results.json    test.py 运行后填写，包含 metrics + ROC/PR 曲线数据 + 混淆矩阵
```

读取日志：

```python
import json, pandas as pd

# 方式 1：CSV
df = pd.read_csv("logs/my-paper/exp_01/metrics.csv")
df.plot(x="step", y=["train_loss", "val_auc"])

# 方式 2：JSONL
with open("logs/my-paper/exp_01/metrics.jsonl") as f:
    rows = [json.loads(l) for l in f]
```

---

## `datasets/` 数据集目录

每个数据集一个子目录，结构固定：

```
datasets/{dataset_name}/
├── raw/          原始下载文件（不进版本控制）
├── processed/    dataset.py 直接读取的预处理文件
└── splits/       train.txt / val.txt / test.txt（每行一个样本路径或 ID）
```

`outputs/{name}/reproduction/dataset.py` 中的 `data_dir` 默认指向 `datasets/{dataset_name}/processed/`。

---

## Harness 设计

| 机制 | 用途 |
|------|------|
| `.claude/commands/*.md` | 每个文件定义一条 Skill 的完整执行逻辑 |
| `.claude/settings.json` | Bash 权限白名单，避免每次工具调用都需要手动确认 |
| `prompts/*.md` | 各阶段系统提示词，Skill 执行前读取，保证输出格式一致 |
| `_template/` 目录 | 各阶段输出的 Schema 参考，Claude 生成内容时对齐此格式 |
| `session.log` | PreToolUse Hook 自动记录每次 Bash 调用 |

数据流是单向的：

```
papers/pdf  →  analyses/raw.md  →  analyses/innovations.md
                    ↓ (figures/)        ↓ (github URL)
          outputs/{name}/html/      analyses/_official_repo/
             summary.html                   ↓
                                    outputs/{name}/reproduction/
                                            ↓
                                       logs/{name}/{run}/
                                            ↓
                               outputs/{name}/html/train.html
                               outputs/{name}/html/evaluate.html
                               outputs/{name}/reproduction/model.py
```

`innovations.md` 是整个流水线的核心契约：Stage 3、Stage 4、`/fix-reproduction` 均以它为主要参考来源。

---

## 依赖

| 依赖 | 用途 | 是否必需 |
|------|------|----------|
| `pdfplumber` | PDF 文本提取 | 是 |
| `requests` | HTTP 下载 | 是（arXiv PDF 下载） |
| `arxiv` | arXiv API 元数据 | 是 |
| `pymupdf` | 架构图提取（光栅 + 矢量渲染） | 推荐 |
| `numpy` | 指标计算 | 是 |
| `scikit-learn` | AUC-PR 等额外指标 | 可选（不安装时自动 fallback） |
| `git` | 克隆官方 GitHub 仓库 | 可选（无 URL 时不需要） |
| Claude Code CLI | 执行所有 Skill | 是 |
| PyTorch ≥ 2.0 | 运行复现代码 | 运行代码时需要 |

```bash
pip install pdfplumber requests arxiv pymupdf numpy
pip install scikit-learn   # 可选
pip install torch torchvision  # 运行复现代码时
```

---

## 命名约定

- **论文名（`{name}`）**：全小写、连字符分隔，最长 60 字符
  - 例：`attention-is-all-you-need`、`denoising-diffusion-probabilistic-models`
- 未指定 `{name}` 时，arXiv 论文从标题自动生成，本地 PDF 从文件名生成
- 同一 `{name}` 再次运行时，PDF 不重复下载，已有 `_official_repo/` 执行 `git pull` 更新
