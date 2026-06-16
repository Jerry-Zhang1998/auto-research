# Auto Research — Harness Workflow

Full pipeline: PDF/arxiv → structured innovation analysis → PyTorch code reproduction.

## Workflow

```
/parse-paper <pdf_path|arxiv_url> [name]
      ↓  papers/{name}.pdf + analyses/{name}/raw.md
/analyze-innovations <name>
      ↓  analyses/{name}/innovations.md  (+ github URL if found in paper)
/reproduce-code <name>
      ↓  outputs/{name}/reproduction/  (PyTorch; uses official GitHub repo if available)
/generate-report <name>
      ↓  outputs/{name}/html/{summary,train,evaluate}.html + outputs/{name}/reproduction/model.py

# Or run all four at once:
/auto-research <pdf_path|arxiv_url> [name]

# If reproduction code fails at runtime:
/fix-reproduction <name> [run_name] [max_attempts]
      ↓  auto-diagnose error → patch code → verify → repeat up to N times
```

## Directory Layout

```
auto-research/
├── .claude/
│   ├── settings.json          # permissions, hooks
│   └── commands/              # skill files (slash commands)
│       ├── parse-paper.md
│       ├── analyze-innovations.md
│       ├── reproduce-code.md
│       ├── generate-report.md
│       ├── fix-reproduction.md  # runtime error auto-fix loop
│       └── auto-research.md   # 4-stage orchestrator
├── papers/                    # input PDFs (auto-saved by parse-paper)
├── analyses/
│   ├── _template/             # blank template for reference
│   │   ├── raw.md             # extracted paper text + sections
│   │   └── innovations.md     # structured innovation analysis (includes Section 0: Repository)
│   └── {paper_name}/          # one folder per paper
│       ├── raw.md             # extracted paper text
│       ├── innovations.md     # structured analysis (github URL in Section 0)
│       ├── figures/           # extracted PDF figures (manifest.json + fig_*.png)
│       └── _official_repo/    # official GitHub repo (cloned by /reproduce-code if URL found)
├── outputs/                   # ← HTML reports + reproduction code (primary deliverable)
│   ├── _template/             # reference templates (skills read these)
│   │   ├── html/              # HTML layout templates
│   │   │   ├── summary.html   # innovation analysis layout
│   │   │   ├── train.html     # training curve chart layout
│   │   │   └── evaluate.html  # ROC/PR/confusion layout
│   │   └── reproduction/      # code templates (skills align to this when generating)
│   │       ├── config.py      # dataclass hyperparameter template
│   │       ├── model.py       # BaseModel subclass template
│   │       ├── loss.py        # loss returning {"total":...} dict
│   │       ├── dataset.py     # Dataset + get_dataloader template
│   │       ├── train.py       # thin BaseTrainer subclass (~60 lines)
│   │       └── test.py        # thin BaseEvaluator usage (~50 lines)
│   └── {paper_name}/          # one folder per paper
│       ├── html/              # all HTML reports
│       │   ├── summary.html   # innovation analysis (self-contained dark HTML)
│       │   ├── train.html     # training curves (loss, AUC, LR) — from logs/
│       │   └── evaluate.html  # ROC, PR, confusion matrix — from test_results.json
│       └── reproduction/      # full runnable reproduction code
│           ├── config.py      # all hyperparameters from paper
│           ├── model.py       # paper architecture
│           ├── loss.py        # paper loss
│           ├── dataset.py     # paper dataset
│           ├── train.py       # PaperTrainer(BaseTrainer) + entry point
│           ├── test.py        # BaseEvaluator entry point
│           └── README.md
├── datasets/                  # 数据集目录（每个数据集一个子目录）
│   └── {dataset_name}/
│       ├── raw/               # 原始下载文件
│       ├── processed/         # dataset.py 直接读取的预处理文件
│       └── splits/            # train.txt / val.txt / test.txt
├── src/                       # 基础代码（所有 reproduction 可共享）
│   ├── base/
│   │   ├── base_model.py      # BaseModel(nn.Module) — save/load/freeze
│   │   ├── base_trainer.py    # BaseTrainer — 训练循环骨架
│   │   └── base_evaluator.py  # BaseEvaluator — 测试时推理
│   ├── metrics/
│   │   ├── classification.py  # AUC, Accuracy, F1, Precision, Recall, AP
│   │   └── regression.py      # MSE, RMSE, MAE, R², MAPE
│   └── utils/
│       ├── logger.py          # MetricLogger → metrics.jsonl + train.log + CSV
│       ├── checkpoint.py      # CheckpointManager → ckpt_best.pt / ckpt_latest.pt
│       └── seed.py            # set_seed(seed, deterministic=False)
├── logs/                      # 训练日志（每次运行一个子目录）
│   └── {paper_name}/
│       └── {run_name}/        # e.g. run_20260615_143022
│           ├── config.json    # 运行配置快照
│           ├── metrics.jsonl  # 逐步指标（每行一条 JSON）
│           ├── train.log      # 人类可读的文本日志
│           ├── metrics.csv    # 训练结束后导出
│           ├── ckpt_best.pt   # 最优验证指标对应的 checkpoint
│           ├── ckpt_latest.pt # 最近一次 epoch 的 checkpoint
│           └── test_results.json  # test.py 运行后填写
├── scripts/
│   ├── fetch_paper.py         # arxiv fetch or local PDF copy
│   ├── parse_pdf.py           # PDF → structured text
│   ├── extract_figures.py     # PyMuPDF figure extraction + arch scoring
│   ├── generate_viz.py        # metrics.jsonl + test_results.json → train/evaluate HTML
│   ├── fetch_repo.py          # clone GitHub repo + analyze structure → JSON
│   ├── parse_errors.py        # parse Python traceback from log → JSON
│   └── utils.py               # shared helpers
└── prompts/
    ├── parse_system.md          # section extraction prompt
    ├── innovations_system.md    # Round 2 (Systems Analyst) analysis guidance
    ├── math_specialist_system.md # Round 1 (Math Specialist) notation-fidelity extraction
    ├── reproduce_system.md      # code generation instructions
    └── html_report_system.md    # HTML design rules
```

## Skills Quick Reference

| Skill | Input | Output |
|-------|-------|--------|
| `/parse-paper` | PDF path or arxiv URL | `analyses/{name}/raw.md` |
| `/analyze-innovations` | paper name | `analyses/{name}/innovations.md` (with GitHub URL) |
| `/reproduce-code` | paper name | `outputs/{name}/reproduction/` (PyTorch; official repo if available) |
| `/generate-report` | paper name | `outputs/{name}/summary.html` + `model.py` + viz HTML |
| `/auto-research` | PDF path or arxiv URL | all four stages |
| `/fix-reproduction` | paper name [run] [attempts] | patches failing code until it runs |

## Setup

```bash
pip install pdfplumber requests arxiv pymupdf
```

## Conventions

- **Paper name** (`{name}`): lowercase-hyphenated, e.g. `attention-is-all-you-need`
- `raw.md` preserves original text; `innovations.md` is Claude's structured analysis
- Code in `outputs/{name}/reproduction/` targets PyTorch ≥ 2.0, single-GPU training by default
- Each reproduction folder is self-contained and runnable
