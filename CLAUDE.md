# Auto Research — Harness Workflow

Full pipeline: PDF/arxiv → structured innovation analysis → PyTorch code reproduction.

## Workflow

```
/parse-paper <pdf_path|arxiv_url> [name]
      ↓  papers/{name}.pdf + analyses/{name}/raw.md
/analyze-innovations <name>
      ↓  analyses/{name}/innovations.md
/reproduce-code <name>
      ↓  reproductions/{name}/{model,loss,train,config,dataset}.py  (PyTorch)
/generate-report <name>
      ↓  outputs/{name}/summary.html + outputs/{name}/model.py

# Or run all four at once:
/auto-research <pdf_path|arxiv_url> [name]
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
│       └── auto-research.md   # 4-stage orchestrator
├── papers/                    # input PDFs (auto-saved by parse-paper)
├── analyses/
│   ├── _template/             # blank template for reference
│   │   ├── raw.md             # extracted paper text + sections
│   │   └── innovations.md     # structured innovation analysis
│   └── {paper_name}/          # one folder per paper
├── reproductions/
│   ├── _template/             # blank PyTorch code template
│   └── {paper_name}/          # one folder per paper
│       ├── model.py           # architecture (PyTorch)
│       ├── loss.py            # loss functions
│       ├── train.py           # training loop
│       ├── config.py          # hyperparameters
│       ├── dataset.py         # data loading
│       └── README.md          # reproduction notes
├── outputs/                   # ← HTML reports (primary deliverable)
│   ├── _template/             # blank HTML + model template
│   │   ├── summary.html       # dark-theme self-contained HTML
│   │   └── model.py           # standalone PyTorch model
│   └── {paper_name}/          # one folder per paper
│       ├── summary.html       # full innovation analysis report
│       └── model.py           # standalone PyTorch model (no local imports)
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
│   └── utils.py               # shared helpers
└── prompts/
    ├── parse_system.md        # section extraction prompt
    ├── innovations_system.md  # innovation analysis prompt schema
    ├── reproduce_system.md    # code generation instructions
    └── html_report_system.md  # HTML design rules
```

## Skills Quick Reference

| Skill | Input | Output |
|-------|-------|--------|
| `/parse-paper` | PDF path or arxiv URL | `analyses/{name}/raw.md` |
| `/analyze-innovations` | paper name | `analyses/{name}/innovations.md` |
| `/reproduce-code` | paper name | `reproductions/{name}/` (PyTorch) |
| `/generate-report` | paper name | `outputs/{name}/summary.html` + `model.py` |
| `/auto-research` | PDF path or arxiv URL | all four stages |

## Setup

```bash
pip install pdfplumber requests arxiv
```

## Conventions

- **Paper name** (`{name}`): lowercase-hyphenated, e.g. `attention-is-all-you-need`
- `raw.md` preserves original text; `innovations.md` is Claude's structured analysis
- Code in `reproductions/` targets PyTorch ≥ 2.0, single-GPU training by default
- Each reproduction folder is self-contained and runnable
