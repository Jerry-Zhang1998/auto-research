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
