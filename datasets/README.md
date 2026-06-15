# datasets/

每个数据集单独一个子目录。目录名全小写、连字符分隔，与论文中的数据集名称对应。

## 目录结构

```
datasets/
└── {dataset_name}/
    ├── raw/            原始下载文件（zip、tar、原始 CSV/JSON 等）
    ├── processed/      预处理后的文件（可被 dataset.py 直接读取）
    └── splits/         划分文件（train.txt / val.txt / test.txt，每行一条样本 ID 或路径）
```

## 示例

```
datasets/
├── imagenet/
│   ├── raw/
│   │   └── ILSVRC2012_img_train.tar
│   ├── processed/
│   │   ├── train/   (解压后的类别子目录)
│   │   └── val/
│   └── splits/
│       ├── train.txt
│       └── val.txt
│
├── cifar10/
│   ├── raw/
│   │   └── cifar-10-python.tar.gz
│   └── processed/
│       ├── train_data.npy    [50000, 3, 32, 32]
│       ├── train_labels.npy  [50000]
│       ├── test_data.npy     [10000, 3, 32, 32]
│       └── test_labels.npy   [10000]
│
└── wmt14-en-de/
    ├── raw/
    └── processed/
        ├── train.en
        ├── train.de
        ├── val.en
        └── val.de
```

## 约定

- `dataset.py`（位于 `reproductions/{paper_name}/`）中的 `data_dir` 默认指向 `datasets/{dataset_name}/processed/`
- 原始文件放 `raw/`，不做版本控制（加入 .gitignore）；`processed/` 中的文件可视情况缓存
- `splits/` 中的划分文件用于控制可复现的 train/val/test 划分，每行一条路径或 ID

## .gitignore 建议

```
datasets/*/raw/
datasets/*/processed/*.npy
datasets/*/processed/*.bin
datasets/*/processed/*.arrow
```
