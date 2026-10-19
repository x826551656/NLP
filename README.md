# NLP 课程实验

本目录用于保存自然语言处理课程的实验代码、语料、运行结果和说明文档。

项目位置：`D:\desktop\NLP`。

## 目录结构

```text
NLP/
├── README.md              # 实验目录总说明
└── Test1/                 # 实验一：中文分词（ChineseSegmentation）
    ├── README.md          # 详细运行说明、模块状态与阶段结果
    ├── data/              # 原始语料、固定训练集和测试集
    ├── dictionary.py      # FMM、RMM、BiMM 与训练词典
    ├── hmm.py             # HMM 与 Viterbi 接口，待实现
    ├── llm.py             # Prompt 与 Few-shot；API 待接入
    ├── bpe.py             # BPE 接口，待实现
    ├── evaluate.py        # P/R/F1、OOV 召回、Fertility
    ├── main.py            # 统一运行与结果输出
    ├── requirements.txt   # 当前仅使用标准库
    ├── corpus.py          # 语料清洗、审计及划分
    ├── storage.py         # 结果存储
    ├── check_modules.py   # 模块样例核验
    ├── verify.py          # 数据隔离与结果核验
    ├── run.ps1            # Windows 运行脚本
    └── results/           # 词典、指标、预测及错误案例
```

## 实验一：中文分词

实验目标：比较词典分词、HMM 分词、在线大模型分词与 BPE 子词切分的表现。

当前已完成第一阶段：

- 检查真实编码与语料格式，清洗并去重。
- 使用固定随机种子，按约 80% / 20% 划分训练集和测试集。
- 仅从训练集构建词典，自行实现 FMM、RMM、BiMM。
- 按词的完整起止边界计算 Precision、Recall、F1。
- 保存完整预测、典型错误案例与语料差异审计记录。
- 核验训练与测试文本不重叠，词典词频只来自训练集。

HMM、在线大模型 API 和 BPE 尚未接入。数据采用行级划分，未保证同一文章的不同段落完全隔离。

详细说明见 [实验一 README](Test1/README.md)。

## 运行实验一

环境要求：Python 3.10 或以上，仅使用标准库，无需安装第三方依赖。

在 PowerShell 中执行：

```powershell
cd D:\desktop\NLP\Test1
python main.py --segmented "data/1998人民日报（分词）.txt" --pos "data/1998人民日报（带词性标注）.txt"
python verify.py
```

也可在该目录执行 `./run.ps1`。

默认随机种子为 `2026`，输出保存到 `Test1/results/`。重复运行会覆盖同名结果文件；修改种子后应重新评测，并同步更新报告中的结果。

## 数据与后续实验约定

- 保留原始语料，预处理结果单独存放。
- 后续方法复用实验一已保存的训练集和测试集，保证比较使用统一文本。
- 训练词典、HMM 和 BPE 时仅使用训练集；大模型 few-shot 示例也只从训练集选取。
- API 密钥通过环境变量配置，不写入源码或提交到版本库。
- 新实验使用 `Test2/`、`Test3/` 等独立目录，并在各目录维护对应的运行说明。