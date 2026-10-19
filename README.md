# NLP 课程实验

本目录用于保存自然语言处理课程的实验代码、语料、运行结果和说明文档。

项目根目录为 `NLP/`；`Test1/` 是实验一子目录。以下命令和数据路径均相对于 NLP 根目录，不依赖固定盘符。

## 目录结构

```text
NLP/
├── README.md              # 实验目录总说明
└── Test1/                 # 实验一：中文分词（ChineseSegmentation）
    ├── README.md          # 详细运行说明、模块状态与阶段结果
    ├── data/              # 原始语料、固定训练集和测试集
    ├── dictionary.py      # FMM、RMM、BiMM 与训练词典
    ├── hmm.py             # HMM 训练与 Viterbi 解码
    ├── llm.py             # Prompt、Few-shot 与在线 API
    ├── bpe.py             # BPE 训练与子词切分
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

当前已完成：

- 检查真实编码与语料格式，清洗并去重。
- 使用固定随机种子，按约 80% / 20% 划分训练集和测试集。
- 仅从训练集构建词典，自行实现 FMM、RMM、BiMM。
- 按词的完整起止边界计算 Precision、Recall、F1。
- 实现 BMES HMM 参数训练与 Viterbi 解码，训练集原文上的 BPE 词表学习。
- 实现在线大模型 Prompt、训练集 Few-shot、API 重试、缓存及严格输出校验。
- 保存完整预测、典型错误案例、方向歧义候选、Fertility 与语料审计记录。
- 核验训练与测试文本不重叠，词典词频只来自训练集。

HMM/Viterbi 与 BPE 已实现并在完整测试集实际运行；在线大模型客户端、缓存与校验已实现，并已完成 DeepSeek Flash 的 100 条真实在线实验。数据采用行级划分，未保证同一文章的不同段落完全隔离。

详细说明见 [实验一 README](Test1/README.md)。

## 运行实验一

环境要求：Python 3.10 或以上，仅使用标准库，无需安装第三方依赖。

在 PowerShell 中执行：

```powershell
# 在 NLP 项目根目录执行
python Test1/main.py
python Test1/verify.py
```

也可在 NLP 根目录执行 `./Test1/run.ps1`。自定义输出示例：`python Test1/main.py --output Test1/results_demo`；相对路径参数以 NLP 根目录为基准。

默认随机种子为 `2026`，输出保存到 `Test1/results/`。重复运行会覆盖同名结果文件；修改种子后应重新评测，并同步更新报告中的结果。

## 数据与后续实验约定

- 保留原始语料，预处理结果单独存放。
- 后续方法复用实验一已保存的训练集和测试集，保证比较使用统一文本。
- 训练词典、HMM 和 BPE 时仅使用训练集；大模型 few-shot 示例也只从训练集选取。
- API 密钥通过环境变量配置，不写入源码或提交到版本库。
- 新实验使用 `Test2/`、`Test3/` 等独立目录，并在各目录维护对应的运行说明。
## 统一配置

全项目使用根目录 `.env`，参考 `.env.sample`。在线入口自动加载，无需另设 PowerShell 环境变量；文件配置优先。运行：`python Test1/main.py --with-llm --llm-limit 100`。实际密钥不提交版本库。

## 给同学复现

先阅读 [复现指南](Test1/REPRODUCE.md)。原始语料随仓库保存，运行结果由程序重新生成；参考指标独立保存在 `Test1/reference_results/`。根目录 `.env.sample` 是占位模板，真实 `.env`、API 缓存及生成结果不提交。
