# 实验一：中文分词对比实验

项目根目录是 `NLP/`，本实验位于 `Test1/`。下文所有命令、输入路径和输出路径均相对于 NLP 根目录。Python 3.10 以上，仅使用标准库，无第三方依赖。

## 功能与结构

```text
NLP/
├── README.md
└── Test1/
    ├── data/                 # 原始语料、train.jsonl、test.jsonl
    ├── dictionary.py         # 训练词典、FMM、RMM、BiMM
    ├── hmm.py                # BMES HMM 训练、对数 Viterbi、保存/加载
    ├── llm.py                # Prompt、Few-shot、在线 API、校验/重试/缓存
    ├── bpe.py                # 字符 BPE 训练、合并规则、子词切分、保存/加载
    ├── evaluate.py           # micro P/R/F1、IV/OOV 召回、OOV 占比、Fertility
    ├── config.py             # 统一加载 NLP 根目录 .env
    ├── corpus.py             # 编码、清洗去重、审计、固定 80/20 划分
    ├── storage.py            # UTF-8 JSON/JSONL 输出
    ├── main.py               # 统一入口与结果摘要
    ├── requirements.txt      # 当前无第三方依赖
    ├── check_modules.py      # 词典分词等人工样例检查
    ├── test_experiment.py    # 算法独立对照与无网络 API/流程测试
    ├── verify.py             # 真实结果的独立计数核验
    ├── run.ps1               # 离线运行脚本
    ├── REVIEW.md             # 按课程要求的审查记录
    └── results/              # 模型、预测、指标、错误、差异案例
```

`documents/`、`history/` 保留原样，不属于本次生成的结果。

## 运行离线实验

在 NLP 根目录执行：

```powershell
python Test1/test_experiment.py
python Test1/main.py
python Test1/verify.py
```

也可以执行 `./Test1/run.ps1`。默认跑 FMM、RMM、BiMM、HMM、BPE，使用同一个完整测试集，不发网络请求。

显式路径与参数示例：

```powershell
python Test1/main.py --segmented "Test1/data/1998人民日报（分词）.txt" --pos "Test1/data/1998人民日报（带词性标注）.txt" --output Test1/results --seed 2026 --bpe-vocab-size 8000 --bpe-merges 500
```

`--output` 相对于 NLP 根目录，重复运行覆盖同名结果。固定划分写入 `Test1/data/`，并在输出目录保存同名副本。`verify.py` 核验默认 `Test1/results/`。如改变种子，后续各方法应一起重跑，不混用旧结果。

## 在线大模型分词

已实现兼容 Chat Completions 的 JSON HTTP 接口。服务商需支持 `/chat/completions` 与 `model/messages` 请求格式；不同服务和模型的兼容性需要实际联调。

默认已配置 DeepSeek 官方地址 `https://api.deepseek.com` 和模型 `deepseek-flash`，官方地址使用非思考模式 `thinking.type=disabled`。在 NLP 根目录 `.env` 中配置：

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
LLM_API_KEY=你的实际密钥
```

在 NLP 根目录执行 `python Test1/main.py --with-llm --llm-limit 100` 即可。项目统一通过 `Test1/config.py` 自动加载根目录 `.env`，无需在 PowerShell 另设变量。文件中的配置优先，缺少的项再回退到进程环境变量，默认服务/模型保持 DeepSeek Flash。也支持 `DEEPSEEK_API_KEY`，建议只配置一种密钥变量。

支持空行、注释、单/双引号值、可选 export 前缀和 UTF-8 BOM；不执行命令、不展开变量，不将配置写入进程环境。格式错误只报告行号，不输出配置内容。实际 `.env` 已被根目录 `.gitignore` 排除；`.env.sample` 可作为配置模板。

默认不设置 temperature，以兼容不支持该参数的模型；支持时可显式传 `--llm-temperature 0`。

Few-shot 使用训练集前 3 条长度在 8–80 字符的样本，测试文本只有去分词空格后的原文，标准分词不发送给模型。Prompt 要求返回 JSON 词数组并逐字保留原文。拒绝空词、非字符串、解释性文本及遗漏/修改字符；校验失败或瞬时网络错误最多重试 2 次，不用词典预测冒充模型输出。HTTP 认证等非瞬时错误直接报错。

在线默认从固定测试集按种子抽 100 行（不筛掉长文本），所有离线方法也在同一子集重新测时与评测。完整测试集共有 3,799 行；需要全量在线比较可传 `--llm-limit 3799`。子集结果位于 `Test1/results/llm_comparison/`，不得直接与离线全测试集指标比较。

每条成功响应缓存于 `Test1/results/llm_cache/`，缓存键包括服务地址、模型、Prompt、Few-shot、参数和测试原文；复跑会复用。`run_config.json` 记录在线任务是否完成；仅成功跑完整个所选子集才生成本次在线汇总指标。失败时可修正配置重跑，已成功请求可从缓存恢复。

`request_info.jsonl` 区分缓存命中与真实请求；`network_seconds` 仅统计本次实际请求，包含其重试等待。缓存中的历史耗时不会计入本次网络耗时。在线总耗时包括缓存 I/O、网络和重试，不能解读为纯模型推理速度。离线方法在同一子集的分词耗时另行实测。

**当前状态：DeepSeek Flash 已完成 100 条真实在线实验。六种方法在同一子集上完成比较。历史结果见 reference_results/，同学需使用自己的 API 密钥重跑；模拟测试不会访问网络。**

DeepSeek 配置依据：[DeepSeek 官方 API 文档](https://api-docs.deepseek.com/) 与 [Chat Completions 参数](https://api-docs.deepseek.com/api/create-chat-completion/)。

通用接口参考：[Chat Completions 官方参考](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。

## 语料检查和防止泄漏

- 分词版 4,804,499 字节，严格 GBK 解码和回编码一致；UTF-8、GB2312 严格解码失败。词性版 10,215,829 字节，严格 UTF-8 解码和回编码一致。两份各 19,484 行，哈希见 `results/manifest.json`。
- 样本单位为原始非空行，可能是标题或段落，不把它们称为文章或单句。只规范分隔空白，保留标点、数字、全半角及原始字形。
- 删除 1 个空行、490 条重复文本，余 18,993 条。重复按拼接后的完整文本判断，划分前处理；若同文本存在不同分词标注则隔离整组，本次无此类冲突。
- 使用 `random.Random(2026)` 打乱，训练取 `floor(0.8*N)`：15,194 行训练、3,799 行测试。完整文本交集为零；同一文章不同段落仍可能跨集合，不声称文章级隔离。
- 词典 49,520 词、最大词长 22，全部由训练集建立。HMM 仅用训练集词标注；BPE 仅用训练原文；Few-shot 仅用训练集。测试标准边界只用于评测和事后分析。
- 词性版包含 `词/词性` 和 `[中央/n ... 电台/n]nt`，用于去组合标记后的逐行审计，不用于补词。共 7,921 行与分词版不一致，其中 1,843 行文本相同但边界不同、6,078 行文本不同。统一用分词版金标准，不混合标注。

## 算法与指标口径

FMM、RMM 每次选最长词，匹配失败回退单字。BiMM 依次优先词数少、单字词数少，仍相同则取 FMM；规则固定，不在测试集调参。

HMM 用词标注转换为 BMES；监督统计初始、转移、字符发射概率，使用 alpha=0.1 加法平滑并约束合法状态转移。未知字符使用训练得到的平滑概率，不能读取测试频次。Viterbi 用对数概率及回溯，终止状态限定 E/S；模型保存在 `hmm_model.json`。

BPE 从训练原文的字符开始，以加权相邻对频次学习合并。无监督预切分规则在训练和推理一致：连续汉字、连续字母数字及单独其他字符；不使用测试标准词边界。使用倒排索引与堆更新频次，并按字符序打破频次平局，结果可复现。默认至多 500 次合并、目标词表上限 8,000，本次实际 5,069 token、500 次合并；不保证达到目标词表大小。未见字符保留原字符，不变为丢失原文的 UNK。模型保存在 `bpe_model.json`。

词级评测比较每个词的完整半开区间 `[start,end)`，行内分别计算后累加：P=正确词数/预测词数，R=正确词数/标准词数，F1=2PR/(P+R)。包含标点，不仅比较切分点。OOV 以标准词是否出现于训练词典判断，OOV 召回按词次统计；另存 IV 召回和 OOV 占比。

BPE 的主要切分粒度指标为 `Fertility = token 总数 / 同批标准词总数`。额外输出其词边界 P/R/F1 方便诊断，但 BPE 目标并非恢复语言学词边界，不能只凭词级 F1 判断其下游语言模型价值。保留未知字符的覆盖能力与准确恢复 OOV 词边界也不是同一个指标。

## 真实运行结果与分析入口

完整测试集标准词 227,890 次，OOV 6,543 次（2.8711%）。

|方法|Precision|Recall|F1|OOV召回|
|---|---:|---:|---:|---:|
|FMM|91.8209%|94.7896%|93.2816%|2.1550%|
|RMM|92.0599%|95.0327%|93.5227%|2.1703%|
|BiMM|92.1027%|95.0244%|93.5407%|2.1550%|
|HMM|81.1469%|81.1786%|81.1628%|53.2783%|
|BPE|44.7665%|62.3849%|52.1273%|2.2620%|

BPE Fertility 为 1.3936。HMM 的总体 F1 低于训练词典方法，但 OOV 召回显著更高，体现字符状态建模的不同取舍。这是当前简单一阶 HMM 的实测结果，不能泛化为所有统计模型都低于词典方法。

最新训练时间、分词时间与吞吐量见自动生成的 `Test1/results/summary.md` 和 `metrics.json`，均为本机单次实测，训练、分词分别计时，分词不含评测与写文件；词典构建时间未计入训练时间列。

- `*_predictions.jsonl`：每条原文、标准词、预测、OOV 词和来源行号。
- `*_errors.jsonl`：所有错误；`*_typical_errors.jsonl`：有 OOV、无 OOV 各选最短 10 条，便于阅读，不代表类别分布。
- `direction_disagreement_cases.jsonl`：FMM/RMM 给出不同切分的真实测试案例，并附全部离线方法结果。方向分歧只是歧义候选，不能自动等同于已人工确认的语义歧义。
- `pos_audit.jsonl`、`annotation_conflicts.jsonl`：输入标注差异与冲突审计。

如“真情动人心”，标准为“真情 / 动 / 人心”，BiMM 为“真情 / 动人 / 心”，是已登录词的交叉歧义；“观雪”被词典算法切成单字，体现未登录词问题。可从各算法预测记录中核对 HMM 与 BPE 在相同案例的差异。

## 核验与提交范围

`test_experiment.py` 包含 11 项测试：Viterbi 与穷举最优路径对照、HMM 概率归一化、BPE 与朴素合并对照、模型保存/加载、零命中和无效边界评测、API 响应校验、模拟请求缓存、同子集在线流程、DeepSeek 默认地址/模型和请求参数。模拟测试不访问网络。

`verify.py` 独立重算真实预测的词边界计数与 F1，核对训练词频和集合交叉；`check_modules.py` 保留词典算法人工样例。所有数据与代码仍在 NLP 项目内，代码未写死 C 盘或 D 盘地址。

已完成实验源程序、完整离线运行和 100 条真实在线比较。五页以内报告可依据这些真实结果另行整理，视频按你的要求不制作。

## 在线失败与续跑

若模型改动原文，程序会将第一个不同字符的位置、应有字符和原文片段反馈给模型，最多重试两次；不读取标准词边界，不自动替换模型结果。成功缓存不删除。每条响应记录实际尝试次数，带纠错重试的实验应在报告中注明。

失败时 `.error.json` 保存响应与诊断，`llm_comparison/failure.json` 记录样本位置，`LLM_partial_predictions.jsonl` 保留已完成预测。重新执行相同命令会复用成功缓存；完整指标只在所选子集全部成功后生成。日志中 cache_hit=True 表示复用缓存，不代表本次再次发起网络请求。

## 从 Git 仓库复现

请先阅读 [REPRODUCE.md](REPRODUCE.md)。仓库保留原始语料和 reference_results/ 历史汇总；results/、派生训练/测试文件和 verification.txt 为本机生成输出，不跟踪。
