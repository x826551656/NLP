# 同学复现指南

所有命令都在克隆后的 NLP 仓库根目录执行，不依赖盘符。实验代码在 Test1/。

## 1. 环境和数据

安装 Python 3.10 以上，建议使用已验证的 Python 3.12。项目全部使用标准库，无需 pip 安装。在线请求使用系统 HTTPS 证书。

仓库包含老师提供的两份原始语料，必须保留原字节和编码；不要用编辑器另存改变编码。程序严格按分词版 GBK、词性版 UTF-8 解码。原始文件 SHA-256 见 reference_results/offline_manifest.json。

可选创建虚拟环境（Windows）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. 无需密钥的离线复现

```powershell
python Test1/test_experiment.py
python Test1/check_modules.py
python Test1/main.py
python Test1/verify.py
```

输出到 Test1/results/；固定划分还写到 Test1/data/train.jsonl 和 test.jsonl。以上生成文件均被 Git 忽略，不会覆盖参考结果。默认随机种子 2026，清洗后 18,993 条，训练 15,194 条、测试 3,799 条。词典只从训练集建立；HMM、BPE 同样仅用训练集。

预期离线 F1：FMM 0.932816292、RMM 0.935226789、BiMM 0.935407303、HMM 0.811627902、BPE 0.521272520。BPE Fertility 1.393562684。耗时随机器、Python 版本和运行负载变化，不要求一致。核验脚本独立检查训练词频、无重复交叉和词边界指标。

## 3. 在线 DeepSeek 复现

复制根目录 .env.sample 为 .env，再在本机填入自己的密钥（不要提交 .env）：

```powershell
Copy-Item .env.sample .env
```

如果已有 .env，不要再次复制覆盖。配置内容如下：

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
LLM_API_KEY=自己的实际密钥
```

执行：

```powershell
python Test1/main.py --with-llm --llm-limit 100
```

这会产生真实 API 请求及服务商计费。程序从根目录 .env 自动加载配置，固定抽取同一批 100 条测试文本，所有方法在该子集上公平评测。模型设置为非思考模式；Few-shot 为训练集样例。若模型改动原文，反馈具体字符错误，最多纠错重试两次；不提供测试金标准答案。

在线响应具有随机性，deepseek-flash 也是服务商可更新的模型别名，因此不保证复现相同预测或指标。只能保证输入划分、抽样、Prompt 和处理规则一致。成功请求在本机缓存，重新执行同一命令可续跑；缓存、请求响应和诊断不进入版本库。

完成后确认 Test1/results/llm_comparison/run_config.json 的 status 为 complete。在线对比指标在同目录 metrics.json，不要与全测试集离线指标混比。部分缓存命中的运行时间不能作为全新 100 次请求的时间。

## 4. 提交内的参考结果

reference_results/ 保留原实验的离线和在线汇总、运行配置、源文件哈希及在线测试样本 ID，用来对照，不被程序覆盖。它们是历史实验记录，不代表同学已经运行。未提交模型、全量预测、错误日志、成功响应缓存或真实密钥。

更多算法说明、输入格式和结果解释见 README.md。课程资料 documents/ 和生成结果仍保留在原作者本机，只从 Git 跟踪范围移除，不删除磁盘文件。
