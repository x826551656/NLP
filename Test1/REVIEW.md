# 实验源程序审查

|课程要求|审查结果|实现位置|
|---|---|---|
|训练集 80% 自建词典、FMM/RMM/BiMM|已实现并实际运行；未使用外部词典|corpus.py、dictionary.py|
|HMM 训练与 Viterbi|已实现并实际运行；BMES 合法路径、对数概率、未知字符平滑|hmm.py|
|在线大模型 Prompt 与 Few-shot|已完成 100 条真实 DeepSeek Flash 在线实验|llm.py、main.py|
|BPE 或 WordPiece|已实现字符 BPE 并实际训练；同一预切分规则用于训练和测试|bpe.py|
|同一批中文文本|离线全量统一；在线子集全部方法统一重算|main.py|
|P/R/F1、OOV、Fertility|已实现，真实词边界计数可独立核验|evaluate.py、verify.py|
|歧义与错误分析|完整错误、典型错误、方向分歧候选已保存|results/|
|速度|分别保存训练、离线分词与在线请求耗时；区分缓存|main.py、llm.py|
|五页以内实验报告|本次聚焦源程序，已有结果摘要与说明；未生成最终报告|README.md、results/summary.md|
|三分钟视频|按用户要求不制作|—|

防泄漏：完整文本先去重再划分，词典/HMM/BPE 只读取训练样本，Few-shot 只从训练集选取。保留行级划分限制，不宣称文章级隔离。

已发现并修复：HMM/BPE/LLM 占位实现；评测零命中时 F1 除零；预测空词验证不足；在线子集缺少同文本速度比较。

不能把模拟 API 测试当作真实在线实验结果。已有真实在线指标，保存在 reference_results/online_metrics.json；与同批 100 条离线方法比较。
