# 四台 DGX Spark：Ring / 1M / 六路 max 思考

[英文部署步骤](README.md) · [实测数据](results/README.md) · [来源与许可证](../../NOTICE.md)

本项目是 Tech2Wild / Kai 公开方案的 fork。我们保留上游历史，并在 `profiles/ring-1m/` 中发布四机直连环、较大的 FP8 KV 池和 max 思考验收。上游根目录的历史测试属于上游设备，不能算作我们的成绩。

## 已运行的服务配置

四台 128 GB DGX Spark，按 A—B—C—D—A 环形直连，TP4，DP1。每台本地 NVMe 保存完整权重和 Engram 表；管理网络负责 SSH 和初始化协调，模型数据通信使用 RoCE。

- 上下文上限：1,048,576 token，输入和输出合计。
- 调度并发：6；默认 thinking ON，reasoning effort=max。
- KV：每机 16 GiB，整个 TP4 实例报告 4,909,644 个逻辑 token 槽位。
- 缓存格式：`fp8_ds_mla`，包含 448 维 FP8 和 64 维 BF16 RoPE；索引器 FP8。没有 FP4 收益混入这些数字。
- 分块 prefill：4096；DSpark：5 个 draft token；CUDA Graph 开启。
- 工具调用与图片输入开启；已验证一次真实的双工具往返和一张合成图片。

**六路并发不等于六路同时住满 1M。** 当前逻辑池约为 4.68 × 1M，也不能再乘四。若为输出预留 262,144 token，输入最多约 786,432 token，还要计入对话模板、工具定义和图片消耗。输出预算不代表测试实际生成了那么多 token。

## 我们增加的内容

主要内存修改将 DeepSeek V4.1 索引器的临时 gather 缓冲从 `40 × 最大上下文` 改成 `min(40, 最大并发) × 最大上下文`。C6/1M 下该分配理论节省约 4.38 GiB/机。元数据分块和实际算子使用同一个上限，KV 数值与运算精度没有变化；非 DeepSeek 模型保持旧逻辑。

SSD Engram 与图捕获预备工作来自上游。这里把它们移植到固定的官方 0909 镜像，保留完整文件的全局行号与 rank 归属，并附上数值与图重放测试。当前仅支持 DP1 和单 ubatch。

## 复现顺序

1. 检查 Ring 邻接关系、RoCE 接口、各直连链路子网及 MTU 9000。不要凭机器排列猜物理端口，也不要把逻辑 HCA 数量当成物理口数量。脚本不会改写现有网络。
2. 将 `config/node.env.example` 复制为不提交的 `node.env`，填写本机路径、rank、管理地址和接口名。
3. 从魔搭获取完整官方 checkpoint。`config/modelscope-shards.json` 记录源版本，`config/checkpoint-shards.json` 记录 48 个分片的 SHA256。每机校验本地文件，避免通过 NFS 做实时 Engram 随机读。
4. 按[英文指南](README.md)构建固定镜像和 NCCL，四机同时运行通信数值检查。
5. 四机分别启动 `scripts/launch-node.sh`，检查所有 worker 日志以及主节点 `/health`、`/v1/models`。
6. 在空闲服务上依次运行 `validation/validate_max_1m.py` 和 `validation/validate_long_max.py`。长输入需要较长等待时间；所有新测试使用 ON/max、temperature=1、top_p=0.95。

公开脚本从实际运行版本中抽出了路径和设备参数，已做语法、文件一致性及相关函数检查，但尚未在另一套空白设备上重装复验。因此这里发布的是可审查、可复现的实验方案，不承诺任意环境一键成功。运行参数保持 `--restart no`，没有配置重启恢复和长期托管。

## 怎样理解吞吐

六个不同代码任务整批得到 75.22 token/s；真正六路同时生成的区间约 100.50 token/s。整批约 180 秒，其中约 109 秒只剩一到两路，尾部空闲会拉低总平均。思考 token 已计入输出，不是漏算了思考。

公开上游最新 boot 10 的 code C6 为 225.5 token/s，但使用同一类短提示、thinking OFF、temperature=0、约 200 token 输出预算。我们的 max 思考、不同任务和长输出预算与之不同。尾部效应只能解释一部分差距，剩余部分仍需同条件测试，不能宣称整体比上游快。

详细数据、完成状态、逐请求时长和分析脚本在 [results](results/README.md)。长上下文测试是三个精确值的检索检查，不代表全面能力评估，也不是六路 1M 压力测试。官方混合精度 KV 自行适配已暂停，当前服务保留 FP8。
