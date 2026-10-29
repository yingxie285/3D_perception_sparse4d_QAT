# 单帧 HBM 长时间无输出的诊断补丁

本补丁修复的是观测不足，不宣称已经解决 HBM 卡住的根因。旧版本在模型 IO 打印之后，到任务创建、提交、等待和释放之间均无日志，不能从现有日志判断停在哪个 API。

## 应用

将 sparse4d_J6M_diagnostic_patch.zip 上传到服务器的 basic_samples/code 目录。先在旧推理终端 Ctrl+C 停止旧任务，再从 code 目录执行：

~~~bash
unzip -o sparse4d_J6M_diagnostic_patch.zip
bash build_x86.sh
cd sparse4d_J6M
LIMIT_SECONDS=600 bash diagnose_x86.sh
~~~

默认诊断预算是 600 秒，超时由外部 GNU timeout 终止进程，并在必要时强制结束。这不是模型正常执行时间的上限结论。没有在 UCP 任务仍运行时主动释放张量内存，也没有将 UCP 超时伪装成成功。

输出为 output_diagnostic/run.log。开头应显示 diagnostic revision 2，确保安装并执行的是新程序。每个关键 API 都打印 ENTER 和 RETURN，并立即刷新 stderr。最后一个没有对应 RETURN 的 API 是本次观察到的停留位置。

| 最后一条日志 | 下一步调查 |
|---|---|
| write hbm_interface.json ENTER | 输出目录、磁盘或网络文件系统 |
| bind inputs ENTER | demo 的名称/shape 映射与数据复制 |
| hbUCPMallocCached ... ENTER | UCP 内存分配 |
| pack input[n] ENTER | demo 的该输入打包处理 |
| input cache clean ... ENTER | UCP 缓存处理 |
| hbDNNInferV2 ENTER | 任务创建，不应当误称为已开始 BPU 运算 |
| hbUCPSubmitTask ENTER | 提交接口内部；运行时可能在这里做初始化或同步工作 |
| hbUCPWaitTaskDone ENTER | 正在等待任务，仍需区分慢仿真和异常等待 |
| hbUCPReleaseTask ENTER | 任务释放，旧程序也会在此之后才打印 inference 耗时 |
| output unpack / decode / visualization ENTER | 对应 demo 读取、解码或绘图阶段 |

如果程序在输出读取之前停留，后处理和可视化尚未执行，不应先通过改置信度或 NMS 解决。

## 与官方工具对照

新版 --dump_inputs 会额外保存 output_diagnostic/packed_inputs/input_0.bin 到 input_6.bin。这些是量化后的真实内存布局，包含 padding，与原来的 inputs/*.bin 逻辑 FP32 数据不同。

SDK 自带 hrt_model_exec 的 read_binary_file 在 bin 文件长度等于 alignedByteSize 时直接读取；本补丁据此导出用于对照的缓冲区。只有输入打包阶段已完成，才能执行下面命令：

~~~bash
# 指定你服务器上同一 SDK 的 x86 可执行文件，不要选 aarch64 版
export HRT_MODEL_EXEC=/实际路径/x86/bin/hrt_model_exec
LIMIT_SECONDS=600 bash compare_hrt_x86.sh
~~~

此脚本沿用 demo 的同一组动态库、同一 model.hbm、同一份量化输入，运行官方 infer 命令，日志在 output_hrt_diagnostic/run.log。

- 官方工具完成而 demo 未完成：优先审查 demo 的任务管理、缓存、内存和运行参数；继续比对两边配置。
- 两者都未完成：问题至少不专属于 demo 的画图/后处理，应继续查共同的输入、HBM、UCP 和 x86 仿真环境；不能仅据此断言 HBM 文件损坏。
- 同一 HBM 在 J6M 完成而两个 x86 程序均未完成：重点检查 x86 仿真路径。此前板端只有随机输入 perf 的结果，不等于真实输入功能已经验证。

单纯 CPU 有活动不能证明推理正确。诊断超时也不能证明死锁。需要结合最后 API、同输入官方工具对照和必要时的线程栈。

## 本地验证

已使用真实 SDK 头文件完成新版 main.cc 的 C++17 语法检查。新增回归测试覆盖用户日志中的全部 7 个输入和 2 个输出的 dtype、shape、stride 和 alignedByteSize，包括 image 每行 704 个有效字节但 stride 为 768、五维 distortion、零维 time_interval、S32 分类和 F16 框输出。测试通过。

这证明张量布局辅助函数在这些契约下通过本地测试，不证明服务器 UCP 推理已经成功。当前没有远程服务器运行权限/连接，不能替用户取得运行时调用栈或完成 HBM 推理验收。

