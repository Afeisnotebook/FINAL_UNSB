# 本地低算力门禁

本机 GPU 只回答“代码能否可信启动和恢复”，不回答四条 lane 谁更好。必须完成：

1. 对真实六域目录生成带内容 hash 的完整 manifest，并与 `DATA_CONTRACT.json`
   的 9153/8553/480/120 计数一致。
2. 物化只含链接的本地 smoke view；不得复制或改图。
3. 四条 lane 均完成真实图像上的 e0 和至少一次 optimizer update。
4. HJ smoke 必须跨过其按 smoke data epoch 换算的介入起点，证明不是只导入了代码。
5. plain 连续两 epoch 与 e1 保存后 resume 到 e2 的 network state hash 完全一致。
6. discovery 每域至少一张走通 deterministic CRN rollout；重复评估字节一致。
7. `pytest`、契约检查、Python compile 和 source tree hash 全部记录。

这些产物写入 `local_validation/`（不提交大 checkpoint），最终只把摘要、环境和 hash
写入 `evidence/local_preflight/`。任何 smoke PSNR 都没有科学解释权。
