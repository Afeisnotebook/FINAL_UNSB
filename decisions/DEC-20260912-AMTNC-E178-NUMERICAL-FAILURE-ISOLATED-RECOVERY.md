# AM-TNC e178 数值故障与隔离恢复裁决

## 裁决

4090A 的错误清理确实留下了持续风险：原 `unsb_cov` 环境不再完整，原训练监督器仍依赖一个 deleted inode，旧前缀不能作为恢复权威。但是，它没有造成已完成训练状态丢失，也没有切断后续恢复路径。

原 AM-TNC trainer PID 3446758 在 e178 之后触发 `AM-TNC replica geometry is nonfinite` 的 fail-closed 检查并退出。监督器从 e178 完整状态自动恢复出 PID 3645974；新进程直接使用已隔离、只读并登记哈希的 Python。随后独立执行的 verify-only 门重新校验了 111/111 个运行时映射文件、解释器、Git、协议与 e178 checkpoint，结果为 `PASS_RUNTIME_AND_CHECKPOINT_RECOVERY_PREFLIGHT`。

因此，当前处置是不重建或覆盖旧环境，不触碰正在运行的恢复进程，把隔离运行时设为唯一恢复权威，并继续观察 e178→e179 是否重复出现相同数值故障。

## 损失边界

- e178 完整 checkpoint、sidecar、RNG、sampler 和方法状态仍然存在并通过哈希验证；已完成 epoch 损失为 0。
- 故障前尚未落盘的 e179 局部进度被重做，最坏小于一个 data epoch，即最多 8,552 个 update。
- e178 状态已复制到本地离机目录并重新核对源、目标 SHA256；最坏回退点由 e175 前移到 e178。
- 当前不是 e200 结果，也不能据此裁决 AM-TNC 机制。若同一状态反复出现同一非有限几何，才将其写为 `current_implementation_numerically_blocked`，不能写成机制证伪。

## 与独立证明审计的关系

同卡的独立证明审计由用户明确授权，并已自然完成，没有被中断。它与故障时间接近，但目前没有因果证据，因此仅记录为运行时混杂，不把它解释为 AM-TNC 数值故障的原因。

## 后继保障

AM-TNC 的 source-bound exporter、export recovery supervisor，以及四条统一评估/最终交付的 pinned-runtime control supervisor 均仍存活。即使仍使用 deleted inode 的等待子进程退出，相应 supervisor 也能从已固定的统一评估隔离运行时恢复。后续不得用旧前缀直接启动新任务。

整个处置没有修改算法、batch、AMP、TF32、数据顺序、优化器、训练协议或 paired 控制；confirmation20 继续封存。
