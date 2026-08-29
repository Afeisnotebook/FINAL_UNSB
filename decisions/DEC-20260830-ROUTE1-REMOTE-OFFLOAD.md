# DEC-20260830：允许单机4090作路线一受控算力副本

状态：`AUTHORIZED_CONDITIONAL_PREFLIGHT`  
上游决策：`DEC-20260829-LOCAL-ROUTE1-ENGINEERING-GATE-PASS.md`

## 用户授权

用户在本地长期锚点运行期间明确提供 `192.168.0.30`，并授权由研究主控判断哪些
长程任务可以通过SSH外包。该明确授权只覆盖此单机受控副本，不恢复旧四服务器方案。
凭据不得写入Git、日志、contract或复现命令。

## 决定

本地GTX 1660轨迹继续作为当前canonical，不中断、不迁移、不与远端状态拼接。
允许RTX 4090在通过代码、manifest、选中图像内容、环境、共同e0、恢复和
zero-intervention门后，运行**完整的host-matched副本**：远端方法只能与同一远端、
同一e0的remote plain比较。

优先外包顺序为：

1. remote plain/HJ/HNEK长期proxy副本，用于提前获得完整e200状态和判断远端proxy；
2. 在结果身份无冲突时，运行远端内部matched的Phase C虚拟分支；
3. 候选公式与代码冻结后，运行远端candidate/plain matched副本。

DT仍受proxy门约束；新候选仍受长期因果图谱、derivation card和candidate gate约束。

## 不允许的混合

- 不用remote HJ/HNEK减去local plain；
- 不从local中途checkpoint切到remote后继续冒充同一轨迹；
- 不把4090数值与1660数值拼成一条晚期曲线；
- 不因remote更快而跳过local正在运行的canonical锚点；
- 不改变small25、seed2026、200 data epochs、固定milestone或CRN协议；
- 不打开confirmation20，不进入全量数据，不运行路线二/handoff/退出阈值；
- 不用remote短程结果选择算法或拟合控制信号。

若两台主机结论冲突，必须标记为hardware/environment interaction并做matched归因，
不得择优报告。远端结果只有在host内plain/method共同身份成立时才具有科学意义。

## 启动门

远端第一次科学运行前必须保存：

- hostname、GPU、driver、Python/PyTorch/CUDA及依赖版本；
- frozen commit、protocol fingerprint、manifest SHA256；
- small25训练文件和discovery70文件逐内容hash核对；
- 从本地已接受shared e0复制后的文件hash和scientific-state hash；
- remote plain twin、inactive HJ/DT、disabled HNEK、resume精确性；
- 守护进程、chunk、heartbeat、失败重试和日志位置；
- `confirmation20_opened=false`。

门禁未全部通过时，服务器只可作环境准备和工程测试，不得产生可报告的算法结果。

