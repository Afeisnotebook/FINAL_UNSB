# DEC-20260830：将单机 RTX 5090 纳入路线一的受控算力副本

状态：`AUTHORIZED_PREFLIGHT_PASS`  
上游决策：`DEC-20260830-ROUTE1-REMOTE-OFFLOAD.md`

## 用户授权与凭据边界

用户明确提供一台新 RTX 5090 服务器，并授权研究主控自行完成环境、数据、代码和长程
任务配置。登录凭据只用于交互式连接；不得写入 Git、脚本、日志、contract、复现命令
或 evidence。服务器在通过全部门禁前只能执行环境准备和工程测试。

## 三台主机的固定职责

1. 本地 GTX 1660 继续当前 canonical small25/e200 轨迹，不迁移、不重启、不改协议。
2. RTX 4090 继续已经启动的 host-matched plain/HJ/HNEK/DT 锚点和长期因果审计。
3. RTX 5090 建立独立的 host-matched 副本：先运行 plain/HJ/HNEK，proxy 校准后才运行
   DT；长期因果图谱产生并冻结新算法后，优先承担候选与本机 plain 的长程比较和
   seed 验证。

增加算力不改变北极星：任务仍是从 UNSB 与历史探针证据中推导长期自稳定、自消隐或
无偏的新算法，而不是重跑旧名字、搜索退出阈值或扩大超参网格。

## 5090 科学身份

- 锚点训练代码固定为 commit
  `0da2a37086cca5bc4ad4488bb07c53096a7152ed`；控制代码和候选代码分别记录自己的冻结
  commit，不允许运行中随 `main` 漂移。
- protocol fingerprint 固定为
  `b0786b222790b84379802996448b8a68b86d69a6892ea0cdc04670cfcb1fb9b2`。
- manifest SHA256 固定为
  `1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b`。
- 使用本地已接受的 exact shared e0；传输后必须同时核对文件 SHA256 和 scientific-state
  SHA256。
- 训练仍是 six-domain small25、batch1、seed2026、真实 200 data epochs；不会因为
  服务器上有全量数据而提前切换 full-data。
- discovery70 只用于固定 milestone 事后标注；confirmation20 保持封存。

## 启动门

第一次科学训练前必须保存并验证：

- 主机、GPU、driver、Python、PyTorch、CUDA、cuDNN 和依赖版本；
- 压缩包路径安全、数据磁盘空间和解压目录结构；
- selected small25 与 discovery70 共 570 个 identity、1140 个文件的大小和内容哈希；
- `confirmation_files_touched=0`；
- exact shared e0 的文件哈希与 scientific-state 哈希；
- CPU contract gate；
- plain twin、inactive HJ/DT、disabled HNEK、full-state resume 和 evaluation isolation GPU
  gates；
- durable supervisor、5-data-epoch chunk、heartbeat、失败重试、日志和磁盘余量。

任一项失败即不得产生可报告的算法结果。

## 跨主机裁决规则

- 只允许 `5090 method - 5090 plain`、`4090 method - 4090 plain` 和
  `1660 method - 1660 plain`；禁止跨主机相减。
- 不把不同主机 checkpoint 串接成一条轨迹；shared e0 仅用于共同初始身份。
- 主机间结论不一致时记录为 hardware/environment interaction，并做主机内 matched
  归因，不择优报告。
- 新候选的公式、source hash 和算法 fingerprint 必须先冻结；任何新 seed 或新主机结果
  都不得反向改算法。
- 5090 不独立进行 paired-guided 搜索，也不使用中间 PSNR 早停或挑最佳 checkpoint。

## Git 控制面

GitHub 保存决策、代码、配置、环境记录、compact evidence、derivation card、裁决和
复现命令。数据、checkpoint、完整日志和凭据留在各自服务器。每个阶段使用独立 commit：

1. 5090 preflight/gate；
2. host-matched anchors/proxy；
3. 因果图谱与 derivation freeze；
4. candidate gate 与 e200；
5. seed validation 与最终 `CANDIDATE.json`。
