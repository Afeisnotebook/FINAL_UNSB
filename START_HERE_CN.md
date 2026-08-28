# 先从这里开始

## 你现在接手的是什么

这是一个最后期限明确的研究执行项目，不是开放式算法头脑风暴。

- 数据：六域物理语料共 9153 个 paired identity；训练严格不使用配对。
- 泄漏安全训练：排除每域 discovery80 + confirmation20 后共 8553 张/侧。
- 训练：200 个数据 epoch，batch size 1，即每 lane 1,710,600 updates。
- 算力：最多四张 RTX 4090 并行一周。
- 目标：在相同 clean canonical 和 seed=2026 下比较四条冻结 lane，得到唯一
  当前最优候选；若全部为负，也必须输出最优失败方向和原因。

## 你不需要重新判断的事情

- 直接激活官方 time branch 的 `TA_MINIMAL` 已在 matched e200 得到
  `-1.092 dB`、0/5 域正；不重开。
- KCK/path-consistency 已在共同 e5 锚点上让目标残差反向恶化；不重开。
- LBST/PTQ/DCUM/AEB、PCOA 及其当前修订没有持续收益；不占四卡名额。
- paired PSNR 不得成为训练输入、在线控制器或 HJ 退出依据。

## 四条 lane

1. `P0_PLAIN`：clean official pooled UNSB。
2. `P1_HJ_HANDOFF`：1.6 epoch 前 plain；1.6–8.0 epoch HJ；之后永久 native。
3. `P2_HNEK`：冻结 `gamma=.25/residual/physical/all`。
4. `P3_MACRO_MARGINAL`：A/B 域独立均匀采样，域内随机，严格无配对。

第四条不是 DCUM：B 不依赖 A 的域，生成器和推理均看不到域标签。

## 主控中心第一次需要做什么

1. 运行 `python tools/validate_contracts.py`。
2. 让用户补充四台服务器的 clone 路径、数据根目录、运行根目录和 GPU 编号；
   这些信息只写入不提交的 `SERVER_ASSIGNMENTS.local.json`。
3. 每台服务器先执行 `server_tasks/00_COMMON_PREFLIGHT_CN.md`，回传环境与 e0
   身份；四台 e0 不一致时禁止长训。
4. 用 `tools/create_run_authorization.py` 合并四份 e0 报告；工具只有在 manifest、
   protocol、runtime 和 e0 全部一致时才会生成授权。
5. 主控审阅并提交 `decisions/RUN_AUTHORIZATION.json` 后，四台服务器才运行各自
   lane。

不要先提出第五条算法，也不要把旧仓库整体迁入这里。
