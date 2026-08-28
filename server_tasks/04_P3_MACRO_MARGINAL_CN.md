# 服务器 4：P3_MACRO_MARGINAL

在 `server.env` 中设置 `FINAL_UNSB_LANE=P3_MACRO_MARGINAL`。A 域和 B 域分别
独立均匀抽取，再在域内均匀抽图；同域时禁止相同 stem。它不是 DCUM，不条件化
B 域，不把域标签输入网络，推理与 plain 相同。

共同门禁与授权通过后执行 `bash scripts/run_lane.sh server.env`；中断只用 exact
epoch-boundary resume。
