# 服务器 3：P2_HNEK

在 `server.env` 中设置 `FINAL_UNSB_LANE=P2_HNEK`。配置固定为
`gamma=.25 / residual / physical / all`，全程介入。它是历史 e200 正结果的清洁
全量复赛，不允许改 gamma 或选择早期最佳点。

共同门禁与授权通过后执行 `bash scripts/run_lane.sh server.env`；中断只用 exact
epoch-boundary resume。
