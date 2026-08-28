# 服务器 2：P1_HJ_HANDOFF

在 `server.env` 中设置 `FINAL_UNSB_LANE=P1_HJ_HANDOFF`。HJ 窗口由实际 train
identity 数换算：`[1.6,8.0)` data epoch；8553 张时为 `[13685,68424)` updates。
窗口结束只关闭 HJ correction，G/F/D/E、优化器、scheduler、Adam moments、RNG
全部原地连续，不重置、不载入 plain。

共同门禁与授权通过后执行 `bash scripts/run_lane.sh server.env`；中断只用 exact
epoch-boundary resume。
