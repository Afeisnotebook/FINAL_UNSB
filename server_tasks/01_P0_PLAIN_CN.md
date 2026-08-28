# 服务器 1：P0_PLAIN

在 `server.env` 中设置 `FINAL_UNSB_LANE=P0_PLAIN`。共同门禁通过、且主控已把
授权决定提交到指定 commit 后，执行 `bash scripts/run_lane.sh server.env`。

这是 matched plain，不能被暂停来迁就候选，也不能换成旧源码的 `train.py`。
中断后只允许设置 `FINAL_UNSB_RESUME=1` 并重跑同一命令。
