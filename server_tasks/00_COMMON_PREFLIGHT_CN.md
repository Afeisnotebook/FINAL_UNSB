# 四台服务器共同门禁

先读根目录 `START_HERE_CN.md` 和 `AGENTS.md`。服务器执行者只负责运行，不能
改变 lane、seed、数据 split、HJ 窗口或确认集状态。

1. clone 后固定到主控给出的 commit；不得在服务器上提交算法改动。
2. 复制 `environment/server.env.example` 为仓库根目录 `server.env`，填写绝对路径、
   本机逻辑 GPU 号和分配的 lane。
3. 执行 `bash scripts/bootstrap_server.sh "$PWD"`。
4. 执行 `bash scripts/server_preflight.sh server.env`。
5. 把 `reports/inbox/<LANE>_ENVIRONMENT.json`、`<RUN_ROOT>/<LANE>/E0_IDENTITY.json`
   和终端日志回传主控；不要开始长训。

主控必须确认四机的 commit、manifest hash、lane config hash、环境栈和 e0 network
hash。跨卡 e0 网络 hash 必须相同；环境不同或数据 hash 不同均禁止授权。

长训结束后先运行 `bash scripts/evaluate_all_milestones.sh server.env`，再运行：

`python tools/prepare_server_return.py --lane <LANE> --run-root <RUN_ROOT> --inbox reports/inbox --output reports/returns/<LANE>`

只提交 `reports/returns/<LANE>` 到 `return/<LANE>` 分支。
