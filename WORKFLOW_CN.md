# Git 主控—四服务器工作流

Git 保存“可审计状态”，不搬运数据和 checkpoint。主控仓库的 `main` 是唯一科学
协议来源；四台执行机固定同一 commit，各自在本机外部路径保存完整状态。

服务器只回传小文件：环境、identity、train summary、checkpoint sidecar、固定评估
JSON 和必要故障日志。训练和评估结束后运行 `tools/prepare_server_return.py`，把白名单
文件复制到 `reports/returns/<lane>/`；再用 `return/<lane>` 分支提交，由主控检查后
合并。禁止提交 `.pt/.pth`、图像、数据 view、heartbeat 或完整日志。checkpoint
另做云盘/快照备份，其 SHA256 已写入 sidecar。

运行阶段不通过对话修改配置。所有变化先形成 `decisions/DEC-xxxx-*.md`，提交后才
生效；读到 discovery 效果后不得回改 lane。若只是机器路径或凭据，写入 ignored
local 文件，绝不提交。

固定里程碑为 e1/e10/e25/e50/e100/e150/e200。e200 是选择点；前序点只判断轨迹、
反转和 plain 是否异常。候选冻结后才允许提交 `CONFIRMATION_UNLOCK.json`，确认集
只能用于一次冻结候选对 matched plain 的最终读取。
