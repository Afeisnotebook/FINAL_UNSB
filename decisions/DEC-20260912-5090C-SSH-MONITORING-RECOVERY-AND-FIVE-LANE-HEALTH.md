# DEC-20260912：恢复5090C无密码监控入口并核查五条活动lane

## SSH监控入口

本次刷新发现5090C的BatchMode登录返回`Permission denied`。这不是Proposal训练故障，但会让
两小时Goal heartbeat无法无人值守读取该宿主。通过用户已经授权的交互凭据登录后，只把现有
FINAL_UNSB route1 ED25519公钥补入`/root/.ssh/authorized_keys`，保持`.ssh/authorized_keys`
权限为`700/600`；本地新增`final-unsb-5090c`别名并继续启用StrictHostKeyChecking。

修复后BatchMode登录和hostname核对通过，公钥标记只出现一次。任何密码都没有写入Git、SSH
config、automation或运行回执。Proposal supervisor/trainer PID `9729/766090`在修复前后未变，
已自然完成e165，说明此操作没有重启、迁移或修改训练。

## 全局只读状态

- 4090A AM-TNC：e169，原PID与deleted-inode边界不变，隔离恢复guard健康、零重启；
- 5090A ST-CGR：e152，训练与增量导出健康；
- 5090B matched plain：e80，训练与增量导出健康；
- 5090C Proposal：e165，训练、guard和导出恢复链健康；
- 本地DCLGAN：e144，继续独占GTX1660。

刷新时还确认本地terminal-audit旧child PID `1392`已在读取原子替换中的incremental import set
时遇到一次瞬时`PermissionError`。冻结control supervisor PID `5180`按合同自动重启为PID
`31284`；累计restart count为2。失败文件当前可读取，新child和combined health PID `9960`
均健康，已完成的三份plain target-blind审计未丢失。因为自动恢复已经闭环，本次没有手动重启
或改写等待链。

两小时Goal heartbeat已同步5090C别名、新PID与恢复事实。所有检查保持metric-blind，未读取
中间paired性能、未改变训练队列、未打开confirmation20。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_5090C_SSH_MONITORING_RECOVERY_AND_FIVE_LANE_HEALTH_20260912T080036.json`。
