# DEC-20260912：增量审计 Windows 发布读取加固

## 裁决

本地 terminal audit 在读取另一个进程正在原子替换的 incremental import-set 时，Windows 曾
短暂返回 `PermissionError`。冻结 supervisor 已自动恢复，当前权威链仍为 supervisor/child
`5180/31284`，restart count为2，状态为`WAITING_FOR_PREFLIGHT_IMPORTS_OR_GPU`；既有三个plain
审计单元没有丢失。

这不是科学失败，但属于会造成无谓重启和尾部延迟的工程缺口。canonical reader现只对
`PermissionError`执行最多10次有界退避，总额外等待不超过2.25秒。`FileNotFoundError`、格式
错误、身份或哈希不一致仍立即fail closed，不会被重试掩盖。

同时，lane receipt和import-set都改为解析与SHA256绑定同一次成功读取的bytes，取消原先解析后
再次打开文件计算哈希的TOCTOU窗口。瞬时lane/import-set锁、永久锁以及相邻relay/recovery、
control、health、terminal audit共81项测试通过。

## 部署边界

修复提交为`f2a601a0e74d8a0000fe8e641f6c64bc532c3378`。当前活跃terminal audit仍由冻结checkout
`8d422dc56e8abbfb20550a02b44bf9749a67b59d`运行；为保留source identity，没有热替换、重启或
修改该健康等待链。现有supervisor仍能对瞬时故障自动恢复，canonical修复用于未来新部署或经
正式来源迁移门验证后的恢复版本。

两小时Goal heartbeat已绑定提交`f2a601a`和“不为采用修复而热切换健康child”的规则，保持
仅失败通知，且未持久化凭据。

本次未读取性能值、未改变训练或审计observable、未打开confirmation20。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_INCREMENTAL_AUDIT_WINDOWS_PUBLISH_READ_HARDENING_20260912T082500.json`。
