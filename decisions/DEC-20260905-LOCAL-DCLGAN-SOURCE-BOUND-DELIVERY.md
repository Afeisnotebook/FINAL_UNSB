# 本地DCLGAN到4090A统一评估的持久交付链

日期：2026-09-05

控制链审计发现，本地GTX1660上的DCLGAN已经有e200长训监督器和source-bound exporter，
但旧统一导入/评估等待器绑定的是已经撤销的5090B来源。若不修复，约九天后会停在“本地
EXPORT_SET完成、统一评估无法取得checkpoint”的状态。

本次新增只用于传输的本地出站push relay。它等待全部固定e100/125/150/175/200 receipt，
逐项验证source commit、adapter fingerprint、manifest、checkpoint、sidecar及scientific-state
哈希；随后通过固定SSH host key连接4090A，再校验hostname和GPU UUID。文件以唯一part名
上传，远端SHA256一致后才rename，`IMPORT_LANE.json`最后发布。Windows绝对路径只作为不可变
来源provenance保留，不会在Linux评估机上被访问。密码只存在于恢复监督器继承的环境中，
不写合同、状态、日志或Git。

4090A上的DCLGAN固定评估器也改由恢复监督器持有。它同时等待本地导入和未来合法的dynamic
first-wave cohort；两者齐备前不申请共享GPU锁。导入完成后只评估固定五个epoch，e200仍是
论文主结果，不选最佳checkpoint。

本地push和远端evaluation均对等待子进程执行了一次故障注入，两者都由冻结命令自动恢复，
重新进入正确等待状态。测试没有终止或修改DCLGAN、AM-TNC及其他训练，也没有复制任何未完成
checkpoint。旧的静态等待器在确认`completed_evaluations=0`后才被替换。

这项修复只关闭未来控制面停滞，不改变DCLGAN算法、数据、优化协议或论文比较关系。它没有读取
paired性能，没有打开confirmation20，也没有让DCLGAN阻塞更早完成的核心论文组合。
