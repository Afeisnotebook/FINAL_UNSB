# 新宿主接入前的物理GPU身份门

日期：2026-09-03

`:44804`端点重复事件说明，SSH host/port不是算力身份。进一步采集四台远端和本地GPU后
发现，三个AutoDL容器的`/etc/machine-id`完全相同，因此machine-id也不能单独证明物理
宿主不同。稳定主键必须是NVIDIA GPU UUID，hostname、machine-id、数据盘device/inode只作
辅助证据。

commit `c62b5a9bb98503232e7f2b3abe2106711e368d9b`新增：

- `configs/PAPER_AIO_HOST_IDENTITY_REGISTRY.json`，登记4090A、5090A/B/C和本地GTX1660的
  GPU UUID及辅助身份；
- `operations/paper_aio_host_identity_gate.py`，把端点分类为registered match、new physical
  candidate、duplicate endpoint或label collision；
- duplicate/label collision均fail closed并禁止由身份门放行长训；
- receipt不保存SSH凭据、不读训练指标、不接触checkpoint。

该门已通过bundle SHA256 `4ff318de...3d2402d`部署到`:44804`的独立只读checkout，并以
requested label `5090D`实跑。远端receipt SHA256为
`dc3b1587fa6d32e16bc301fc6323299a083bc75d19ff159d8b8301ef4714a7ea`，返回GPU UUID
`GPU-578d4047-4c22-8c6a-d216-0f7938e99194`、`registered_label=5090B`和
`DUPLICATE_ENDPOINT_OF_REGISTERED_HOST`。门禁后CUT/CycleGAN及matched-plain后继PID均
保持存活，未启动第三lane。

以后新端点必须先通过此门；`NEW_PHYSICAL_GPU_CANDIDATE`只证明算力未重复，仍需继续通过
代码、数据、runtime、resume、评估和方法授权门，不能直接启动科学长训。

本决定不改变任何在飞算法或运行时关系，不读取paired性能，不打开confirmation20。
