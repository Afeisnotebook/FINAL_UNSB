# AM-TNC 溢出安全度量归约恢复门

受保护e178副本的只读重放在offset `6245`、全局zero-based update `1528679`、G/F player精确复现冻结故障。134个参数贡献中131个完全有限；三个G参数块的两个原始gradient、pre-step Adam scale、replica mean/difference和Adam缩放向量也全部有限。唯一首发非有限量是这些缩放向量在float32中的平方和交叉乘积。最大缩放向量已达约`1.38e22`，其平方超过float32范围。

因此，当前证据否定“删除环境直接导致故障”“checkpoint损坏”“原始gradient先NaN”和“Adam scale先NaN”。它确认的是冻结AM-TNC实现的数值归约缺陷，不是AM-TNC交换反对称机制或恢复性能的证伪。

批准实现以下最小修复：保持原float32路径不变；只有冻结路径计算出的全局几何量非有限时，才用float64重新计算同一个Adam度量内积。它不裁剪gradient、不改变scale、不改变投影公式、超参数、样本顺序、RNG或训练目标。普通有限输入必须与冻结实现逐位一致，overflow路径必须保持replica交换均值性质。

科学主lane暂不恢复。修复必须先从相同只读e178副本运行至少6250 updates，并满足：首次float64回退仍发生在offset 6245的G/F player；无异常；回放后模型和优化器状态全部有限；源checkpoint前后SHA相同；不读取paired性能或confirmation20。门禁通过后，是否迁移e178 full-state并恢复到e200需要新的source-bound裁决和Git提交。
