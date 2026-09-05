# DEC-20260906：4090A运行环境恢复与动态论文交付链接管

## 结论

4090A上配置的`/home/yc/anaconda3/envs/unsb_cov`目录曾意外缺失，而健康的AM-TNC训练仍在使用已经删除但尚未释放的运行时inode。该状态不会立即破坏训练，却会使进程一旦退出便无法按冻结命令恢复，也会阻断后续统一评估。现已在不中断、不重启AM-TNC的条件下，按原进程可核验的软件版本重建并验证该环境；AM-TNC监督器和训练PID保持不变。

动态论文交付链也已从当前`origin/main`的已审核runtime relation注册表部署。统一评估使用5090B fresh-e0 matched plain、5090C Proposal、5090B CUT和CycleGAN；ST-CGR使用已审核的5090B matched plain关系；最终交付等待这些结果以及AM-TNC同宿主matched结果。此前绑定已取消5090A plain或旧cohort的等待器已在新链连续健康后退出，避免未来重复交付。

## 恢复与验证

- 恢复环境固定为Python 3.10.20、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、NumPy 2.1.3、Pillow 11.3.0、LPIPS 0.1.4与clean-fid 0.1.35。
- staging和正式路径各通过31项动态交付定向测试；共同e0和plain e100 checkpoint可读取；LPIPS Alex缓存权重与Clean-FID导入通过。
- 环境另外保留同inode hardlink恢复快照，防止同类目录级事故再次切断恢复路径。
- 这次重建只恢复未来fail-closed resume和评估所需的同版本软件，不把4090A重新声明为新的训练runtime cohort。当前AM-TNC继续使用原进程；只有真实故障时才允许按原冻结命令从完整状态精确恢复。

## DCLGAN等待链修复

DCLGAN统一评估子进程始终健康，但它的恢复监督器因为上述运行时路径缺失而退出。环境恢复后，按固定控制commit重新部署监督器并收养原子进程，`restart_count=0`，没有重启评估子进程或创建重复交付。新的健康监视器连续通过两轮检查。

## 科学边界

本次没有读取性能数值，没有使用paired指标控制训练或调度，没有修改训练协议，没有打开confirmation20，没有迁移或重启健康训练，也没有删除数据、checkpoint或历史证据。动态链只在所有固定e200输入和严格runtime关系齐备后执行；等待不等于算法冻结或Goal完成。

完整机器回执见`evidence/paper_aio/PAPER_AIO_4090A_RUNTIME_RECOVERY_AND_DYNAMIC_DELIVERY_20260906T062831.json`。
