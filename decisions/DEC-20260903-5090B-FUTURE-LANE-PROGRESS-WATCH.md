# 5090B：预先覆盖matched plain与DCLGAN的未来静默停滞风险

日期：2026-09-03

CUT和CycleGAN当前已有live progress sentinel，但CUT完成后才启动的matched plain，
以及matched plain完成后才启动的DCLGAN，原先只有PID/epoch-heartbeat健康检查。它们
各自还可能出现“trainer仍活着但不再前进”的盲区。

progress watcher v4把最大后代深度纳入不可变合同。它从既有successor PID出发，只
在限定深度内唯一命中冻结trainer命令片段时采样；目标尚未启动、进程正在工程门禁，
或出现多个匹配时均保持等待，不发送信号、不启动训练。这样可以跨越
`successor -> supervisor -> trainer`，同时避免把门禁子进程或bootstrap shell误认
为正式训练。

5090B已部署两个等待型watcher：PID 140048绑定未来matched plain，PID 140049绑定
未来DCLGAN。断开SSH后两者PPID均为1。当前分别等待CUT e200和matched plain e200，
没有改变后继顺序，没有读取性能指标，也没有影响CUT/CycleGAN共驻训练。
