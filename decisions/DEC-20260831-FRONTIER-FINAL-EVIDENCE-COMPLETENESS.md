# DEC-20260831：frontier终交付不得压缩长期证据

## 决策

最终主排名只是阅读入口，不能把已经完成的多算法长期证据压缩成一个名字和两个小数。
frontier终交付必须在发布前fail closed地验证并输出：

- selected receipt与derivation card、implementation、executor contract的精确hash/身份绑定；
- 固定e200而非最佳checkpoint，batch1、30000 updates、seed2026和延期seed的事实；
- 主候选每个里程碑、每个域的candidate/plain绝对PSNR/SSIM/LPIPS及delta；
- selected算法proposal-only、observable-only、projected/full三项e200的同样逐域分解；
- 4090和5090各自完整的宿主内排名，禁止跨宿主delta合并；
- 公式、可执行配置、源码hash、复杂度、风险和可直接追溯的复现合同；
- 在报告中分别列出科学结论、工程失败、代理失真和未测试假设。

这些字段没有真实终点文件时不得用模板或`null`冒充。终交付等待器只在所有source-bound
e200证据存在后运行；任何card、implementation、receipt、trajectory、metric或executor
contract不一致都会拒绝生成`CANDIDATE.json`。

## 执行边界

实现最终冻结于`970784c`并已作为新的4090终交付等待器部署。替换过程中只停止旧的等待进程，
PC-RSMG消融、5090两条前沿训练、跨机路由和winner-specific消融等待器均保持原身份运行。
该改动不读取中间paired指标，不改变算法、训练轨迹、排序规则或算力分配。
