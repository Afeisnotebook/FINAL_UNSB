# DEC-20260830：分离算法定义、证据谱系与执行身份

状态：`FROZEN_BEFORE_FIRST_CANDIDATE`

## 决定

路线一候选使用两层指纹：

- `algorithm_fingerprint` 绑定公式、数学条件、算法超参/状态、实现配置、注册源码和共同
  训练协议；它不包含 host-specific atlas、matrix 或 e0，因此相同公式与源码在1660、
  4090和新seed上仍被识别为同一算法。
- `candidate_fingerprint` 额外绑定完整derivation card、implementation manifest、atlas、
  causal matrix、完整训练核心、共同e0、local view和runner源码；它表示一次可审计的
  “证据—实现—执行”注册，跨host/seed必然不同。

算法定义仍必须通过candidate registration追溯到各自主机的证据hash。分离不是丢弃
谱系，而是防止“同一算法因证据文件不同被误判为两个算法”，同时防止把不同e0或不同
运行环境伪装成同一次matched实验。

该身份规则在第一个Generation-1候选生成前冻结；seed2027/2028只允许改变执行身份，
不得改变`algorithm_fingerprint`。
