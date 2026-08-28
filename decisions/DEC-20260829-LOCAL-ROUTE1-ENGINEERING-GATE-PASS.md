# DEC-20260829：本地路线一工程门通过

状态：`ACCEPTED`  
上游决策：`DEC-20260829-LOCAL-ROUTE1-REOPEN.md`

## 决定

接受commit `a4883eb`及协议指纹`b0786b...9b2`作为本轮长期锚点的唯一执行
基座。允许按`plain -> HJ -> HNEK`顺序运行small25 e200；当前不授权DT、新算法
长训、路线二或4090。

## 原因

CPU与GTX 1660门禁证明共同e0、inactive方法路径、full-state resume、两套独立
sampler、全部RNG以及CRN评估均满足精确一致性。DT核心保留历史语义，HJ的e5物理
起点与HNEK disable identity也已审计。

## 非结论

工程门通过不代表HJ/HNEK/DT有效，不代表small25 proxy有效，也不代表已有候选。
proxy是否可用必须由HJ/HNEK真实e200轨迹决定；若二者均无法校准，暂停DT和新算法
长训并诊断proxy/谱系，不能写成机制死亡。

## 证据

- `evidence/local_route1_gate/CPU_GATE.json`
- `evidence/local_route1_gate/GPU_GATE.json`
- `evidence/local_route1_gate/LINEAGE.json`
- `evidence/local_route1_gate/ENGINEERING_GATE_REPORT_CN.md`
