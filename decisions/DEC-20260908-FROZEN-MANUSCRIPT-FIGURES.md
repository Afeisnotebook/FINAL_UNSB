# DEC-20260908：冻结后的论文轨迹图和六域图成为终局交付项

## 裁决

论文终局不能只留下机器可读CSV。新增标准库实现的确定性SVG renderer，并把
`MANUSCRIPT_FIGURES_RECEIPT.json`加入预结果写作合同的必要产物。它只在已提交的算法、
基线和claim freeze生成表格回执后执行，因此不会提前看到结果或影响训练决策。

## 固定图形

1. `ALGORITHM_SUSTAINED_PSNR.svg`：完整显示Proposal、ST-CGR、AM-TNC等所有终局方法在
   e150/e175/e200的matched宏PSNR delta；包含零基线，不挑最佳checkpoint。
2. `ALGORITHM_E200_DOMAIN_PSNR.svg`：显示每个终局方法在固定e200的六域PSNR delta；正负域
   均保留，不因算法未通过科学门而隐藏。

renderer会重新验签`MANUSCRIPT_TABLES_RECEIPT.json`绑定的两张CSV，拒绝缺失晚三点、缺失
域、非CRN、非合法runtime relation、best-checkpoint或任何修改后的来源。输出SVG及回执均
不可变并带SHA256，方便直接进入论文或后续转换为PDF。

## 边界

本次没有读取任何在途性能值，没有启动、停止或调整实验，没有打开confirmation20，也没有
改变训练fingerprint。实际图形仍等待完整e200统一评估、人工/Codex claim review和冻结表格；
等待状态不算Goal完成。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_FROZEN_MANUSCRIPT_FIGURE_INTERFACE_20260908T042500.json`。
