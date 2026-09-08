# DEC-20260908：六域数据来源、定制复用与发布边界

## 裁决

本项目的六个目录名与 MPMF-Net（AAAI 2025）官方仓库提供的六个测试数据集完全一致，
MPMF-Net论文也逐域给出了对应的退化类型和原始引用。因此当前数据具有高置信度的命名与协议
谱系。但是本地数据目录没有保留README、LICENSE、原始下载回执或上游archive checksum，不能
声称本地9,153对图像已经与某个上游下载包逐字节证明同一。

更重要的是，MPMF-Net把这六个集合称为测试集；FINAL_UNSB把它们重新划为8,553个训练identity、
每域80个discovery和20个封存confirmation，并用全局B边际做无配对训练。因此论文必须把它
表述为“基于既有评测集合构造的受控自定义无配对split”，不能写成任何上游数据集的官方训练
协议，也不能把我们的数字与上游paired测试数字直接横比。当前所有方法在同一哈希锁定split上
重训，所以这项披露不否定内部matched比较。

## 发布边界

MPMF-Net在审核commit `13feb0d`下没有顶层LICENSE，GitHub license endpoint返回404；Foggy
Cityscapes官方页面还明确指出主图像因Cityscapes许可只能经Cityscapes网站获取。因此在取得作者
确认或逐域许可之前：

- 不公开图像、重打包archive或含样本像素的非必要产物；
- 公开代码、split生成逻辑、聚合计数和整体hash；
- stem和逐文件hash是否公开须先完成条款复核；
- 默认发布让用户自行取得上游数据后运行的manifest builder与hash verifier；
- 所有未确认域保持`license_not_verified_do_not_redistribute`，不得把“无LICENSE”解释成可自由使用。

该审计不读取性能结果、不改训练队列、不打开confirmation20。机器可读合同和证据分别为
`configs/PAPER_DATASET_PROVENANCE_CONTRACT.json`与
`evidence/paper_aio/PAPER_AIO_DATASET_PROVENANCE_AND_REUSE_AUDIT_20260908T090300.json`。
