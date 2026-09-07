# DEC-20260908：预注册论文结果分支进入确定性导出器

## 裁决

此前的八分支写作合同只是文档约束，最终表格导出器并不会强制使用它。现在将该合同作为
`paper_aio_manuscript_tables.py` 的唯一合法分支来源：导出前必须确认合同已提交且工作区无
修改、三条方法均有终局 PASS/FAIL 裁决、八种布尔组合恰好各出现一次，并重新验证合同绑定的
理论、基线和 novelty 文件哈希。

导出器新增 `MANUSCRIPT_RESULT_BRANCH.json`，把 Proposal、ST-CGR、AM-TNC 的终局 disposition
确定性映射到预注册 route，并将合同、portfolio、freeze receipt 和 claims 哈希写入最终 receipt。
任一方法尚未裁决、合同不完整、使用非 canonical 合同或绑定文件变化都会 fail closed。

这项修改只作用于结果冻结后的写作导出链，不读取当前训练的性能值、不改变训练 fingerprint，
也不会预定唯一赢家。全套测试为 734 passed。实现 commit 为
`95ae918839dee86e56dcd3e5e00cbe415cbc77ae`，机器可读证据见
`evidence/paper_aio/PAPER_AIO_MANUSCRIPT_BRANCH_ENFORCEMENT_20260908T020000.json`。
