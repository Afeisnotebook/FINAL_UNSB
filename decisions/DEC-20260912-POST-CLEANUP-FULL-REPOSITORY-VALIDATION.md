# DEC-20260912：误清理恢复后的全仓回归裁决

## 裁决

在 commit `3662eaa` 上对 `tests/` 与 `src/tests/` 的779个测试做CPU-only回归。Windows下
`CUDA_VISIBLE_DEVICES=""` 会形成 `torch.cuda.is_available()==True` 但device count为0的
矛盾状态，因此第一次长套件中的两个Invalid-device失败不是代码回归。另一个跨进程文件锁
测试在整套负载下超时，但三项单独复跑全部通过。

改用可验证的 `CUDA_VISIBLE_DEVICES=-1` 后，单个长寿命pytest进程仍出现一次Windows原生
`access violation`；对应入口测试单独通过。为了不把运行时偶发崩溃误写成测试成功或科学
代码失败，最终按文件边界拆分验证：前30个文件逐个运行，其余120个文件分四批运行，另行
运行 `src/tests/`。总计 `777 passed / 2 CPU-only expected skipped / 0 failed`，所有779个用例
均有终态；compileall、665个JSON解析及重复键检查、git diff检查全部通过。

本次不为了获得单进程形式上的全绿而修改科学代码。验证期间显式禁用测试CUDA，本地DCLGAN
及全部控制/relay进程保持原PID，训练与协议未改变。

证据：`evidence/paper_aio/PAPER_AIO_POST_CLEANUP_FULL_REPOSITORY_VALIDATION_20260912T042000.json`。
