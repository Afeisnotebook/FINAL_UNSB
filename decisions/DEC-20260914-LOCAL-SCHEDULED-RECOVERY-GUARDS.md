# DEC-20260914：将本地端点与交付恢复层提升为Windows计划任务守护

## 问题

44804端点watcher以及43172的两条本地relay恢复链在约09:47同时消失，而本地DCLGAN科学训练继续存活。没有证据把该事件归因于算法或checkpoint；但它证明普通`Start-Process`不足以作为最终持久层。

## 决定

注册三个每五分钟触发的Windows计划任务。任务先按模块和唯一output片段检查目标supervisor：零个时执行冻结命令；一个时无操作成功退出；多于一个时fail closed。任务不含密码，relay继续只引用冻结private-key路径和source-bound contract。

三项已有进程no-op测试均返回0且保持单supervisor。随后只对无状态44804 supervisor做故障注入：PID `6916`被验证后终止，计划任务恢复PID `14068`并收养原child `23708`；新health PID为`27988`，状态`HEALTHY`。任何科学训练未被触碰。

该层只解决本机控制进程意外消失，不能启动provider实例、不能猜测43172远端训练状态，也不能绕过44804的GPU UUID identity gate。
