# MhcSinkhorn 系统测试

案例定义位于 `cases/coverage.json`、`cases/correctness.json`、`cases/mock.json`。
分别包含 30 个输入覆盖、105 个正确性（含 3 个 sensitivity 和 6 个 convergence_order）、5 个固定 Mock 案例。

运行入口为 `scripts/run_tests.py`；Mock 独立入口为 `scripts/profile_mock_workloads.py`。
目录职责、案例选择、计时协议和命令见 [测试说明](../README.md)。
