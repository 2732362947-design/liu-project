# Agent System

正式评测入口是仓库根目录的 `user_agent.py`。接口、响应模式、安全边界和提交前检查详见 [SUBMISSION.md](SUBMISSION.md)。

## 安装

正式运行只需 Python 标准库：

```bash
pip install -r requirements.txt
```

本地测试和 Intern-S 调试工具使用开发依赖：

```bash
pip install -r requirements-dev.txt
```

## 本地检查

以下检查均使用 fake client，不访问真实 API：

```bash
pytest -q
python3 dev_tools/check_submission_ready.py
bash dev_tools/check_clean_environment.sh
```

## 高级领域真实 API 手动 sanity

以下命令只供人工执行，不属于自动测试。runner 强制并发为 1，并对 `data/real_api_sanity_advanced.jsonl` 中每个领域最多运行 1 题：

```bash
python3 dev_tools/run_advanced_real_sanity.py \
  --input data/real_api_sanity_advanced.jsonl \
  --output outputs/real_api_sanity_advanced_results.json \
  --concurrency 1 \
  --limit-per-domain 1
```

输出只汇总 `domain`、`solver_key` 和 `final_response_nonempty`，写入已忽略的 `outputs/`，不得提交真实 API 输出。

当前高级领域均使用独立轻量 prompt 模板：

| domain | solver_key |
| --- | --- |
| `numerical_analysis` | `numerical_analysis` |
| `measure_theory` | `measure_theory` |
| `differential_geometry` | `differential_geometry` |
| `abstract_algebra` | `abstract_algebra` |
| `stochastic_processes` | `stochastic_processes` |
| `statistics` | `statistics` |
| `functional_analysis` | `functional_analysis` |
| `linear_regression` | `linear_regression` |
| `mathematical_analysis` | `mathematical_analysis` |
