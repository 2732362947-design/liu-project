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
  --model intern-s2-preview \
  --concurrency 1 \
  --limit-per-domain 1
```

runner 会打印 `requested_model`，只向正式 agent 传入题目和 `idx`，并在每题结束后原子更新结果文件。JSON 保存完整 `final_response`、路由、响应模式、API 调用次数、retry、验证、fallback、本地工具、耗时及精简 trace；终端只显示答案前 300 字。高级 sanity 不支持 resume，重新执行会从第一题开始，并在首题完成后覆盖旧结果。输出位于已忽略的 `outputs/`，不得提交真实 API 结果。

查看高级 sanity 结果：

```bash
python3 -m json.tool outputs/real_api_sanity_advanced_results.json
```

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

## Omni-MATH 真实 Intern-S API 评估

评估数据由本地完整 Omni-MATH `test.jsonl` 分层、定种子生成，不会自动下载数据。`--sample-size 250` 表示 main 与 holdout 合计 250 题，另生成 30 题 smoke；默认拆分为 smoke 30、main 200、holdout 50：

```bash
python3 dev_tools/prepare_omni_evaluation.py \
  --source ~/.cache/modelscope/hub/datasets/AI-ModelScope/Omni-MATH/test.jsonl \
  --output evaluation/datasets/omni_math_eval_250.jsonl \
  --sample-size 250 \
  --smoke-size 30 \
  --holdout-size 50 \
  --seed 20260720
```

真实评估只向 `ReasoningAgent.solve()` 传入 `{"idx": ...}`；仅显式增加 `--use-subject-hint` 时才附加 `subject`。标准答案、solution、标签状态和复核备注不会进入 agent metadata。默认并发为 1，逐题追加结果并支持断点续跑。

先运行 smoke：

```bash
python3 dev_tools/run_omni_real_api_eval.py \
  --input evaluation/datasets/omni_math_smoke_30.jsonl \
  --output evaluation/results/omni_math_smoke_results.jsonl \
  --model intern-s2-preview \
  --concurrency 1 \
  --resume
```

确认 smoke 后运行 main：

```bash
python3 dev_tools/run_omni_real_api_eval.py \
  --input evaluation/datasets/omni_math_main_200.jsonl \
  --output evaluation/results/omni_math_main_results.jsonl \
  --model intern-s2-preview \
  --concurrency 1 \
  --resume
```

holdout 仅在主评估结束后运行：

```bash
python3 dev_tools/run_omni_real_api_eval.py \
  --input evaluation/datasets/omni_math_holdout_50.jsonl \
  --output evaluation/results/omni_math_holdout_results.jsonl \
  --model intern-s2-preview \
  --concurrency 1 \
  --resume
```

离线评分不使用 SymPy；只自动判断整数、小数、科学计数法、简单分数/百分数、单变量赋值、多根、简单有序对/同余类、选择题和无解。证明/推导/解释题、复杂表达式及疑似错误标签进入人工复核：

```bash
python3 dev_tools/score_omni_results.py \
  --dataset evaluation/datasets/omni_math_smoke_30.jsonl \
  --results evaluation/results/omni_math_smoke_results.jsonl \
  --output-json evaluation/reports/omni_math_smoke_report.json \
  --output-md evaluation/reports/omni_math_smoke_report.md
```

`evaluation/results/`、`evaluation/reports/` 与其他真实 API 输出默认被 Git 忽略，只保留目录占位文件。
