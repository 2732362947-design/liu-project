# 提交说明

## 1. 项目入口

本项目的官方评测入口位于仓库根目录：

```python
from user_agent import ReasoningAgent

agent = ReasoningAgent(client=official_client)
result = agent.solve(problem, metadata)
```

`solve(problem, metadata)` 返回一个 `dict`，基本结构为：

```json
{
  "final_response": "最终答案或完整证明过程",
  "trace": []
}
```

- `final_response` 是非空字符串。计算、选择、填空和解方程题返回短答案；明确要求证明、推导、解释或论证的题目保留可独立判分的完整过程。
- `trace` 用于记录分类、规划、求解、提取答案、验证、最终化等步骤。
- `trace` 不包含 API key、token、Authorization header 或个人隐私。

## 2. 与官方评测环境的适配

- 正式评测时由平台提供 `official_client`。
- `user_agent.py` 不读取 `.env`。
- `user_agent.py` 不硬编码 API key。
- `user_agent.py` 不调用本地 `intern_s1_client.call_intern_s1`。
- `user_agent.py` 只使用传入的 `client.chat(...)`。
- 本地 `.env` 仅用于开发调试，不属于官方入口依赖。

## 3. 智能体流程

```text
Local Exact Tools
-> Classifier
-> Planner
-> Solver Router
-> Official Client LLM Call
-> Answer Extractor
-> Local Verifier
-> Retry Correction when needed
-> Final Response
```

- Local Exact Tools：对可形式化的小规模组合图问题，优先使用本地精确工具求解。
- Classifier：识别题目领域和 `solver_key`。
- Planner：生成解题计划。
- Solver Router：加载对应数学方向 prompt 模板。
- Official Client LLM Call：通过平台传入 client 调用 Intern-S 系列模型。
- Answer Extractor：提取最终答案。
- Local Verifier：检查答案非空、格式、基本数学类型约束等。
- Retry Correction when needed：当本地验证器发现答案为空、格式异常或验证不通过时，系统最多进行一次修正调用。
- Final Response：按 `short_answer` 或 `worked_solution` 模式返回官方要求的 `final_response`。

retry 使用原题、第一次解答和本地验证反馈构造 correction prompt，不使用标准答案或隐藏评测信息。

本地精确工具不访问网络，不使用标准答案，不读取隐藏数据；对不匹配的问题，系统仍走 LLM 推理流程。
工具输出会记录可审计的图规模、边数、孤立点数量、最大独立集大小和一个独立集 witness；若外部样本参考答案与形式化计算冲突，系统以可复现工具计算为准。

## 4. 本地提交前自检

```bash
pytest -q
python3 dev_tools/check_submission_ready.py
```

- `pytest -q` 用于运行单元测试。
- `check_submission_ready.py` 用于检查官方入口、返回格式、JSON 序列化、敏感信息风险、绝对路径风险等。
- 该自检工具默认不调用真实 API。

## 5. official-style 本地模拟 runner

```bash
python3 dev_tools/run_user_agent_local.py \
  --input data/user_agent_smoke.jsonl \
  --output-dir outputs_user_agent \
  --overwrite
```

也可以运行：

```bash
bash scripts/run_user_agent_local.sh
```

- 该工具模拟官方平台调用 `ReasoningAgent(client=...)` 和 `solve(problem, metadata)` 的流程。
- 默认使用 FakeClient。
- 不访问网络。
- 不读取 `.env`。
- 输出结果位于 `outputs_user_agent/`。
- `outputs_user_agent/` 是本地生成文件，不应提交到仓库。

## 6. 依赖说明

正式运行安装：

```bash
pip install -r requirements.txt
```

- `requirements.txt` 应只包含正式入口需要的最小依赖。
- 当前正式入口及传递导入链只依赖 Python 标准库。
- 不要求 GPU。
- 不依赖本地绝对路径。
- 不依赖隐藏测试集或标准答案。

本地开发安装：

```bash
pip install -r requirements-dev.txt
```

开发依赖只用于 pytest、本地 Intern-S client 和调试脚本，不会被正式入口导入。

## 7. 本地开发工具说明

```text
dev_tools/check_submission_ready.py：提交前自检
dev_tools/run_user_agent_local.py：模拟官方平台调用 user_agent.py
dev_tools/run_user_agent_real_smoke.py：用于本地手动验证 user_agent.py 在真实 Intern-S API client 下能跑通；不是官方入口，不用于批量评测
dev_tools/run_advanced_real_sanity.py：按单并发、每领域一题手动检查高级数学领域路由；不属于自动测试
dev_tools/convert_omni_math.py：Omni-MATH 本地样本转换
dev_tools/check_domain_coverage.py：领域覆盖检查
```

这些工具仅用于本地开发和验证，不是官方入口。

## 8. 安全与合规说明

- 不提交 `.env`。
- 不在代码中硬编码 API key。
- 不在 trace 中记录 token、Authorization header 或个人隐私。
- `solve(problem, metadata)` 不使用 `metadata["answer"]` 作为最终答案。
- metadata key 会先忽略大小写，并将空格、连字符规范化为下划线，再按精确 key 过滤。当前 denylist 为：`answer`、`expected_answer`、`gold_answer`、`reference_answer`、`solution`、`gold`、`reference`、`ground_truth`、`expected`、`expected_solution`、`official_answer`、`label`、`target`。
- denylist 不做模糊子串匹配；`idx`、`domain`、`subject`、`source` 等正常字段仍可保留。
- 正式评测不会依赖本地输出文件、dev review 文件或外部数据集。
- 本地 Omni-MATH 样本仅用于压力测试和领域覆盖分析。

## 9. 推荐提交前检查清单

- [ ] `user_agent.py` 位于仓库根目录
- [ ] `ReasoningAgent` 可以被 import
- [ ] `ReasoningAgent(client=official_client)` 可以初始化
- [ ] `solve(problem, metadata)` 返回 dict
- [ ] 返回包含非空字符串 `final_response`
- [ ] `trace` 可以 JSON 序列化
- [ ] `pytest -q` 通过
- [ ] `python3 dev_tools/check_submission_ready.py` 通过
- [ ] `bash dev_tools/check_clean_environment.sh` 通过
- [ ] 没有提交 `.env`
- [ ] 没有提交 `outputs_user_agent/`
- [ ] 没有硬编码 API key
- [ ] 没有使用本地绝对路径
