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
