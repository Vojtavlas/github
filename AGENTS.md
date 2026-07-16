# ReasonFlow project notes

## Environment gotchas

- The default `python`/`pytest` in this shell points to a Hermes venv without `torch`.
- Use the system Python 3.11 installation for this repo:
  ```powershell
  py -3.11 -m pip install -e .
  py -3.11 -m pytest -q
  py -3.11 examples/simple_demo.py --max-new-tokens 10
  ```

## Verification commands

```powershell
ruff check src tests examples
py -3.11 -m pytest -q
```

- `ruff check src tests examples` lints the codebase.
- `py -3.11 -m pytest -q` runs the test suite. CI sets `SKIP_ENGINE_TESTS=1` to skip GPT-2 integration tests.

## Demo expectations

- `examples/simple_demo.py` defaults to `Qwen/Qwen3.5-0.8B`. Use it for both demos and tests; set `HF_TOKEN` for higher Hugging Face Hub rate limits.
- Set `HF_TOKEN` for higher Hugging Face Hub rate limits.
