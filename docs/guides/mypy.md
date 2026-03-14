# Running mypy in this repository

This project uses [mypy](https://mypy.readthedocs.io/) for static type checking. Configuration lives in `pyproject.toml` under `[tool.mypy]`.

## Quick start

Run mypy against the entire package:

```bash
uv run --active mypy src/statista
```

Check a single module:

```bash
uv run --active mypy src/statista/distributions.py
```

## Configuration

Key settings in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
check_untyped_defs = true
strict_optional = true
ignore_missing_imports = true
warn_return_any = false
warn_redundant_casts = true
warn_unreachable = true
```

Notable choices:

- `ignore_missing_imports = true` — third-party libraries without type stubs (matplotlib, scipy, etc.) do not cause errors.
- `strict_optional = true` — variables typed as `Optional[X]` must be narrowed before use. Always add a `None` guard before performing operations on optional values.
- `check_untyped_defs = true` — functions without annotations are still checked internally.
- Test files (`tests.*`) and `setup` are excluded from checking via `[[tool.mypy.overrides]]`.

## Common error patterns and fixes

### `[assignment]` — incompatible types in assignment

A variable's inferred type conflicts with a later assignment:

```python
# Bad: mypy infers merged as list[int] from the empty literal
merged = []
merged = np.array(merged)  # error: incompatible types

# Fix: annotate with the broader type, or use a separate variable
merged_list: list[float] = []
merged_arr = np.array(merged_list)
```

### `[arg-type]` — wrong argument type

Often caused by passing `**kwargs` (typed as `dict[str, Any]`) where specific parameter types are expected. Fix by passing arguments explicitly or adding a targeted `# type: ignore[arg-type]`.

### `[operator]` — unsupported operand types

Usually from operating on `Optional` values without a `None` check:

```python
# Bad
def f(x: float | None) -> float:
    return x * 2  # error: unsupported operand types

# Fix: narrow first
def f(x: float | None) -> float:
    if x is None:
        x = 0.0
    return x * 2
```

### `[override]` — incompatible method signature in subclass

Subclass method signatures must match the parent. If the difference is intentional, suppress with `# type: ignore[override]`.

### `[valid-type]` — using `callable` instead of `Callable`

```python
# Bad
def run(func: callable) -> None: ...

# Fix
from typing import Callable
def run(func: Callable[..., Any]) -> None: ...
```

### `[var-annotated]` — missing type annotation

Mypy needs a type hint when it cannot infer the type of an empty container:

```python
# Bad
items = []  # error: need type annotation

# Fix
items: list[float] = []
```

## Integrating with CI

Add mypy to your CI pipeline alongside tests:

```yaml
- name: Type check
  run: uv run --active mypy src/statista
```

## References

- mypy documentation: https://mypy.readthedocs.io/
- Type hints cheat sheet: https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
