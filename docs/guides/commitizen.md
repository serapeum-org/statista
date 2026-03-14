# Using Commitizen in this repository

This project is configured to use Commitizen with the "Conventional Commits" standard. Commitizen helps you:
- write consistent commit messages interactively,
- bump versions based on commit history,
- and update the changelog automatically.

Configuration lives in `pyproject.toml` under `[tool.commitizen]` and uses:
- version source: PEP 621 `[project].version`
- tag format: `$version` (e.g., `0.6.3`)
- changelog path: `docs/change-log.md`

Note: In this repo, Commitizen may already be available if you install the "docs" optional dependency group. However, installing it as a development tool (pipx/Poetry dev) is typically preferred.

## Making commits (interactive)
Use Commitizen's interactive prompt to standardize commit messages:

- `cz commit`
  - or shorthand: `cz c`

You will be asked for:
- type (feat, fix, docs, refactor, test, chore, build, ci, perf, style, etc.),
- optional scope (e.g., `distributions`),
- short subject line,
- longer body (optional),
- breaking change note (if any),
- issue references (optional).

Examples of resulting commit messages:
- `feat(distributions): add Weibull MLE fit`
- `fix(eva): handle empty series edge case`
- `docs: expand installation instructions`

Tip: You can still use `git commit` directly, but `cz commit` helps you stay within the Conventional Commits spec.

## Bumping the version and updating the changelog
Commitizen can analyze commit history and pick the next version automatically (major/minor/patch) following semantic versioning.

- Dry run to see what would happen:
  - `cz bump --dry-run`

- Perform the bump (updates `[project].version` in `pyproject.toml`, creates a VCS tag, and updates the changelog):
  - `cz bump`

- Non-interactive (skip confirmations):
  - `cz bump --yes`

- Pre-releases:
  - `cz bump --pre alpha`  (also supports `beta`, `rc`, etc.)

Notes:
- Tags use the configured format `$version` (no `v` prefix).
- The changelog is written to `docs/change-log.md`, which is included in the documentation site under About → Change-log.

## Typical release flow
1. Ensure your main branch is clean and all tests pass.
2. Use `cz commit` for all changes merged into main.
3. Run `cz bump` to update the version, changelog, and create a tag.
4. Push commits and tags:
   - `git push && git push --tags`
5. Cut a GitHub release from the created tag if desired, or let your CI pick it up.


## Troubleshooting
- Commitizen not found: ensure it is installed in your current environment (`pipx list` or `pip show commitizen`).
- Bump fails to determine type: verify that recent commit messages follow Conventional Commits.
- No tag created: confirm you have a clean Git status and that your repo has Git initialized.
- Changelog not updating: check `[tool.commitizen]` in `pyproject.toml` and that `docs/change-log.md` is writable.

## References
- Conventional Commits: https://www.conventionalcommits.org/
- Commitizen docs: https://commitizen-tools.github.io/commitizen/
