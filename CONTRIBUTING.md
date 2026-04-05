# Contributing

## Branching

- `main`: release-ready
- feature branches: `feat/...`
- fixes: `fix/...`
- experiments: `exp/...`

## Commit style

Use readable, specific commit messages:

- `data: add balanced category sampler`
- `model: train skip-gram 300d run`
- `eval: add retrieval recall@10`
- `docs: update master plan after scope freeze`

## Pull request rule

A PR is not complete unless it updates:

- relevant code
- relevant notebook
- relevant README
- experiment note or decision log when the change affects results

## Non-negotiables

- No large raw data in Git
- No notebook with massive output cells committed
- No undocumented experiment results
- No metric claims without saved config + split details
