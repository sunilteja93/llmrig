## What changed?

Describe the change and why it is useful.

## How was it tested?

- [ ] `python -m py_compile llmrig.py`
- [ ] `python -m unittest discover -s tests -v`
- [ ] `python llmrig.py check`
- [ ] I tested platform/model-specific behavior on real hardware when the change requires it

## Model or hardware metadata changes

If this PR changes model sizes, tags, context limits, platform support, or hardware recommendations, link the primary/upstream sources used to verify the change.

## Checklist

- [ ] I added or updated tests for behavior changes
- [ ] I updated documentation for user-visible changes
- [ ] I did not add secrets, private paths, tokens, or unreviewed benchmark output
- [ ] Newly discovered third-party models are not made auto-installable without verification
