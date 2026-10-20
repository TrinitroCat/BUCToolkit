## Summary

<!-- Describe the change and why it is needed. -->

## Local validation

Remote CI only runs lightweight CPU regression and smoke checks. Please run the
relevant suites locally before opening the pull request.

- [ ] `python -m pytest` passes (GPU-only tests may skip on a CPU machine).
- [ ] With a GPU available: `buctoolkit --test fast` runs to completion.
- [ ] Reviewed the printed numerical diagnostics from the suites I ran; a green
      `fast`/`main` run does not mean every printed deviation passed.
- [ ] For strict scientific changes or releases: `buctoolkit --test main` runs
      to completion and its printed diagnostics were reviewed.
- [ ] New or changed behavior has a focused regression test.
- [ ] No unrelated worktree files or generated artifacts are included.

If no GPU was available, list which GPU/Triton tests could not be run.

See `DEVELOPMENT.md` section 13 for the exact commands and optional extras.
