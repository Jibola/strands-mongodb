# Releasing

A release is one push of one tag. Everything after that is automatic except the single approval that
stops it, and everything before it is undoable. This document is the order of those steps, because
the cost of not knowing it is paid exactly once, at the only moment that cannot be reversed: an
upload to PyPI can be yanked, never replaced.

The procedure is `.github/workflows/release-python.yml`. That file is the authority; this one is the
part a person needs and cannot get by reading YAML at speed. `AT-163` asserts that the two agree
about the jobs, the tag and the environment, so a fifth job added there fails the suite here rather
than going unmentioned.

## What a release is made of

The version is not written down anywhere. `hatch-vcs` derives it from the git tag, matching
`^python-v(?P<version>.+)$`, so `python-v0.1.0` builds `0.1.0`. A build from a tree with no such tag
gets a development version counted from the first commit, such as `0.0.1.dev69+g20b0ea579`, and a build
with no git metadata at all falls back to `0.0.0`. Neither can be mistaken for a release.

This means there is no version bump commit and nothing to keep in sync. It also means the tag is the
whole decision: tag the commit you mean, because the artifact is built from it and from nothing else.

## Before you tag

1. **Be on `main`, at the commit you intend to release.** `git log -1` and read it.
2. **Update [CHANGELOG.md](CHANGELOG.md).** Move everything under `## [Unreleased]` into a heading for
   the version you are about to tag, and give that heading the release date. Nothing else in the
   repository records what a version contains, and after the upload nobody can add it retroactively to
   the artifact.
3. **If this is the first release ever**, remove the
   `<!-- skip: these reach PyPI for a package that is not published yet -->` marker from both
   `README.md` files in the same commit. `AT-164` couples the two: while the marker is present the
   changelog may date nothing, and once it is gone the changelog must date something.
4. **Run the gate locally.** `hatch run format-check`, `hatch run lint`, `hatch run typecheck`,
   `hatch run cov`. The release workflow runs all four again and a failure there wastes only time, but
   it wastes it after you have already pushed a tag, and a tag pointing at a commit that cannot be
   released has to be deleted from the remote by hand.

## Tagging

```bash
git tag python-v0.1.0
git push origin python-v0.1.0
```

The `python-v` prefix is what starts a release — a plain `v0.1.0` tag does nothing at all. The prefix
exists because this repository is a monorepo: a future `typescript/` versions independently off the
same history, and its tags must not reach this workflow.

## What runs, in order

| Job | Name in the Actions UI | What it does |
|---|---|---|
| `gate` | Release gate | `validate-pyproject` on the manifest, then formatting, lint, types and both coverage floors, with `STRANDS_MONGODB_BUILD_TIER` set |
| `contract` | Contract tier | The contract tests against a real `mongo:8` service container |
| `build` | Build and verify the artifact | `hatch build`, then the check below, then `twine check --strict` on what PyPI will render, then uploads `python-dist` |
| `publish` | Publish to PyPI | Downloads that artifact and uploads it |

`gate` and `contract` run in parallel; `build` needs both; `publish` needs `build`.

The check inside `build` is the one worth knowing about: the built wheel's version must equal the
tag, compared as PEP 440 objects rather than as strings, so `python-v0.1.0-rc1` and `python-v0.1.0rc1`
are the same answer while a genuinely different version still fails. A mismatch means `hatch-vcs` saw
a dirty tree or no tag, and publishing that would put a `.dev` or `0.0.0` artifact on PyPI forever.

## The approval

`publish` is bound to the `pypi` environment, and that environment has required reviewers. A tag does
not publish: the run stops at `publish` and waits for a human to approve the deployment in the
Actions UI. That pause is the last stop before the irreversible step, and it is the one part of this
process you cannot discover by reading the repository — the environment and its reviewer list live in
repository settings, not in a file.

**Neither exists yet.** There is no `pypi` environment in the repository settings and no trusted
publisher on PyPI. GitHub creates a missing environment on first reference, with no reviewers, so a tag
pushed today would reach the upload unapproved and fail there only because PyPI has nothing to
exchange the token with. Do not push a `python-v` tag until both are configured.

If the run appears to hang at `publish`, it is not hung. It is waiting for somebody, and if nobody
who can approve is watching, it waits until the run expires.

## Credentials

There are none. The upload uses PyPI's trusted publishing over OIDC: no API token exists in this
repository, so there is nothing to leak or rotate. PyPI verifies the identity of that workflow file,
in that repository, running in the `pypi` environment, and mints a token good for one upload.

Rename the workflow file or the environment and the upload stops working until the publisher on PyPI
is updated to match. That coupling is the security property, not an inconvenience.

## When it goes wrong

| Symptom | What happened | What to do |
|---|---|---|
| No run started | The tag did not match `python-v*` | Delete the tag locally and on the remote, tag again with the prefix |
| `gate` or `contract` red | The commit was not releasable | Fix on `main`, delete the tag from the remote, tag the new commit |
| `wheel is X, tag says Y` | `hatch-vcs` saw a dirty tree or no tag | Tag an exact commit on a clean checkout |
| Stuck at `publish` | Waiting for a required reviewer | Approve the deployment, or find somebody who can |
| Wrong artifact uploaded | It is on PyPI permanently | Yank that version on PyPI and release a new patch. There is no replace |

The last row is the reason for every row above it.
