# Releasing

A release is one push of one tag. Everything after that is automatic except a single approval, and
everything before it can be undone. This document gives the order of those steps, because the upload
at the end cannot be reversed: an upload to PyPI can be yanked, never replaced.

The procedure is `.github/workflows/release-python.yml`. That file is the authority; this one is the
part a person needs to follow it. A test checks that the two agree about the jobs, the tag and the
environment, so a job added to the workflow and not described here fails the suite.

## What a release is made of

The version is not written down anywhere. `hatch-vcs` derives it from the git tag, matching
`^python-v(?P<version>.+)$`, so `python-v0.1.0` builds `0.1.0`. A build from a tree with no such tag
gets a development version counted from the first commit, of the form `0.0.1.devN+g<sha>`, and a build
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
   `README.md` files in the same commit. A test couples the two: while the marker is present the
   changelog may date nothing, and once it is gone the changelog must date something.
4. **Run the gate locally.** `hatch run format-check`, `hatch run lint`, `hatch run typecheck`,
   `hatch run cov`. The release workflow runs all four again and a failure there wastes only time, but
   it wastes it after you have already pushed a tag, and a tag pointing at a commit that cannot be
   released has to be deleted from the remote by hand.
5. **Run the Atlas tiers on that commit.** Start `CI - Python` by hand from the Actions tab on `main` and
   wait for `Atlas tiers`. It is the only job that runs `autoembed`, the package's default search mode,
   against Atlas, and it does not run on a merge.

## Tagging

```bash
git tag python-v0.1.0
git push origin python-v0.1.0
```

The `python-v` prefix is what starts a release. A plain `v0.1.0` tag does nothing. The prefix
exists because this repository is a monorepo: a future `typescript/` versions independently off the
same history, and its tags must not reach this workflow.

## What runs, in order

| Job | Name in the Actions UI | What it does |
|---|---|---|
| `gate` | Release gate | `validate-pyproject` on the manifest, then formatting, lint, types and both coverage floors, with `STRANDS_MONGODB_BUILD_TIER` set |
| `contract` | Contract tier | The contract tests against a real `mongo:8` service container |
| `build` | Build and verify the artifact | `hatch build`, then the check below, then `twine check --strict` on what PyPI will render, then uploads `python-dist` |
| `sign` | Sign the artifact | Checksums the wheel and the sdist, signs all three with Garasign, uploads `python-signatures` |
| `publish` | Publish to PyPI | Downloads `python-dist` and uploads it |
| `github-release` | Attach the signatures to the GitHub Release | Creates the Release for the tag if it does not exist, attaches the artifacts, the `.sig` files and `SHA256SUMS` |
| `sbom` | Augment and attach the SBOM | Generates a CycloneDX SBOM, augments it through SilkBomb, attaches it to the Release. **Skipped** unless `KONDUKTO_SECRET_ID` is set |
| `trace` | Record the authorized publication | Writes the authorized-publication report and uploads it to S3. **Skipped** unless `AWS_RELEASE_BUCKET` is set |

`gate` and `contract` run in parallel; `build` needs both; `sign` needs `build`; `publish` needs
`sign`; `github-release` needs `publish`; `sbom` needs `github-release`; `trace` needs `publish`.

`publish` needs `sign` rather than `build` on purpose: an artifact that was not signed cannot reach
PyPI, because the job that uploads it does not start until the job that signs it has succeeded.

`sbom` and `trace` are gated on repository variables and are skipped, not failed, while those
variables are empty. A skipped job is green. They become live the moment the variables are set, with
no change to this file or to the workflow.

The check inside `build` compares versions: the built wheel's version must equal the
tag, compared as PEP 440 objects rather than as strings, so `python-v0.1.0-rc1` and `python-v0.1.0rc1`
are the same answer while a genuinely different version still fails. A mismatch means `hatch-vcs` saw
a dirty tree or no tag, and publishing that would put a `.dev` or `0.0.0` artifact on PyPI forever.

## The two environments

`sign`, `sbom` and `trace` are bound to the `release-signing` environment. It carries **no required
reviewers**: it is a credential scope rather than an approval gate, naming the AWS role those jobs
may assume over OIDC. `pypi` is where a person approves, and two prompts for one release would teach
everybody to click through both.

Signing is required rather than gated, because the Garasign credentials are the one piece of this
that has actually been provisioned. If the AWS trust policy does not yet permit this repository to
assume the signing role, a tag fails at `sign` — loudly, and before anything reaches PyPI.

## The approval

`publish` is bound to the `pypi` environment, and that environment has required reviewers. A tag does
not publish: the run stops at `publish` and waits for a human to approve the deployment in the
Actions UI. That pause is the last stop before the irreversible step, and it is the one part of this
process you cannot discover by reading the repository: the environment and its reviewer list live in
repository settings, not in a file.

**The environments do not exist yet.** The repository has none at all, and GitHub creates a missing
one on first reference *with no reviewers* — so a tag pushed today would reach the upload unapproved.
Create `pypi` with its reviewer list, and `release-signing` without one, before the first tag.

The PyPI trusted publisher **is** registered: project `strands-mongodb`, owner `mongodb`, repository
`strands-mongodb`, workflow `release-python.yml`, environment `pypi`. The owner is what makes the tag
have to be pushed to `mongodb/strands-mongodb`: the OIDC claim carries the repository the workflow
ran in, so a tag pushed to a fork mints a claim PyPI will reject.

If the run appears to hang at `publish`, it is not hung. It is waiting for somebody, and if nobody
who can approve is watching, it waits until the run expires.

## Credentials

There are none. The upload uses PyPI's trusted publishing over OIDC: no API token exists in this
repository, so there is nothing to leak or rotate. PyPI verifies the identity of that workflow file,
in that repository, running in the `pypi` environment, and mints a token good for one upload.

Rename the workflow file or the environment and the upload stops working until the publisher on PyPI
is updated to match. That coupling is intended: it is what ties the upload to this workflow.

## When it goes wrong

| Symptom | What happened | What to do |
|---|---|---|
| No run started | The tag did not match `python-v*` | Delete the tag locally and on the remote, tag again with the prefix |
| `gate` or `contract` red | The commit was not releasable | Fix on `main`, delete the tag from the remote, tag the new commit |
| `wheel is X, tag says Y` | `hatch-vcs` saw a dirty tree or no tag | Tag an exact commit on a clean checkout |
| Stuck at `publish` | Waiting for a required reviewer | Approve the deployment, or find somebody who can |
| `sign` red, AWS "not authorized to perform sts:AssumeRoleWithWebIdentity" | The trust policy does not cover this repository | Ask DevProd Release Infrastructure to add it; nothing has reached PyPI |
| `sbom` or `trace` skipped | Their repository variable is empty | Expected while the credentials behind them are unprovisioned |
| Wrong artifact uploaded | It is on PyPI permanently | Yank that version on PyPI and release a new patch. There is no replace |
