# ABX SSDLC release pipeline for strands-mongodb

Date: 2026-11-08
Branch: `INTPYTHON-1170-SSDLC`
Tickets: INTPYTHON-1125 (epic), INTPYTHON-1111, DEVPROD-44346 (provisioning)

## Goal

Implement the four steps of the ABX release how-to guides in
`mongodb/strands-mongodb`: dependency scanning (SilkBomb/Kondukto), artifact
signing (Garasign), artifact tracing, and PyPI trusted publishing. The shared
action code lives in `abx-release-tools` and is consumed with `uses:`.

## Non-goals

- Adding TypeScript to the monorepo.
- Registering the PyPI pending publisher and creating the `pypi` environment.
  Casey is doing both separately; the workflow is written as if they exist.
- Evergreen. ABX repos release from GitHub Actions.

## Where the action code lives

`strands-mongodb` consumes `caseyclements/abx-release-tools/actions/...`,
hash-pinned. Both that fork and the `aclark4life` upstream are public, so
visibility blocks nothing. The repo is expected to move to a `mongodb` or
`mongodb-labs` org around 2026-11-09/10; at that point the `uses:` owner is
re-pointed and re-pinned, which is a one-line change per call site.

One risk cannot be checked without org admin: whether the `mongodb` org's
Actions allow-list permits a `caseyclements/*` action. `gh api
orgs/mongodb/actions/permissions` returns 403 here. If the org restricts
third-party actions, the `uses:` fails at **runtime**, not at lint. Promotion to
an org repo removes this risk as well.

Hash-pinning is required regardless: `test_release_ci.py` demands a digest for
every action in a job holding `id-token: write`, and MongoDB's zizmor policy
covers neither a `caseyclements/*` nor an `abx-release-tools` owner.

## Part A — harden the actions in abx-release-tools

The actions were extracted verbatim from `drivers-github-tools@99b1cbd`. The
only ABX repo with Garasign signing green in production,
`mongodb-developer/claude-agent-sdk-mongodb-session-store`, does **not** use
them: it has local hardened rewrites. Those rewrites fix real defects the
extraction still carries. We backport them.

The README says to keep semantics aligned with upstream so fixes cherry-pick
easily. Its maintenance rules also say generic fixes go upstream so drift stays
visible. Both are satisfied by fixing here, recording the divergence in each
file's provenance comment, and opening the upstream PRs as tracked follow-up —
not by pretending the upstream PRs are done.

Fixes, in order of severity:

1. **`gpg-sign` word-splits filenames.** `for filename in ${{ inputs.filenames }}`
   breaks on any name containing whitespace. Replace with a NUL-delimited list
   fed over stdin, as the sibling repo does.
2. **`gpg-sign` does not pin the key.** No `--local-user "$GPG_KEY_ID"`, so a
   Garasign account holding several keys can sign with the wrong one and the
   signature verifies against a key nobody expects.
3. **Template injection.** `gpg-sign`, `authorized-pub` and `upload-s3-assets`
   interpolate `${{ inputs.* }}` directly into `run:` bodies. Move every one
   into `env:` and reference it as a shell variable.
4. **No newline validation before `docker --env-file`.** A newline in a
   credential silently corrupts the file and the failure is reported as bad
   auth. Validate presence and reject newlines before writing.
5. **`setup.sh` uses bare `mkdir`.** Fails on a rerun. Use `mkdir -p`.
6. **Secret id handling.** Use the blank-alias form `",<id>"` so a name or a
   full ARN both work, and reject an id containing a comma.

Inputs and action names stay as they are. Only the implementations change.

## Part B — the workflow

`.github/workflows/release-python.yml` keeps its trigger (`push` on tag
`python-v*`), its top-level `permissions: contents: read`, and its
`defaults.run.working-directory: python`.

```
gate ─────┐
          ├─> build ─> sign ─> publish ─> github-release ─> sbom   (gated)
contract ─┘                  └─────────────────────────────> trace (gated)
```

### Jobs

- **`gate`, `contract`, `build`** — unchanged, except that `build` keeps
  uploading exactly the artifact `python-dist` containing only the wheel and
  the sdist. Nothing else may enter `dist/`: `pypa/gh-action-pypi-publish`
  uploads whatever it finds there, so a checksum file in `dist/` would be
  offered to PyPI.
- **`sign`** — `needs: build`. Holds `id-token: write`, bound to a new
  `release-signing` environment with *no* required reviewers (it is a
  credential scope, not an approval gate; `pypi` is the approval gate, and two
  reviewer prompts for one release would be noise). Downloads `python-dist`,
  computes `SHA256SUMS` over the wheel and sdist, signs all three with
  Garasign, and uploads `python-signatures` containing the `.sig` files and
  `SHA256SUMS`.
- **`publish`** — `needs: sign`, so an unsigned artifact can never reach PyPI.
  Otherwise unchanged: `id-token: write`, environment `pypi`, downloads
  `python-dist` into `dist/`, publishes with no `password:`.
- **`github-release`** — `needs: [publish]`, `permissions: contents: write`.
  Downloads both artifacts and attaches the wheel, sdist, `.sig` files and
  `SHA256SUMS` to the GitHub Release for the tag, creating it if absent. Guide 2
  says to attach signatures to the GitHub Release as pymongo does; the repo
  creates no Release today, so this job introduces one. A signature nobody can
  download does not meet the intent of the guide.
- **`sbom`** — `needs: github-release`, gated on `vars.KONDUKTO_SECRET_ID != ''`.
  Generates a CycloneDX SBOM from the project, runs SilkBomb `update` then
  `augment`, and attaches the result to the Release.
- **`trace`** — `needs: publish`, gated on `vars.AWS_RELEASE_BUCKET != ''`.
  Writes `authorized-publication.txt` with `authorized-pub` and uploads it with
  `upload-s3-assets`. Papertrail itself is reachable only from Evergreen and
  Drone hosts inside AWS, never from a GitHub Actions runner, so this S3 report
  is the sanctioned GitHub Actions path, as in `mongo-python-driver`.

Every new job carries `timeout-minutes`.

### Credentials, without `secrets.*`

The suite forbids the string `secrets.` anywhere in the file, comments
included. That assertion is not relaxed. The sign job instead reads
`vars.AWS_SIGNING_ROLE_ARN`, `vars.AWS_SIGNING_REGION` and
`vars.AWS_SIGNING_SECRET_ID`.

This is truthful rather than a dodge. A role ARN, a region, and the *name* of a
secret are not credentials; the OIDC token minted at run time is, and it is
held by `id-token: write` under an environment binding. The sibling repo used
environment secrets for these; this is a deliberate divergence. It also makes
`if: vars.X != ''` usable as a job-level gate, which is what makes the blocked
steps inert rather than absent.

### What is gated and what is not

Signing is **required**, not gated, because Garasign credentials are the one
piece DevProd has actually delivered. SBOM and tracing are leaf jobs gated on
their variables: green today, live the moment the variables are set, with no
second code review.

The consequence, stated plainly: if the AWS OIDC trust policy does not yet
cover `mongodb/strands-mongodb` (it covers `repo:mongodb-developer/*`; see
Blockers), a real tag fails at `sign`. That is the correct failure — loud, and
before anything reaches PyPI — and nothing regresses, because a tag cannot
publish today either.

## Part C — the test suite

`python/tests/test_release_ci.py` reads the workflow as YAML and gates its
structure. Two assertions must change. Both are **narrowed honestly**, keeping
the property they exist to protect; neither is deleted or loosened.

1. **`id-token: write` on exactly one job** (asserted in
   `TestNothingHereIsALongLivedCredential` and again in
   `TestTheActionThatUploadsIsPinnedToContentRatherThanToAPlace`). Counting
   holders was a proxy for blast radius. The property becomes: *every* job
   holding `id-token: write` must be bound to an environment and have every
   `uses:` digest-pinned, and exactly one job may run the PyPI publish action.
   That still forbids an unscoped or unpinned OIDC job while admitting `sign`.

2. **Exactly one `upload-artifact` and one `download-artifact`.** The
   docstring says the point is whether the payload arrives; the pair-counting
   was a vacuity guard. The property becomes: every download names an artifact
   some upload actually wrote, and the publish job's `packages-dir` equals the
   `path` of its own download. Add an assertion that the publish job downloads
   `python-dist` and nothing else, which is what the old count implied.

Everything else holds unchanged: digest pinning, the `secrets.` ban, top-level
`permissions`, `timeout-minutes` on every job, `if-no-files-found: error`.

New assertions to add alongside:

- The job that publishes must `need` the job that signs, transitively or
  directly. This is the rule that keeps signing from being quietly removed.
- A gated job's `if:` must test a `vars.` expression, so "inert" stays visible
  as a deliberate state rather than looking like dead code.
- Every `uses:` pointing at `abx-release-tools` is digest-pinned, in every job,
  not only the OIDC ones.

## Part D — RELEASING.md

`TestSomebodyCanActuallyPerformARelease` requires the doc to name every job,
the tag, and the environment. The new jobs, the `release-signing` environment,
the repository variables and what each gates, and the fact that the GitHub
Release is now part of a release all go in. The existing warning that the
`pypi` environment and the trusted publisher do not yet exist stays until they
do.

## Testing

The workflow cannot be run end to end without a tag, and a tag publishes. So:

- `test_release_ci.py` is the primary check and runs in CI on every push.
- `pre-commit run --all-files` in `abx-release-tools` (check-github-actions
  validates each `action.yml` against its schema; shellcheck covers `setup.sh`).
- The shell changes in `gpg-sign` are exercised locally against a file whose
  name contains a space, to confirm the word-splitting fix.
- First real exercise is a release candidate tag, after the blockers clear.

### There is no dry run, and none should be added

`test_a_tag_push_is_the_only_trigger` forbids `workflow_dispatch`, deliberately:
a manually triggerable publish is a publish anybody with write access can fire.
Do not add one to get a test run, and do not add one temporarily.

So the first execution of this pipeline is a real release. Mitigate it in this
order, not by weakening the trigger:

1. **Smoke-test in a scratch repo first.** A throwaway repo under the same org
   can carry a copy of the workflow, a trivial package and TestPyPI as the
   publish target. That exercises the OIDC trust policy, the Garasign
   container and the action wiring without touching strands-mongodb's name on
   PyPI.
2. **Then tag `python-v0.1.0rc1` on strands-mongodb.** A pre-release is the
   closest thing to a dry run that the trigger permits: `pip install` will not
   resolve to it without `--pre`, so a broken artifact does not reach anyone
   who did not ask for it.

A tag is permanent for this purpose. PyPI does not allow re-uploading a
filename, and deleting a release does not free the version, so `rc1` is spent
once. Expect to burn `rc2`, `rc3` and so on; budget for that rather than
planning to reuse one.

## Blockers needing an administrator

1. **AWS OIDC trust policy** covers `repo:mongodb-developer/*`;
   `mongodb/strands-mongodb` is in the `mongodb` org and is probably not
   covered. *Unverified inference.* Zakhar Kleyman (ProdSec, 2026-11-09) cannot
   read it out of the policy either and asked for an **empirical** answer: run a
   workflow in a `mongodb`-org repository that tries to assume the same role,
   and report what STS says. The scratch-repo smoke test under
   "There is no dry run" answers this as a side effect, which is the cheapest
   way to get it; the alternative is waiting on his CloudTrail pull.
   Note the scratch repo must be in the **`mongodb`** org to answer the
   question — one in `mongodb-developer` is already covered and proves nothing
   about `mongodb`.
2. **Kondukto** RNDSEC onboarding ticket for the repo, plus a CI API token.
   (#ask-product-security.) Zakhar reports Kondukto is **owned by Product
   Security now and may be being deprecated** — "Silk" in SilkBomb is the
   vendor before Kondukto, and the tooling turns over often. Ask what the
   successor is before filing an onboarding ticket for a tool on its way out.
   This does not change the workflow: the `sbom` job is gated on
   `vars.KONDUKTO_SECRET_ID` and stays skipped until somebody sets it. What
   changes is only which ticket to file and against which tool.
3. **S3 bucket** agreed with DevProd Release Infrastructure. Not a *release*
   bucket — Zakhar: release buckets are CDN-connected, and this is an
   **Evergreen** bucket, which is the right kind for the Papertrail
   authorized-publication report. He recalls a self-service route through
   Backstage and is unsure whether it is still supported. The workflow reads
   `vars.AWS_RELEASE_BUCKET`, whose name is now inaccurate; rename it when the
   bucket is actually provisioned, so the variable and the thing match.
4. **Papertrail credentials** — the guide does not confirm delivery
   (DEVPROD-44346).
5. **`pypi` environment with required reviewers**, and the PyPI pending
   publisher. Casey is handling these.
6. **`release-signing` environment.** The repo has no environments at all.
7. **Signed release tags** plus a tag ruleset blocking delete and force-push
   (Guide 4, step 4). Not in `RELEASING.md` today; the `git-sign` action covers
   the signing half.

Items 1 and 6 block the first real tag. Items 2 through 4 only keep the gated
jobs inert.
