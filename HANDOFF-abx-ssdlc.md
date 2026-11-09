# ABX SSDLC — where this stands

Last updated: 2026-11-09, end of day.
Branch: `INTPYTHON-1170-SSDLC` (strands-mongodb). Tickets: INTPYTHON-1125 (epic),
INTPYTHON-1111, DEVPROD-44346.
Spec: `docs/superpowers/specs/2026-11-08-abx-ssdlc-design.md` — read it first; this
file is only the state, the spec is the reasoning.

## Done

### abx-release-tools (two PRs open, awaiting Alex Clark)

| PR | Branch | What |
|---|---|---|
| #1 | `INTPYTHON-1111-claude-sdk-lessons` | Six hardening fixes backported from `mongodb-developer/claude-agent-sdk-mongodb-session-store`, plus the README rewrite |
| #2 | `INTPYTHON-1111-action-tests` | Tests over the actions, CI on every push, shellcheck over the `run:` bodies |

#2 is stacked on #1 but had to be based on `main`: a cross-fork PR cannot use a
base branch that does not exist in the target repo. The body says so.

58 tests pass; pre-commit clean. Upstream has no PR template — both bodies hand-written.

### strands-mongodb (committed, NOT pushed)

`0173087` — the four SSDLC jobs. `8aec36d` — ungating the SBOM.

```
gate ────┐
         ├─> build ─> sign ─> publish ─> github-release ─> sbom
contract ┘                  └─────────────────────────────> trace (gated)
```

`publish` needs `sign`, not `build`: an unsigned artifact cannot reach PyPI
because the uploading job does not start until the signing job has succeeded.

`trace` is gated on `vars.AWS_RELEASE_BUCKET` and is skipped (green) until it is set.
`sbom` is **not** gated — only its SilkBomb/Kondukto steps are, on
`vars.KONDUKTO_SECRET_ID`. A release ships `cyclonedx.sbom.json` from the first tag.

`python/tests/test_release_ci.py`: **39 passed**. Five assertions were narrowed (not
deleted) where the new jobs made their locators ambiguous; two were added for the SBOM
gating. Each narrowed test's docstring says what it replaces and why.

`python/RELEASING.md` names every job, both environments and the dependency chain.

## Blocked, and on whom

1. **Write + admin on `mongodb/strands-mongodb`.** Requested; the original author is
   in India, reply expected 2026-11-10. `gh api repos/mongodb/strands-mongodb --jq
   .permissions` currently returns pull-only. This blocks *everything* below: creating
   the two environments needs admin, pushing the release tag needs push.
2. **AWS OIDC trust policy.** Covers `repo:mongodb-developer/*`; almost certainly not
   `mongodb/strands-mongodb`. Zakhar Kleyman (ProdSec) cannot read it out of the policy
   either and is pulling CloudTrail. See "the rc1 tag is the test" below.
3. **S3 bucket.** Zakhar: this is an **Evergreen** bucket, not a CDN-connected release
   bucket. He recalls a Backstage self-service route, unsure if still supported.
   Note our `AWS_RELEASE_BUCKET` follows pymongo's `pymongo-release-assets` naming, per
   the DBX wiki — worth saying back to him, as it changes which team provisions it.
4. **Kondukto.** Owned by Product Security now and possibly being deprecated. Ask
   #ask-product-security what the successor is *before* filing an RNDSEC onboarding
   ticket. Costs nothing to wait: the augmentation steps are gated and the SBOM ships
   regardless.
5. **Papertrail credentials** (DEVPROD-44346). Papertrail is Evergreen/Drone only —
   GHA runners are off the internal network, no Kanopy (DEVPROD-36740). The S3 report
   is the sanctioned substitute, not a workaround we invented.

## Done, not blocked

PyPI **pending publisher** registered: project `strands-mongodb`, owner `mongodb`,
repo `strands-mongodb`, workflow `release-python.yml`, environment `pypi`. The owner
is why the tag must be pushed to `mongodb/strands-mongodb` — the OIDC claim carries
the repo the workflow ran in, so a tag on a fork mints a claim PyPI rejects.

## Next steps, in order

1. **Push `INTPYTHON-1170-SSDLC` and open the PR.** Needs explicit approval to push.
   The fork `caseyclements/strands-mongodb` is **private** (parent is INTERNAL), so a
   teammate who can read the parent still 404s on fork URLs — link the PR, not blobs.
2. **When permissions land, create two environments** in repo settings:
   - `pypi` — required reviewers. The approval gate.
   - `release-signing` — **no** reviewers. A credential scope, not a gate; two prompts
     for one release teaches everybody to click through both.
3. **Set `vars.AWS_SIGNING_ROLE_ARN`, `AWS_SIGNING_REGION`, `AWS_SIGNING_SECRET_ID`.**
4. **Tag `python-v0.1.0rc1`.** This is the test, not a scratch repo — see below.
5. Reply to Zakhar with the bucket-naming point and the Kondukto successor question.

## The rc1 tag is the OIDC test

There is no dry run and none should be added: `test_a_tag_push_is_the_only_trigger`
forbids `workflow_dispatch` deliberately, because a manually triggerable publish is a
publish anyone with write access can fire.

A scratch repo was considered and **dropped**. The workflow is already ordered as its
own dry run: `sign` runs before `publish`, so a tag whose `AssumeRoleWithWebIdentity`
is rejected dies at `sign` — red, nothing uploaded, and the `rc1` version still
unclaimed on PyPI, since a version is only consumed by a *successful* upload. That
failure is the empirical evidence Zakhar asked for, written from a real `mongodb`-org
repo. A scratch repo would have to be in `mongodb` to prove anything, and creating one
there needs the same permission we are already waiting on.

If it fails, fix the policy and tag `rc2`. Do not try to delete and re-push a tag.
PyPI never frees a filename, so each rc is spent once — budget for rc2, rc3.

## Authoritative sources

- How-to guides: ABX Project: Release How-To Guides (Alex Clark, gdrive)
- DBX wiki, incl. the S3 IAM policy and the Kondukto-token precedent:
  https://wiki.corp.mongodb.com/spaces/DRIVERS/pages/427464463
- Why GHA cannot reach Papertrail: DEVPROD-36740
- Reference implementation: `mongodb-developer/claude-agent-sdk-mongodb-session-store`
- People: Zakhar Kleyman (ProdSec/DevProd), Alex Clark (abx-release-tools)
