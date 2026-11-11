# ABX SSDLC release provisioning: what `mongodb/strands-mongodb` needed, and why

Written 2026-11-10, after provisioning this repo for Garasign release signing.
Source material for a DEVPROD-44346 comment and for the ABX release how-to guide.

The workflow was already written and merged (PR #56). Everything below is the
*provisioning* half: the AWS and GitHub state a new ABX repo needs before a
signed release can run. It took far longer than the workflow did, almost entirely
because the facts below were not written down anywhere.

**This file is the status document for the ABX SSDLC work in this repository.**
It is the one that is version-controlled and reviewable, and it is the source for
the DEVPROD-44346 write-up and the ABX release how-to guide. The reasoning behind
the design is in `docs/superpowers/specs/2026-11-08-abx-ssdlc-design.md`; the
operator's view is `python/RELEASING.md`. Working notes under `.claude/` are
laptop-local and gitignored — they are scratch, not a second source of truth, and
anything in them that matters belongs here instead.

## The one role

ABX repos do not get their own IAM role. They share
`arn:aws:iam::857654397073:role/drivers-test-secrets-role` in the Drivers account.

This is worth stating plainly because the name actively misleads. The role is
dual-purpose: Evergreen assumes it for driver test secrets (it is hardcoded in
`mongodb-labs/drivers-evergreen-tools/.evergreen/secrets_handling/setup_secrets.py`),
*and* ABX repos assume it by GitHub OIDC for release signing. Every other release
role in the account follows a `<repo>-release-role` convention, so the natural
assumption is that ABX needs one too, or that a role named `-test-` cannot be the
signing role. Neither is true.

The role is not Terraform-managed. It carries no tags, its description is still
the stock *"Allows EC2 instances to call AWS services on your behalf"* from its
creation in January 2023, and its trust policy has been extended by hand, one ABX
repo at a time. Nothing reverts a console or CLI edit, and nothing records one
either.

Do not confuse this with `arn:aws:iam::901841024863:role/ecr-role-gha-ro`, the ECR
pull role in the devprod account. That one *is* Terraform-managed, in
`mongodb-devprod-infrastructure/devprod-platforms-ecr/iam.tf`, and is changed by PR
plus an env0 apply.

## What the trust policy keys off

The OIDC subject condition is per repo *and per environment*:

```
repo:<org>/<repo>:environment:release-signing
```

The environment name is load-bearing. A job that assumes this role must declare
`environment: release-signing`, spelled exactly that way, or the subject will not
match no matter how correct the role ARN is.

### The `@<id>` forms

Two subject shapes coexist in the policy:

```
repo:mongodb-developer/claude-agent-sdk-mongodb-session-store:environment:release-signing
repo:mongodb-developer@45798453/mlflow-mongodb@1356498874:environment:release-signing
```

GitHub emits the second form — org ID and repo ID appended — for repos created
after the org enabled immutable subject claims. This is what caused the October
failure for the session-store repo, and the comment in `devprod-platforms-ecr/iam.tf`
attributes it to repos migrated between orgs. Either way, which form a given repo
emits is not something you can determine ahead of time from outside.

The cheap answer is to add both. An extra subject that never matches costs
nothing; a missing one costs a failed release at tag-push time, and PyPI never
frees a filename, so the release candidate is spent.

For this repo that meant:

```
repo:mongodb/strands-mongodb:environment:release-signing
repo:mongodb@45120/strands-mongodb@1360224859:environment:release-signing
```

(`mongodb` org ID is `45120`; get a repo ID with `gh api repos/<org>/<repo> --jq .id`.)

## The vault must live under `drivers/`

The SSDLC guide says release secrets belong at `drivers-release/<name>`, and
explicitly warns against `drivers/`, "since those are visible to non-admins."

That guidance cannot be followed with this role. The `drivers-test-secrets` policy
attached to it grants Secrets Manager only on:

```
arn:aws:secretsmanager:*:857654397073:secret:drivers/*
```

A vault at `drivers-release/*` would be created successfully and then be
unreadable by the very role that needs it. ABX's existing vault sits at
`drivers/claude-agent-sdk-mongodb-session-store/release-signing` for this reason,
not by oversight.

**This is a real exposure, not a naming quibble.** Everyone in DBX has read access
to the Drivers account, so every ABX Garasign credential is readable by anyone in
the org — verified by reading the existing vault's key names with nothing but the
`drivers-test-sso` profile. Resolving it means extending the policy to cover
`drivers-release/*` and migrating the vaults, which touches a role more than
twenty repos depend on. It should be its own ticket; it should not be attempted as
a side effect of onboarding one repo.

For now this repo follows the existing precedent:
`drivers/strands-mongodb/release-signing`, holding the same four keys the setup
action requires — `garasign-username`, `garasign-password`, `gpg-key-id`,
`gpg-public-url`.

## Settings, and why they are variables

```
AWS_SIGNING_ROLE_ARN   = arn:aws:iam::857654397073:role/drivers-test-secrets-role
AWS_SIGNING_REGION     = us-east-1
AWS_SIGNING_SECRET_ID  = drivers/strands-mongodb/release-signing
```

All three are **environment variables** on the `release-signing` environment, not
secrets. The reference repo stores them as secrets; this repo deliberately does
not, and a test enforces that `release-python.yml` contains no `secrets.*`
reference anywhere.

None of the three is confidential. An ARN, a region and a vault *name* are
addresses; the credentials they lead to are fetched at runtime over the
OIDC-assumed role, and access is controlled by the trust policy, not by hiding the
names. Storing them as secrets costs real debuggability — a secret is write-only,
so nobody, including the person who set it, can ever read the value back. That is
why the session-store repo's role ARN had to be recovered from IAM rather than
from the repo that uses it.

`gpg-key-id` is a key *inside* the vault, not a GitHub setting. Only the vault name
is ever configured.

## `AWS_RELEASE_BUCKET` is intentionally unset

No ABX bucket exists; the Drivers account holds only the seven
`*-driver-release-assets` buckets. The `trace` job tests `vars.AWS_RELEASE_BUCKET
!= ''` at the *job* level, so leaving it unset skips the job cleanly.

That `if:` has to be job-level rather than step-level, and the variable has to be
repository-level rather than environment-level, because a job-level `if:` is
evaluated before the environment resolves — environment-scoped variables are
invisible to it.

## Minimum checklist for the next ABX repo

1. `gh api repos/<org>/<repo> --jq .id` — note the repo ID.
2. Add both subject forms to `drivers-test-secrets-role`'s trust policy, scoped to
   `:environment:release-signing`.
3. Create `drivers/<repo>/release-signing` with the four Garasign keys.
4. Create the `release-signing` environment (no reviewers needed; a deployment tag
   policy such as `tag:python-v*` is sensible hardening).
5. Set the three variables above on that environment.
6. Create the `pypi` environment and register the PyPI pending publisher, matching
   the environment name as a literal string.

Steps 2 and 3 need AWS access to the Drivers account that a release engineer may
not have; they are the ones to ask for in advance.

## What was changed tonight, exactly

- `drivers-test-secrets-role` trust policy: two subjects appended, nothing else
  altered. Confirmed by diff before applying.
- New secret `drivers/strands-mongodb/release-signing`, seeded with the same
  credentials and the same `ai-builder-experience.pub` team key as the existing ABX
  vault.
- Three environment variables set on `release-signing` in `mongodb/strands-mongodb`.

Untested end to end. The first tag push is the real test of the OIDC claim, and
there is deliberately no dry run: `release-python.yml` triggers on tag push only.
The failure mode is safe — `publish` depends on `sign`, so a rejected
`AssumeRoleWithWebIdentity` fails `sign`, `publish` never starts, nothing reaches
PyPI, and the release candidate version stays unspent.

## Still open

**The pinned `abx-release-tools` commit cannot run in `sign` or `trace`.** Both jobs download an
artifact and check nothing out, and `setup.sh` at `e89529bd` configures the release bot's git
identity with a repository-local `git config`, which fails with `fatal: not in a git directory`.
The script runs under `set -e`, so the whole setup dies there — before `$RELEASE_ASSETS` is even
created, which makes the error surface as a missing directory in a later step. `publish` needs
`sign`, so this fails the tag with nothing uploaded and the version unspent, but it does fail it.
The fix is `git config --global`, proposed upstream as mongodb-labs/abx-release-tools#2; **the pin
in `release-python.yml` has to move to a commit that carries it before the first tag.**

**Whether SilkBomb and Kondukto are still the supported path.** The release ships an SBOM from its
first tag — that part is settled and runs unconditionally. What is not settled is the vendor half.
The upload to Kondukto is gated on `vars.KONDUKTO_SECRET_ID != ''` and is therefore off until a token
exists, which was a deliberate choice: the SBOM is worth generating whether or not anything ingests
it, so the two were separated rather than shipped as one switch.

**TODO: follow `mongo-python-driver`'s lead and write the report to S3.** This is the next step
here, and it does not wait on the Kondukto question above. PyMongo's release calls
`drivers-github-tools/python/post-publish`, which runs `full-report` and then `upload-s3-assets`:
`full-report` assembles the authorized-publication report, the SBOM, the SARIF export and the
compliance report, and the S3 upload is what actually delivers them, because Papertrail is reachable
only from Evergreen and Drone hosts inside AWS and never from a GitHub Actions runner
(DEVPROD-36740). The `trace` job already has that shape — `authorized-pub` followed by
`upload-s3-assets` — and is inert solely because `vars.AWS_RELEASE_BUCKET` is unset. So the work is
not code: it is agreeing a bucket name (see below), creating it, and setting the variable. Widening
`authorized-pub` toward the rest of `full-report` can follow once the destination exists.

Before asking for that token, confirm with DevProd that SilkBomb and Kondukto are what ABX should be
targeting at all. Both predate the current tooling and neither appeared in the ABX material we worked
from; the reference repo does not upload to Kondukto either. If there is a successor, the gated step
should point at it rather than being switched on as written. Until that is answered, treat the
Kondukto path as unproven rather than merely unconfigured.

**The `drivers/` versus `drivers-release/` exposure**, described above. Needs its own ticket.

**The bucket name, if step 3 is ever switched on.** `pymongo-release-assets` is the precedent for a
single-package bucket, and the SSDLC wiki lists `*-driver-release-assets` for the drivers. Neither
pattern obviously covers an ABX integration package, and the name is not worth guessing at — ask
before creating a bucket, since the variable's whole purpose is that the job stays off until one
exists.

**Code Security is disabled on the repository.** `GET /repos/mongodb/strands-mongodb` reports
`code_security: disabled`, and the code scanning API answers 403. Any workflow that files SARIF —
zizmor here, and CodeQL or semgrep if either is ever wired up properly — fails on the upload rather
than on its findings, with `Resource not accessible by integration` well after the scanner has
exited 0. The zizmor job reports through pull request annotations instead, which works but keeps
findings out of the Security tab and leaves nothing to query across repositories.

Enabling it is a per-repository setting under an org licence, so it is a question for whoever holds
that budget rather than something to flip while onboarding one package. Worth asking as part of the
ABX cohort rather than once per repo.
