# Contributing

Thanks for your interest in this project. Contributions of any size are welcome: a bug report, a fix, a
documentation correction, a new feature.

Read this first, and read [AGENTS.md](AGENTS.md) before you write code. This file is the process. That one
is the conventions the code has to follow, the commands, the test tiers, and the reasoning behind the
layout. It also lists what not to do, which is the shorter and more useful list.

## Reporting a bug

Search the existing issues first, including closed ones. When you open a new one, include:

- A reproducible test case or a sequence of steps.
- The version of this package, of `strands-agents`, and of Python.
- Which MongoDB deployment you are on: Atlas, an Atlas CLI local deployment, or a self-managed server, and
  its version. Behaviour differs across the three, and the search tiers differ most.
- Anything unusual about the environment, and whether you have already tried the latest release.

Please do not report a security vulnerability in a public issue. See [Security](#security).

## Asking for a feature

Open an issue describing the problem before the solution. Say what you are trying to do and what is in the
way. If the request is about an API this package exposes, say which SDK surface you are wiring it into,
because both surfaces here implement protocols owned by the Strands Agents SDK and what they can do is
bounded by those.

Many things that look absent are deliberate rather than overlooked: they are open questions, or out of
scope for v0.1.0, with the reasoning recorded against them.

## Setting up an environment

You need Python 3.10 or newer and, for anything beyond the default tier, a MongoDB deployment.

```bash
git clone git@github.com:mongodb/strands-mongodb.git
cd strands-mongodb/python
pip install hatch
hatch run test
```

That is where the repository lives today. It is expected to move to a MongoDB-owned organisation before the
first release, and the URLs in `pyproject.toml` already point at the intended home rather than at this one,
with a note beside them recording that the organisation is unconfirmed. Clone from wherever you found this
file.

Hatch builds the environment from `pyproject.toml`, so there is nothing else to install. If you would
rather use a plain virtualenv:

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
python -m pytest
```

Either way, every command runs from `python/`, not from the repository root.

## Running the tests

Most of the suite needs no deployment at all: around 700 tests in roughly 50 seconds. The tiers that need
one are gated on an environment variable and skip when it is empty, so a first run on a laptop with no
cluster is green.

| Tier | Gate | Needs |
|---|---|---|
| default | none | Nothing |
| `contract` | `MONGODB_URI` | Any MongoDB deployment |
| `integration` | `MONGODB_ATLAS_URI` | Atlas, or an Atlas CLI local deployment with `mongot` |
| `autoembed` | `MONGODB_AUTOEMBED_URI` | A cluster with an Atlas embedding model registered |
| `build` | `STRANDS_MONGODB_BUILD_TIER` | Network only. Builds a wheel and installs it into an empty venv |

Put the URIs in a `.env` file at the repository root, which is gitignored. Never commit one, and never
paste a connection string into an issue, a test or a document: a test in the suite walks every Markdown
file in the repository looking for exactly that.

Because `.env` is loaded automatically, a bare test run on a machine that has one starts the contract tier.
`AGENTS.md` explains how to opt out, along with how long an index build and an embedding actually take, and
those two figures are worth reading before you conclude that a cluster test has hung.

## What a pull request has to pass

Run this before pushing. It is the same four checks the first two CI jobs run, in the same order, except
that it formats the tree in place where CI only reads it:

```bash
hatch run prepare
```

Then the jobs in `.github/workflows/ci-python.yml`. Four gate a change; two run on the weekly cron:

| Job | Runs |
|---|---|
| `Lint` | `hatch run format-check`, `hatch run lint`, `hatch run typecheck` |
| `Test - Python <version>` | The default tier on 3.10, 3.11, 3.12 and 3.13. No deployment, so no secrets |
| `Contract tier` | `hatch run test -m contract` against a `mongo:8` service container. No secret: the tier's definition is any real MongoDB, and a container is one |
| `Coverage floors` | `hatch run cov`: 90% aggregate and 70% per module, both enforced |
| `Latest SDK compatibility` | Weekly, against the newest published `strands-agents` rather than the pinned one. Typechecks, then runs `compat`, then runs `contract` |
| `Atlas tiers` | Weekly. `hatch run test -m "integration or autoembed"` against two repository secrets |

The two tiers that need a cluster run on the schedule rather than on your pull request, because a pull
request from a fork receives no secret and the resulting skip would report as a pass. So CI will not
exercise search behaviour on your branch. A pull request that touches it should say which tiers the
author ran locally, with `MONGODB_URI` and `MONGODB_ATLAS_URI` pointing at a deployment of their own.

## Sending the pull request

1. Work against `main` and keep the change focused on one thing.
2. **Add or update a test.** A change with no test is a change nobody will be able to keep.
3. **Write the requirement down if behaviour changed.** `AGENTS.md` explains the order: the requirement
   first, then the test that would falsify it, then the code.
4. Use a conventional commit subject: `feat:`, `fix:`, `docs:`, `test:`, `refactor:` or `chore:`.
5. Run `hatch run prepare` and make sure it is clean.
6. Open the pull request against `main`, describe what changed and why, and mention any issue it closes.
7. Stay engaged with review. Squash and rebase if asked.

## Code of conduct

<!-- TODO(code-of-conduct): adopt a code of conduct for this repository and link it here. The AWS template
this package is modelled on points at the Amazon Open Source Code of Conduct, which is the wrong document
to publish under a MongoDB repository's name. Blocks the first public release, not the code. -->

This project expects the ordinary standard: be respectful, assume good faith, and keep discussion on the
work. The formal code of conduct is `TODO(code-of-conduct)` and will be linked here before the first
release.

## Security

Do not open a public issue for a security vulnerability.

<!-- TODO(security-contact): confirm the reporting address for this repository. The template names the AWS
vulnerability reporting page, which would route a report about this package to the wrong company. -->

Report it privately instead. The reporting address is `TODO(security-contact)` until this repository's home
is settled.
Until then, contact the maintainers directly rather than filing publicly, and please do not create a public
issue describing the vulnerability.

## Licensing

This project is licensed under Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Contributions are accepted under the same license. Larger changes may need a Contributor License Agreement,
and a maintainer will say so on the pull request if it applies.
