# Security policy

## Reporting a vulnerability

**Do not open a public issue, a discussion or a pull request for a security vulnerability.** A public report
discloses the vulnerability before a fix exists.

Report it privately to **mongodb-partners@mongodb.com** instead.

Include as much of this as you have. None of it is required, and the first two are worth more than the rest:

- **What an attacker can do**, in one sentence. Impact is what decides how quickly a report moves.
- **The smallest reproduction you have**, as a script or as a sequence of calls. A failing test written
  against this repository is the most useful form and the hardest to misread.
- **Versions.** This package's, from `pip show strands-mongodb`, and the deployment it ran against: the
  output of `db.version()`, or that it was Atlas and which tier.
- **Whether the finding is public anywhere already**, and any disclosure deadline you are working to.

You do not need a working exploit and you do not need to be certain. Report it anyway.

## What to expect

This is a pre-1.0 package maintained by a small group, so what follows is what the maintainers intend rather
than a service-level agreement:

- an acknowledgement that a person has read the report, rather than an automated receipt;
- an answer on whether the behaviour is a vulnerability, and if it is, what the fix is and roughly when, or a
  reason if there is no date yet;
- credit in the release notes under whatever name you ask for, or none if you would rather have none;
- a request to hold publication until a fixed release exists, with the timing agreed with you rather than
  set for you.

If a report goes unanswered, it is more likely to have been lost than ignored. Send it again.

## What is in scope

This policy covers the `strands-mongodb` package in this repository: the source under `python/src`, the
examples it ships, and the artifacts published to PyPI under that name.

**MongoDB Server, Atlas and the official drivers are not in scope here, because this project's maintainers
cannot fix them.** MongoDB has its own process for those, described in
[Create a Vulnerability Report](https://www.mongodb.com/docs/manual/tutorial/create-a-vulnerability-report/),
which asks for the [vulnerability report webform](https://www.mongodb.com/bug-submission-form) and allows up
to a week to acknowledge a submission. Sending one of those here delays it by however long forwarding takes.

A vulnerability in a dependency this package only carries belongs to that project. The part that is ours is
the version range: if this package's floor or ceiling is what pins you to a vulnerable release, that is worth
telling us about even after you have reported it upstream.

## Supported versions

Fixes land on `main` and ship in the next release. Before v1.0.0 there is no maintained branch for an earlier
minor version, so the supported version is the latest one on PyPI.
