"""Create or delete the throwaway Atlas cluster the ``integration`` and ``autoembed`` tiers run against.

CI runs this around the Atlas tiers, and you can run it the same way locally::

    python scripts/atlas_ci_cluster.py up                 # prints the name and the connection string
    python scripts/atlas_ci_cluster.py down strands-ci-<suffix>

``up`` creates an M10 replica set on the newest MongoDB release, with storage auto-scaling on (automated
embedding requires it), a database user and an access-list entry for this machine's address. The user and
the entry expire on their own after :data:`LIFETIME`, so nothing outlives a run whose teardown never ran.
It returns only once the cluster's search process answers, because the suite skips, rather than fails, a
search test whose search process is unreachable.

``down`` deletes the cluster and its user, and succeeds if they are already gone. ``up`` also deletes any
cluster this script created more than :data:`LIFETIME` ago, which is the backstop for a runner that died
before its teardown step ran. CI names the cluster from the run before ``up`` starts (``ATLAS_CI_CLUSTER``),
so teardown can delete it even when ``up`` failed half way.

Authentication is the Atlas CLI's own: ``MONGODB_ATLAS_CLIENT_ID`` and ``MONGODB_ATLAS_CLIENT_SECRET`` for a
service account with the Project Owner role, or a logged-in ``atlas auth login`` session. The project is
``MONGODB_ATLAS_PROJECT_ID``.

In GitHub Actions the password and the connection string are masked before anything else is printed, and
reach later steps as the ``name`` and ``uri`` step outputs, never through a command line.
"""

from __future__ import annotations

import datetime
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from typing import Any, Final

__all__ = ["LIFETIME", "PREFIX", "cluster_config", "connection_uri", "expired", "main"]

PREFIX: Final = "strands-ci-"
"""Name prefix of every cluster this script creates, and the only clusters it ever deletes."""

LIFETIME: Final = datetime.timedelta(hours=3)
"""How long a cluster, its user and its access-list entry may live. A full run takes under an hour."""

_SEARCH_WAIT_SECONDS: Final = 600

Runner = Callable[[Sequence[str]], str]


def _atlas(arguments: Sequence[str]) -> str:
    """Run one Atlas CLI command and return its standard output.

    Args:
        arguments: The arguments after ``atlas``.

    Returns:
        What the command printed.

    Raises:
        subprocess.CalledProcessError: If the command fails. Its stderr is passed through to ours.
    """
    try:
        completed = subprocess.run(["atlas", *arguments], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        # Captured so `down` can recognise CLUSTER_NOT_FOUND, and printed so a CI log shows why.
        print(f"atlas {' '.join(arguments[:2])} failed: {error.stderr or error.stdout}", file=sys.stderr)
        raise
    return completed.stdout


def cluster_config(name: str, *, region: str = "US_EAST_1") -> dict[str, Any]:
    """Describe the cluster ``up`` creates, as the Admin API's create-cluster request body.

    Args:
        name: The cluster name.
        region: The AWS region.

    Returns:
        The request body the Atlas CLI reads from ``--file``.
    """
    return {
        "name": name,
        "clusterType": "REPLICASET",
        # The newest release, so the weekly run is the canary for a server change before users meet it.
        "versionReleaseSystem": "CONTINUOUS",
        "terminationProtectionEnabled": False,
        "tags": [{"key": "purpose", "value": "strands-mongodb-ci"}],
        "replicationSpecs": [
            {
                "regionConfigs": [
                    {
                        "providerName": "AWS",
                        "regionName": region,
                        "priority": 7,
                        "electableSpecs": {"instanceSize": "M10", "nodeCount": 3},
                        # Automated embedding on a dedicated cluster requires storage auto-scaling.
                        "autoScaling": {"diskGB": {"enabled": True}},
                    }
                ]
            }
        ],
    }


def expired(clusters: Sequence[dict[str, Any]], *, now: datetime.datetime) -> list[str]:
    """Pick the clusters this script created that are older than :data:`LIFETIME`.

    Args:
        clusters: Cluster descriptions as ``atlas clusters list -o json`` reports them.
        now: The current time, timezone-aware.

    Returns:
        Their names. A cluster without this script's prefix is never included, whatever its age.
    """
    names = []
    for cluster in clusters:
        name = str(cluster.get("name", ""))
        created = cluster.get("createDate")
        if not name.startswith(PREFIX) or not created:
            continue
        if now - datetime.datetime.fromisoformat(str(created).replace("Z", "+00:00")) > LIFETIME:
            names.append(name)
    return names


def connection_uri(standard_srv: str, username: str, password: str) -> str:
    """Put the credentials into the cluster's SRV connection string.

    Args:
        standard_srv: ``mongodb+srv://<host>`` as Atlas reports it.
        username: The database user.
        password: Its password. Hex only, so it needs no percent-encoding.

    Returns:
        A connection string the test tiers can use.

    Raises:
        ValueError: If ``standard_srv`` is not an SRV connection string.
    """
    scheme = "mongodb+srv://"
    if not standard_srv.startswith(scheme):
        raise ValueError("the cluster reported no mongodb+srv:// connection string")
    host = standard_srv[len(scheme) :].rstrip("/")
    return f"{scheme}{username}:{password}@{host}/?retryWrites=true&w=majority"


def _wait_for_search(uri: str) -> None:
    """Wait until the cluster's search process answers ``$listSearchIndexes``.

    Args:
        uri: The cluster's connection string.

    Raises:
        TimeoutError: If search does not answer within :data:`_SEARCH_WAIT_SECONDS`.
    """
    from pymongo import MongoClient
    from pymongo.errors import OperationFailure, PyMongoError

    deadline = time.monotonic() + _SEARCH_WAIT_SECONDS
    client: MongoClient[dict[str, Any]] = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    try:
        while True:
            try:
                list(client["strands_ci_probe"]["probe"].list_search_indexes())
                return
            except OperationFailure as error:
                # Code 26 is "namespace not found": the call reached the search process and it answered.
                if error.code == 26:
                    return
            except PyMongoError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"the cluster's search process did not answer within {_SEARCH_WAIT_SECONDS}s")
            time.sleep(10)
    finally:
        client.close()


def _publish(name: str, uri: str) -> None:
    """Hand the cluster to the rest of the run: GitHub step outputs in CI, export lines otherwise.

    Args:
        name: The cluster name.
        uri: Its connection string.
    """
    outputs = os.environ.get("GITHUB_OUTPUT")
    if outputs:
        print(f"::add-mask::{uri}")
        with open(outputs, "a", encoding="utf-8") as handle:
            handle.write(f"name={name}\nuri={uri}\n")
        print(f"cluster {name} is ready")
        return
    print(f"cluster {name} is ready. Run the tiers, then: python scripts/atlas_ci_cluster.py down {name}")
    print(f"export MONGODB_ATLAS_URI='{uri}' MONGODB_AUTOEMBED_URI='{uri}'")


def up(run: Runner = _atlas, *, now: datetime.datetime | None = None) -> str:
    """Create the cluster, its user and an access-list entry, and wait until search answers.

    Args:
        run: Runs one Atlas CLI command. Replaced in tests.
        now: The current time, for tests.

    Returns:
        The cluster name.

    Raises:
        ValueError: If ``ATLAS_CI_CLUSTER`` names a cluster without this script's prefix.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    listed = json.loads(run(["clusters", "list", "-o", "json"]) or "{}").get("results", [])
    for stale in expired(listed, now=now):
        print(f"deleting {stale}, left over from an earlier run")
        run(["clusters", "delete", stale, "--force"])

    name = os.environ.get("ATLAS_CI_CLUSTER") or f"{PREFIX}{now.strftime('%Y%m%d%H%M%S')}"
    if not name.startswith(PREFIX):
        raise ValueError(f"ATLAS_CI_CLUSTER must start with {PREFIX!r}, so that down and the reaper can find it")
    password = secrets.token_hex(24)
    if os.environ.get("GITHUB_OUTPUT"):
        # Masked before any command runs, so an error message that echoed it would show it masked.
        print(f"::add-mask::{password}")
    delete_after = (now + LIFETIME).strftime("%Y-%m-%dT%H:%M:%SZ")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(cluster_config(name, region=os.environ.get("ATLAS_CI_REGION", "US_EAST_1")), handle)
    try:
        print(f"creating {name}; an M10 takes about ten minutes. If this run stops, delete it with: down {name}")
        run(["clusters", "create", "--file", handle.name, "--watch", "--watchTimeout", "1800"])
    finally:
        os.unlink(handle.name)

    run(
        [
            "dbusers", "create",
            "--username", name,
            "--password", password,
            "--role", "atlasAdmin@admin",
            "--scope", name,
            "--deleteAfter", delete_after,
        ]
    )  # fmt: skip
    run(["accessLists", "create", "--currentIp", "--deleteAfter", delete_after, "--comment", name])

    described = json.loads(run(["clusters", "connectionStrings", "describe", name, "-o", "json"]))
    uri = connection_uri(described["standardSrv"], name, password)
    _wait_for_search(uri)
    _publish(name, uri)
    return name


def down(name: str, run: Runner = _atlas) -> None:
    """Delete the cluster and its user. The access-list entry expires on its own.

    Args:
        name: The cluster name ``up`` printed.
        run: Runs one Atlas CLI command. Replaced in tests.

    Raises:
        ValueError: If ``name`` is not a cluster this script creates.
        subprocess.CalledProcessError: If deleting the cluster fails for a reason other than its absence.
    """
    if not name.startswith(PREFIX):
        raise ValueError(f"refusing to delete {name!r}: this script only deletes clusters named {PREFIX}*")
    try:
        run(["clusters", "delete", name, "--force"])
    except subprocess.CalledProcessError as error:
        if "CLUSTER_NOT_FOUND" not in f"{error.stdout}{error.stderr}":
            raise
        print(f"{name} does not exist, so there is nothing to delete")
    try:
        run(["dbusers", "delete", name, "--force"])
    except subprocess.CalledProcessError:
        print(f"the user {name} was already gone, or expires on its own")
    print(f"teardown of {name} complete")


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``up`` or ``down``.

    Args:
        argv: The arguments after the script name.

    Returns:
        A process exit code.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["up"] and len(arguments) == 1:
        up()
        return 0
    if arguments[:1] == ["down"] and len(arguments) == 2:
        down(arguments[1])
        return 0
    if arguments == ["down"] and os.environ.get("ATLAS_CI_CLUSTER"):
        down(os.environ["ATLAS_CI_CLUSTER"])
        return 0
    print("usage: atlas_ci_cluster.py up | down [<name>]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
