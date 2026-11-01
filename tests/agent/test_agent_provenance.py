import subprocess

from agent.provenance import code_version
from agent.trace import Trace


def fake_git(outputs: dict[str, str]):
    def git(args: list[str]) -> str:
        return outputs[args[0]]

    return git


def test_clean_tree():
    git = fake_git({"rev-parse": "abc1234\n", "status": ""})
    assert code_version(git) == {"commit": "abc1234", "dirty": False}


def test_dirty_tree():
    git = fake_git({"rev-parse": "abc1234\n", "status": " M src/agent/llm.py\n"})
    assert code_version(git) == {"commit": "abc1234", "dirty": True}


def test_git_missing_gives_none():
    def git(args):
        raise FileNotFoundError("git")

    assert code_version(git) is None


def test_not_a_repo_gives_none():
    def git(args):
        raise subprocess.CalledProcessError(128, ["git", *args])

    assert code_version(git) is None


def test_session_start_records_code_version():
    event = Trace("s").session_start(
        {}, "SYS", "u", "m", 16000, None, code_version={"commit": "abc1234", "dirty": False}
    )
    assert event["code_version"] == {"commit": "abc1234", "dirty": False}
