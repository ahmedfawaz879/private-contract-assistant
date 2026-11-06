from __future__ import annotations

import os
from typing import Protocol


class Backend(Protocol):
    name: str
    egress: bool  # True if prompts leave the machine

    def generate(self, items: list[dict]) -> list[dict]: ...


def get_secret(name: str) -> str | None:
    """Environment variable first, then Kaggle secrets if running on Kaggle."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001
        return None
