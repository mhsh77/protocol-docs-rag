"""Guard against committing secrets: .env.example must only hold placeholders."""

import re

from docrag.config import PROJECT_ROOT

SECRET_VARS = re.compile(r"^[ \t]*([A-Z_]*(?:KEY|TOKEN|SECRET)[A-Z_]*)[ \t]*=[ \t]*(\S+)", re.M)


def test_env_example_has_no_secret_values() -> None:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    leaked = [name for name, _ in SECRET_VARS.findall(text)]
    assert leaked == [], f".env.example must not contain values for: {leaked}"
