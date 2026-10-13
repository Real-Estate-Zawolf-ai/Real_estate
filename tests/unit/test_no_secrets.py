"""Nothing credential-shaped may be committed, and the example env file must stay empty."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Shapes of real provider credentials (Groq, Google, Meta, OpenAI-style, AWS, Slack, GitHub, private keys).
SECRET_SHAPES = re.compile(
    r"(AIza[0-9A-Za-z_\-]{30,}|gsk_[A-Za-z0-9]{20,}|EAA[A-Za-z0-9]{40,}|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[abp]-[0-9A-Za-z-]{20,}|ghp_[A-Za-z0-9]{30,}"
    r"|postgres(?:ql)?(?:\+\w+)?://[^\s:@/<>]+:(?!password|pass|secret|\*|\[|<|\$)[^\s@/<>]{6,}@[a-z0-9.-]+\.(?:neon\.tech|aws|azure|amazonaws\.com))"
)
SCANNED_DIRS = ("app", "scripts", "alembic")
SCANNED_FILES = ("render.yaml", "Dockerfile", "docker-compose.yml", "vercel.json", ".env.example", "README.md", "alembic.ini")
SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".jpeg", ".docx", ".csv", ".sqlite3"}


def tracked_text_files():
    for name in SCANNED_FILES:
        path = ROOT / name
        if path.exists():
            yield path
    for directory in SCANNED_DIRS:
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix not in SKIP_SUFFIXES and "__pycache__" not in path.parts:
                yield path


def test_no_file_contains_a_credential_shaped_string():
    leaks = []
    for path in tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        leaks += [f"{path.relative_to(ROOT)}: {match.group(0)[:12]}..." for match in SECRET_SHAPES.finditer(text)]
    assert leaks == []


def test_the_example_env_file_has_no_values_for_secret_settings():
    secret_names = (
        "SESSION_SECRET", "ADMIN_API_KEY", "DATABASE_URL", "POSTGRES_PASSWORD", "QDRANT_API_KEY", "GROQ_API_KEY",
        "GEMINI_API_KEY", "WHATSAPP_VERIFY_TOKEN", "WHATSAPP_APP_SECRET", "WHATSAPP_ACCESS_TOKEN", "WHATSAPP_GATEWAY_SECRET",
        "TELEGRAM_BOT_TOKEN", "SMTP_PASSWORD", "STAFF_WEBHOOK_URL",
    )
    values = dict(
        line.split("=", 1) for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    filled = {name: values[name] for name in secret_names if values.get(name, "").strip()}
    assert filled == {}


def test_real_env_files_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignore and ".env.*" in ignore and "!.env.example" in ignore


def test_the_docker_context_excludes_env_files_and_history():
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert {".env", ".env.*", ".git"} <= set(ignore)


def test_the_detector_itself_recognises_real_looking_keys():
    fake_google = "AIza" + "SyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY"
    fake_groq = "gsk_" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123"
    fake_dsn = "postgresql://owner:realpassword99@ep-cool-1.eu-central-1.aws.neon.tech/db"
    for sample in (fake_google, fake_groq, fake_dsn):
        assert SECRET_SHAPES.search(sample), sample
    for harmless in ("postgresql://user:password@localhost/db", "postgresql://aqar:aqar_test_pw@127.0.0.1:55432/aqar_test", "gsk_[A-Za-z0-9]{10,}"):
        assert not SECRET_SHAPES.search(harmless), harmless
