"""Production configuration is validated before the app serves a single request."""

import pytest

from app.helpers.config import Settings

GOOD = dict(
    APP_ENV="production",
    DATABASE_URL="postgresql://u:p@ep-x.eu-central-1.aws.neon.tech/db?sslmode=require",
    SESSION_SECRET="s" * 40,
    ADMIN_API_KEY="a" * 30,
    CORS_ALLOWED_ORIGINS="https://app.example.com",
    GROQ_API_KEY="gsk_x",
    GEMINI_API_KEY="AIza_x",
    QDRANT_URL="https://abc.eu-central.aws.cloud.qdrant.io:6333",
    QDRANT_API_KEY="qk",
    TRUSTED_PROXY_HOPS=1,
    RATE_LIMIT_ENABLED=True,
)


def make(**overrides) -> Settings:
    # Explicit None must stay explicit: it overrides the variables conftest exports,
    # otherwise a "missing" setting would silently be picked up from the environment.
    return Settings(_env_file=None, **{**GOOD, **overrides})


def test_a_complete_production_configuration_has_no_problems():
    assert make().production_problems() == []


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"DATABASE_URL": None}, "DATABASE_URL"),
        ({"SESSION_SECRET": "short"}, "SESSION_SECRET"),
        ({"SESSION_SECRET": None}, "SESSION_SECRET"),
        ({"ADMIN_API_KEY": "tooshort"}, "ADMIN_API_KEY"),
        ({"ADMIN_API_KEY": None}, "ADMIN_API_KEY"),
        ({"CORS_ALLOWED_ORIGINS": "*"}, "CORS_ALLOWED_ORIGINS"),
        ({"CORS_ALLOWED_ORIGINS": "http://app.example.com"}, "must use https"),
        ({"CORS_ALLOWED_ORIGINS": "https://a.example.com,*"}, "CORS_ALLOWED_ORIGINS"),
        ({"GROQ_API_KEY": None}, "GROQ_API_KEY"),
        ({"GEMINI_API_KEY": None}, "GEMINI_API_KEY"),
        ({"QDRANT_URL": "http://localhost:6333"}, "QDRANT_URL"),
        ({"QDRANT_URL": None}, "QDRANT_URL"),
        ({"QDRANT_API_KEY": None}, "QDRANT_API_KEY"),
        ({"TRUSTED_PROXY_HOPS": 0}, "TRUSTED_PROXY_HOPS"),
        ({"WHATSAPP_ACCESS_TOKEN": "t"}, "WHATSAPP_APP_SECRET"),  # a half-configured channel is an error
    ],
)
def test_each_missing_or_unsafe_setting_is_reported(override, fragment):
    problems = make(**override).production_problems()
    assert any(fragment in problem for problem in problems), problems


def test_problems_never_contain_secret_values():
    settings = make(SESSION_SECRET="short-secret-value", ADMIN_API_KEY="short-admin-value", GROQ_API_KEY=None)
    text = " ".join(settings.production_problems())
    assert "short-secret-value" not in text and "short-admin-value" not in text


def test_gemini_key_is_not_required_when_embeddings_run_locally():
    assert make(GEMINI_API_KEY=None, EMBEDDING_BACKEND="local").production_problems() == []


def test_local_origins_are_fine_in_development_but_flagged_for_production():
    assert Settings(_env_file=None, APP_ENV="development").is_production is False
    assert make(CORS_ALLOWED_ORIGINS="http://localhost:3000").production_problems() == []  # localhost is exempt from https


def test_pasted_quotes_and_whitespace_are_stripped_from_secrets():
    # Hosting dashboards make a trailing newline or quotes easy to paste.
    settings = make(GROQ_API_KEY='  "gsk_abc"\n', SESSION_SECRET=" " + "s" * 40 + " ")
    assert settings.GROQ_API_KEY == "gsk_abc"
    assert settings.SESSION_SECRET == "s" * 40


def test_blank_optional_values_count_as_unset():
    settings = make(DATABASE_URL="", FILE_MAX_SIZE="", POSTGRES_PORT="", ENABLE_API_DOCS="")
    assert settings.DATABASE_URL is None and settings.FILE_MAX_SIZE is None and settings.ENABLE_API_DOCS is None


def test_docs_and_metrics_are_closed_in_production_and_open_in_development():
    assert make().docs_enabled is False and make().metrics_public is False
    dev = Settings(_env_file=None, APP_ENV="development")
    assert dev.docs_enabled is True and dev.metrics_public is True
    assert make(ENABLE_API_DOCS="true").docs_enabled is True  # explicit opt-in wins


def test_staff_channels_are_detected_from_complete_settings_only():
    assert make().staff_channels == []
    assert make(TELEGRAM_BOT_TOKEN="t").staff_channels == []  # chat id missing
    assert make(TELEGRAM_BOT_TOKEN="t", TELEGRAM_CHAT_ID="1", STAFF_WEBHOOK_URL="https://hooks.example/x").staff_channels == [
        "webhook",
        "telegram",
    ]


def test_database_url_helpers_handle_neon_and_special_characters():
    from sqlalchemy.engine import make_url

    from app.database import url as urls

    neon = make(DATABASE_URL="postgres://user:p%40ss%2Fw@ep-1.eu-central-1.aws.neon.tech/neondb")
    sync = urls.sync_url(neon)
    assert sync.drivername == "postgresql+psycopg"
    assert sync.password == "p@ss/w"
    assert sync.query["sslmode"] == "require"  # remote hosts always get TLS

    async_url = urls.async_url(neon)
    assert async_url.drivername == "postgresql+asyncpg" and async_url.query["ssl"] == "require"
    assert "sslmode" not in async_url.query  # asyncpg would reject it

    local = urls.sync_url(make(DATABASE_URL="postgresql://u:p@localhost:5432/db"))
    assert "sslmode" not in local.query

    parts = make(DATABASE_URL=None, POSTGRES_HOST="db.internal", POSTGRES_USERNAME="u", POSTGRES_PASSWORD="p@ss:w/rd",
                 POSTGRES_PORT="5432", POSTGRES_MAIN_DATABASE="app")
    built = urls.sync_url(parts)
    assert make_url(built.render_as_string(hide_password=False)).password == "p@ss:w/rd"  # escaped, so it round-trips

    assert "secret" not in urls.redacted(make_url("postgresql://u:secret@h/db"))


def test_incomplete_postgres_parts_name_what_is_missing_without_values():
    from app.database import url as urls

    with pytest.raises(urls.DatabaseConfigError) as error:
        urls.sync_url(make(DATABASE_URL=None, POSTGRES_HOST="h", POSTGRES_PASSWORD="hunter2"))
    assert "POSTGRES_USERNAME" in str(error.value) and "hunter2" not in str(error.value)


def test_production_without_a_database_is_an_error_not_a_silent_sqlite_fallback():
    from app.database import url as urls

    with pytest.raises(urls.DatabaseConfigError):
        urls.resolve_url(make(DATABASE_URL=None))
