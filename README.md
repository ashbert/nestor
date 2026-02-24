# Nestor — Your Private Family Butler on Telegram

<p align="center"><em>"Very good, Sir. I shall attend to it at once."</em></p>

Nestor is a private, self-hosted Telegram bot that acts as a shared family assistant. Inspired by the unflappable butler from Tintin's Marlinspike Hall, he manages your calendar, sends emails, researches things on the web, and keeps notes — all through natural conversation.

Deploy him on any VPS. Only whitelisted family members can interact with him. Everyone else is silently ignored.

---

## What Nestor Can Do

| Capability | Examples |
|---|---|
| **Shared Calendar** | "Schedule a dentist appointment for Tuesday at 10am" |
| | "What's on the calendar this week?" |
| | "Move tomorrow's meeting to 3pm" |
| **Email** | "Email the school about next week's absence" |
| | "Check if I have any unread emails" |
| | "Read the latest email from Amazon" |
| **Web Research** | "When does school break for Thanksgiving?" |
| | "Find the lunch menu for Lincoln Elementary" |
| | "What's the weather like in Denver this weekend?" |
| **Notes & Docs** | "Save a note about the plumber's phone number" |
| | "What notes do we have about vacation plans?" |
| **Schedule Summaries** | `/today` — today's agenda |
| | `/week` — the week ahead |

Nestor uses tool-calling AI (Anthropic Claude or OpenAI GPT-4) to understand requests and take action. He manages a multi-step workflow internally — checking the calendar, creating events, confirming results — and replies with a concise, butler-appropriate response.

## Architecture

```
Telegram → Whitelist Gate → LLM Brain (agentic tool-calling loop)
                                │
                    ┌───────────┴────────────┐
                    │                        │
            Google Calendar       Google Drive/Docs
            Gmail (SMTP/IMAP)     Web Search
            DateTime              SQLite Memory
```

**Key design choices:**
- **Composable tools** — each capability is a self-contained `BaseTool` subclass. Add new tools in minutes.
- **Provider-agnostic LLM** — swap between Anthropic and OpenAI with one env var.
- **Portable** — Docker, docker-compose, or systemd. All config via environment variables.
- **Zero secrets in code** — everything from `.env` or env vars. Git history is clean.
- **Private by default** — Telegram user ID whitelist. Silent rejection of strangers.

## Prerequisites

- Python 3.12+ (or Docker)
- A Telegram Bot token ([create one via @BotFather](https://t.me/BotFather))
- An Anthropic or OpenAI API key
- A Gmail account for Nestor (for calendar, drive, and email)
- A Google Cloud project with Calendar, Drive, and Docs APIs enabled

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USER/nestor.git && cd nestor

# Configure
cp .env.example .env
# Edit .env — see configuration section below

# Install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Google OAuth (first run only — opens a browser)
python google_auth_setup.py

# Run
python main.py
```

## Docker

```bash
cp .env.example .env    # fill in values
mkdir -p data           # persistent storage
docker compose up -d
```

## systemd (VPS)

```bash
sudo cp nestor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nestor
sudo systemctl start nestor
```

## Configuration

All configuration is via environment variables (or `.env` file). See [`.env.example`](.env.example) for the full list.

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | From @BotFather |
| `ALLOWED_TELEGRAM_IDS` | Yes | Comma-separated Telegram user IDs |
| `LLM_PROVIDER` | No | `anthropic` (default) or `openai` |
| `ANTHROPIC_API_KEY` | If using Anthropic | API key |
| `OPENAI_API_KEY` | If using OpenAI | API key |
| `GMAIL_ADDRESS` | For email | Nestor's Gmail address |
| `GMAIL_APP_PASSWORD` | For email | Gmail App Password ([create one](https://myaccount.google.com/apppasswords)) |
| `GOOGLE_CREDENTIALS_FILE` | For calendar/drive | Path to OAuth `credentials.json` |
| `GOOGLE_CALENDAR_ID` | No | Calendar ID (default: `primary`) |
| `NESTOR_TIMEZONE` | No | IANA timezone (default: `America/Los_Angeles`) |

## Google Cloud Setup

1. Create a [Google Cloud project](https://console.cloud.google.com/)
2. Enable **Google Calendar API**, **Google Drive API**, and **Google Docs API**
3. Configure the **OAuth consent screen** (External, add Nestor's Gmail as a test user)
4. Create **OAuth credentials** (Desktop app) and download `credentials.json`
5. Run `python google_auth_setup.py` to complete the OAuth flow
6. Share Nestor's calendar with your family members

## Gmail Setup (Email)

Nestor uses SMTP/IMAP with a Gmail App Password (no Google Cloud API needed for email):

1. Enable **2-Step Verification** on Nestor's Gmail account
2. Create an [App Password](https://myaccount.google.com/apppasswords)
3. Set `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD` in `.env`

## Getting Telegram User IDs

Message [@userinfobot](https://t.me/userinfobot) on Telegram. It replies with your numeric user ID. Add all family members' IDs (comma-separated) to `ALLOWED_TELEGRAM_IDS`.

## Adding New Tools

Create a file in `nestor/tools/` and subclass `BaseTool`:

```python
from nestor.tools import BaseTool

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get current weather for a location."
    parameters = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
        },
        "required": ["location"],
    }

    async def execute(self, **kwargs) -> str:
        location = kwargs["location"]
        # ... implementation ...
        return f"Weather in {location}: 72°F, sunny"
```

Register it in `main.py`'s `_register_tools()` function. Nestor's LLM will automatically discover and use it.

## Project Structure

```
nestor/
├── main.py                      # Entry point — wires everything together
├── nestor/
│   ├── config.py                # Env-based configuration
│   ├── telegram_handler.py      # Telegram bot + whitelist
│   ├── brain.py                 # Agentic loop (LLM + tools + memory)
│   ├── llm.py                   # Anthropic & OpenAI abstraction
│   ├── memory.py                # SQLite conversation history
│   ├── google_auth.py           # Shared Google OAuth2
│   ├── prompts/system.txt       # Nestor's personality
│   └── tools/
│       ├── __init__.py          # BaseTool + ToolRegistry
│       ├── calendar_tool.py     # Google Calendar CRUD
│       ├── drive_tool.py        # Google Drive/Docs
│       ├── email_tool.py        # Gmail via SMTP/IMAP
│       ├── search_tool.py       # Web search + page fetcher
│       └── datetime_tool.py     # Timezone-aware date/time
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # One-command deployment
├── nestor.service               # systemd unit file
├── google_auth_setup.py         # One-time OAuth helper
└── .env.example                 # Configuration template
```

## Security

- All secrets in `.env` (gitignored) — never in source code
- Telegram whitelist uses immutable numeric user IDs
- Strangers are silently ignored — the bot's existence is never revealed
- Google OAuth tokens stored locally with `600` permissions
- No inbound ports required — Telegram uses long-polling
- Git history contains zero secrets or PII

## License

MIT
