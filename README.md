# Nestor — The Butler Bot

A private Telegram butler bot inspired by Nestor from Tintin's Marlinspike Hall.
Manages a shared family Google Calendar, takes notes in Google Drive, and
researches things on the web — all through natural conversation.

## Prerequisites

- Python 3.12+
- A Telegram Bot token (from [@BotFather](https://t.me/BotFather))
- A Google Cloud project with Calendar, Drive, and Docs APIs enabled
- An Anthropic or OpenAI API key

## Quick Setup

```bash
# 1. Clone & enter
git clone <repo-url> nestor && cd nestor

# 2. Configure
cp .env.example .env
# Edit .env with your actual values

# 3. Create venv & install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Google OAuth (first run only)
# Place your credentials.json in the project root
# On first run, a browser window opens for OAuth consent
# The token is cached in token.json for subsequent runs

# 5. Run
python main.py
```

## Docker Setup

```bash
cp .env.example .env   # fill in values
mkdir -p data          # for persistent DB + tokens
docker compose up -d
```

## systemd Setup (VPS)

```bash
sudo cp nestor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nestor
sudo systemctl start nestor
```

## Google Cloud Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g. "Nestor Butler")
3. Enable these APIs:
   - Google Calendar API
   - Google Drive API
   - Google Docs API
4. Go to **Credentials** → **Create Credentials** → **OAuth client ID**
   - Application type: **Desktop app**
   - Download the JSON and save as `credentials.json` in the project root
5. Add your Gmail accounts as test users under **OAuth consent screen** (while in testing mode)

### Calendar Sharing

Log into Nestor's Gmail account and share the primary calendar with
your and your wife's Google accounts (Settings → Share with specific people).

## Getting Telegram User IDs

Send a message to [@userinfobot](https://t.me/userinfobot) on Telegram.
It will reply with your numeric user ID. Do this for both you and your wife,
then add both IDs (comma-separated) to `ALLOWED_TELEGRAM_IDS` in `.env`.

## Architecture

```
Telegram → telegram_handler.py (whitelist gate)
         → brain.py (agentic loop)
           → llm.py (Anthropic/OpenAI with tool calling)
           → tools/ (calendar, drive, search, datetime)
           → memory.py (SQLite conversation history)
```

**Key files:**

| File | Purpose |
|---|---|
| `main.py` | Entry point — wires everything together |
| `nestor/config.py` | Env-based configuration |
| `nestor/telegram_handler.py` | Telegram bot with whitelist |
| `nestor/brain.py` | Agentic loop: LLM + tools + memory |
| `nestor/llm.py` | Anthropic & OpenAI provider abstraction |
| `nestor/memory.py` | SQLite conversation & metadata store |
| `nestor/google_auth.py` | Shared Google OAuth2 (single token, all scopes) |
| `nestor/tools/` | Extensible tool system |
| `nestor/prompts/system.txt` | Nestor's personality & instructions |

## Adding New Tools

1. Create a file in `nestor/tools/` (e.g. `weather_tool.py`)
2. Subclass `BaseTool`:

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
        # ... your implementation ...
        return f"Weather in {location}: 72°F, sunny"
```

3. Register it in `main.py`'s `_register_tools()` function.

## Security Notes

- **All secrets** live in `.env` (gitignored) or environment variables — never in code
- **Telegram whitelist** uses immutable numeric user IDs, not usernames
- **Strangers are silently ignored** — the bot's existence is not revealed
- **Google tokens** are stored locally and gitignored
- **No inbound ports** needed — Telegram uses polling mode
