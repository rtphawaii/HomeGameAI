# HomeGameAI

HomeGameAI is a real-time, multiplayer No-Limit Texas Hold’em Poker platform powered by Django Channels, WebSockets, and LLM-driven poker bots.
It supports both human and AI players, runs seamlessly in the browser, and allows each player to interact with a live game environment hosted on a scalable backend.

## Demo & Gameplay
https://vimeo.com/1131859523

[Lobby Screen](https://i.imgur.com/PfiQJZD.png)

[Gameplay](https://i.imgur.com/aZjJkV8.png)

## 🚀 Features

🎮 Real-time gameplay — built on Django 5 + Channels (ASGI) with Daphne and WebSockets for synchronized multiplayer.

🧠 Intelligent Poker Bots — the LLMPokerBot uses large language models (via OpenRouter/OpenAI API) to evaluate game states and make decisions (fold/check/call/raise/all-in).

📊 Poker Engine — full implementation of Texas Hold’em logic: betting rounds, blinds, pot management, and showdown resolution.

💾 State Management — game states are serialized and cached, allowing reproducibility and LLM policy/action/meta caching for repeated scenarios.

🌐 Web Frontend — clean, responsive HTML/CSS UI with a light minimal theme and animated elements for cards, chips, and player interactions.

🖥️ Production Ready — deployed on DigitalOcean using systemd, Nginx, and Gunicorn/Daphne with .env-based configuration.

## 🏗️ Tech Stack
Layer	Tools
Backend	Django 5.x, Channels, Daphne, ASGI
Frontend	HTML5, CSS3, Vanilla JS (WebSocket-driven)
AI Logic	Python (asyncio, httpx, pydantic, decimal), OpenRouter/OpenAI API
Deployment	Nginx, systemd, DigitalOcean Ubuntu Droplet
Data Handling	JSON state caching, custom logging for policy/meta/action events

## 🧠 LLM Poker Bot (Overview)

The LLMPokerBot uses both rule-based heuristics and LLM inference to make real-time betting decisions:

Calculates equity via Monte Carlo simulations (equity_mc.py)

Evaluates game state context (pot odds, stack depth, position, board texture)

Caches previous policy → action → meta decisions for reuse

Queries fast LLMs (e.g. deepseek, grok-4-fast) with structured JSON prompts

Returns deterministic action outputs

## ⚙️ Local Setup
1️⃣ Clone and install dependencies
git clone https://github.com/<yourusername>/HomeGameAI.git
cd HomeGameAI/poker_site
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2️⃣ Create .env file

Example:

DEBUG=True
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=127.0.0.1,localhost
OPENROUTER_API_KEY=your_openrouter_api_key
GATE_PASSWORD=your_gate_password

3️⃣ Run migrations and collect static files
python manage.py migrate
python manage.py collectstatic --noinput

4️⃣ Run server locally
daphne -b 0.0.0.0 -p 8000 poker_site.asgi:application

## 🧩 Roadmap

 Integrate reinforcement learning and offline LLM fine-tuning

 Add in-browser game replays and summaries

 Expand caching for post-flop decision reuse

 Introduce leaderboard and bankroll tracking

## 🤝 Contributing

Pull requests are welcome!
Please open an issue first to discuss major changes.

Fork the repo

Create your feature branch: git checkout -b feature/new-idea

Commit your changes: git commit -m "Add new feature"

Push to branch: git push origin feature/new-idea

Submit a PR

