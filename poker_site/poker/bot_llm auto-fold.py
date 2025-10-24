import sys
sys.path.append('../pokerlib')

from argparse import ArgumentParser
from pokerlib import Player, PlayerSeats
from pokerlib import Table
from pokerlib import HandParser as HandParser
from pokerlib.enums import Rank, Suit
from collections import OrderedDict, defaultdict
from decimal import Decimal, ROUND_HALF_UP, getcontext
import random
import copy
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import os, json, math, asyncio
from decimal import Decimal, InvalidOperation
from typing import Literal, Optional, Dict, Any, List
import httpx
from pydantic import BaseModel, Field, ValidationError
import random
import string
import os, json, re, asyncio, traceback
from decimal import Decimal, InvalidOperation
from openai import OpenAI
from dotenv import load_dotenv
from .fold_range import *
# inside LLMPokerBot
import asyncio, traceback
from .postflop_rules import *

# Note on Architecture of Bot
# In order to speed up decisions ->
# Simple decisions (fold obvious trash) → Rule-based (instant)
# Standard plays (pot odds, equity) → Traditional poker solver (fast)
# Complex spots (multi-way, ICM, opponent reads) → LLM (slower but smart)

import httpx
import os
import json
import asyncio
import traceback

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class LLMError(Exception):
    pass

async def llm_call_openrouter(model: str, messages, *, timeout_sec: float = 1.2) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://yourapp.com",
        "X-Title": "HomeGame Poker",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 64,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},  # <-- force JSON if supported
    }

    timeout = httpx.Timeout(timeout_sec, connect=0.6, read=timeout_sec, write=0.6)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise LLMError(f"bad LLM response structure: {data}")

    # Try strict JSON first
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        # permissive fallback
        import re
        m = re.search(r"\{.*?\}", content, flags=re.S)
        if not m:
            raise LLMError(f"no JSON object in content (len={len(content)}): {content!r}")
        obj = json.loads(m.group(0))

    action = (obj.get("action") or "").strip().lower()
    amount = str(obj.get("amount") or "0")
    norm = action.replace("-", "").replace(" ", "")
    if norm in ("allin", "shove", "jam"):
        action = "all-in"
    elif norm not in ("fold", "check", "call", "raise", "allin"):
        raise LLMError(f"unknown action: {action}")

    return {"action": action, "amount": amount}





JSON_OBJ = re.compile(r"\{.*?\}", re.S)
def _extract_json(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("LLM response is not text")
    m = JSON_OBJ.search(text.strip())
    if not m:
        raise ValueError("No JSON object found in response")
    return json.loads(m.group(0))

def _money(x) -> Decimal:
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def flatten_to_string(data):
    """Safely flatten lists/dicts into a readable string."""
    if isinstance(data, (dict, list)):
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return str(data)
    else:
        return str(data)

# put this at module level (top of file), not inside the class
ENGINE_WANTS_TOTAL_FOR_CHECK = True  # your engine wants total-to-price (e.g., 0.1) instead of 0 for a legal check

from functools import lru_cache
from .equity_mc import *
import asyncio
import traceback

def _canon_cards(cards):
    """
    Turn [(Rank, Suit), ...] into a stable, hashable, sortable tuple of ints.
    Never compares Enum objects directly; always uses .value (or int fallback).
    """
    print('_canon_cards')
    out = []
    for (r, s) in cards:
        rv = getattr(r, "value", r)
        sv = getattr(s, "value", s)
        out.append((int(rv), int(sv)))
    # Sort by (rank_value, suit_value) for a deterministic order
    out.sort()
    return tuple(out)

def _eq_key(hand, board, players, trials):
    """
    Final cache key: ((h1,h2,...), (b1,b2,...), players, trials)
    where each hi/bi is a (rank_val, suit_val) pair.
    """
    print('_eq_key')
    return (_canon_cards(hand), _canon_cards(board), int(players), int(trials))

@lru_cache(maxsize=10000)
def _equity_cache(key):
    print('_equity_cache')
    # DEBUG breadcrumb — should always print on first unique key
    print("_equity_cache: starting for key", key[:2], "players/trials=", key[2:])
    hand_key, board_key, players, trials = key

    # Reconstruct (Rank, Suit) tuples from ints for the MC fn
    # Import enums locally to avoid circulars
    from pokerlib.enums import Rank, Suit
    hand = [(Rank(r), Suit(s)) for (r, s) in hand_key]
    board = [(Rank(r), Suit(s)) for (r, s) in board_key]

    # NOTE: you can bump trials here; leaving as passed-through
    return estimate_equity_mc(hand, board, num_opponents=players,
                              trials=trials, rank7_fn=rank7_with_handparser)

class LLMPokerBot:
    def __init__(self, player_id, balance, table):
        self.player_id = player_id
        self.balance = balance
        self.out_of_balance = False
        self.hand = []
        self.currentbet = 0
        self.handscore = None
        self.hand_forscore = None
        self.table = table
        self.name = f'player #{self.player_id[-5:]}'
        self._bal_lock = asyncio.Lock()

    def __repr__(self):
        return (f'{self.name}')
    
    async def calc_equity(self, opponents: int = 1, trials: int = 1000) -> float:
        """
        Asynchronously estimate equity via Monte Carlo (cached).
        Use ~800–1500 trials for quick decision gates.
        """
        print('calc_equity')
        try:
            # Snapshot current state
            hand = tuple(self.hand)
            board = tuple(getattr(self.table, "board", ()))
            key = _eq_key(hand, board, opponents, trials)
        except Exception as e:
            print("calc_equity: key build failed:", repr(e))
            traceback.print_exc()
            return 0.0

        loop = asyncio.get_running_loop()
        try:
            # Use functools.partial-like lambda; surface exceptions explicitly
            result = await loop.run_in_executor(None, lambda: _equity_cache(key))
            print("calc_equity: got result", result)
            return float(result)
        except Exception as e:
            print("calc_equity: executor failed:", repr(e))
            traceback.print_exc()
            return 0.0
    
    async def auto_fold(self):
        try:
            # Hand sanity
            if not isinstance(self.hand, (list, tuple)) or len(self.hand) != 2:
                print("AUTO FOLD SKIP: no 2-card hand yet")
                return None

            pos = self.table.position_of_player(self)

            # Only use the range if it's truly FIRST-IN preflop
            if not self.table.is_unopened_pot_preflop():
                print(f"AUTO FOLD SKIP: not unopened preflop (pos={pos})")
                return None

            # In an unopened pot, BB should not auto-fold (they can check)
            if pos == "BB":
                print("AUTO FOLD SKIP: BB unopened (free check).")
                return None

            # Fast membership check (normalized inside helper)
            should_fold = is_fold_combo(pos, tuple(self.hand))

            # (Optional) concise debug
            try:
                label = canonical_label(*self.hand)
            except Exception:
                label = "??"
            print(f"AUTO FOLD CHECK: pos={pos}, hand={self.hand} [{label}] → fold? {should_fold}")

            if should_fold:
                return {"action": "fold"}
            return None

        except Exception as e:
            print("AUTO FOLD BROKEN:", repr(e))
            return None

    async def decide(self, game_state) -> Dict[str, str]:
        async def decide_action(game_state):
            prompt = (
                "You are a poker agent playing No-Limit Texas Hold'em in a server runtime.\n"
                "Return ONLY a strict JSON object with this schema:\n"
                "{\n"
                '  "action": "fold" | "check" | "call" | "raise" | "all-in",\n'
                '  "amount": "<decimal string, required only if action == \'raise\'>"\n'
                "}\n\n"
                "Rules:\n"
                "- Output a single JSON object and nothing else (no prose, no code fences).\n"
                "- Obey legality constraints provided in the input.\n"
                '- \"amount\" is the TOTAL bet size (not delta) when you choose \"raise\".\n'
                "- If check is legal, prefer check over fold.\n"
                "- When checking, amount is 0.\n"
                "- Preflop: players who posted blinds and can check must use action=call or raise; if checking is legal, set amount=0.\n"
                "- To call a bet, the TOTAL must equal the current required call size.\n"
                "- Reasonable sizing: 2x–3x preflop; 50–120% pot postflop unless constrained.\n"
                "- Use \"all-in\" only when committing your entire stack (no partial all-ins).\n"
                "BELOW IS THE LOG OF THE GAME. Return the JSON only:\n"
            )

            load_dotenv()
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                print("LLM DECIDE ERROR: OPENROUTER_API_KEY missing/empty")
                return {"action": "check", "amount": "0"}  # conservative fallback

            print("LLM: key present, prefix:", api_key[:6] + "…")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key.strip(),
            ).with_options(timeout=40.0)

            user_content = prompt + flatten_to_string(game_state)
            print("LLM DECIDE payload length:", len(user_content))

            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model="x-ai/grok-4-fast",
                messages=[{"role": "user", "content": user_content}],
                extra_headers={"HTTP-Referer": "https://yourapp.com", "X-Title": "HomeGame Poker"},
            )

            content = completion.choices[0].message.content if completion.choices else ""
            print("LLM DECIDE RAW:", content)

            # Parse JSON (with fallback extraction)
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = _extract_json(content)

            print("LLM DECIDE PARSED:", data)

            action = (data.get("action") or "").strip().lower()
            amount = str(data.get("amount") or "0")

            # Normalize synonyms
            norm = action.replace("-", "").replace(" ", "")  # e.g., "all-in" -> "allin"
            if norm in ("allin", "shove", "jam"):
                action = "all-in"
            elif norm in ("fold", "check", "call", "raise"):
                # keep as is
                pass
            else:
                # Unknown → safe default
                return {"action": "check", "amount": "0"}

            return {"action": action, "amount": amount}

        # --- PREFLOP fast path (auto-fold) ---
        if self.table.preflop:
            try:
                action = await self.auto_fold()  # <-- call it
                if not action:
                    print('AUTO FOLD BYPASSED - HAND: ', self.hand)
                    return await decide_action(game_state)
                else:
                    print('AUTO FOLD - ', action)
                    return action
            except Exception as e:
                print('LLM PREFLOP BROKEN:', repr(e))
                return {"action": "check", "amount": "0"}

        # --- Postflop or fallback ---
        try:
            return await decide_action(game_state)
        except Exception as e:
            print("LLM DECIDE EXCEPTION:", e.__class__.__name__, str(e))
            traceback.print_exc()
            return {"action": "check", "amount": "0"}

    async def placebet(self, current_price, valid=True, cancel_event=None):
        print('PLACEBET LLM ENTERED')

        def _needs_total_for_check(preflop: bool, my_total: Decimal, target_total: Decimal) -> bool:
            # Only preflop, and only when we've already posted money this street (i.e., a blind).
            # Example: BB with my_total == target_total == 0.1 → engine expects 0.1, not 0.
            return preflop and (my_total > 0) and (target_total >= my_total)

        MAX_ATTEMPTS = 6
        SLEEP_BETWEEN = 0.6

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if cancel_event and cancel_event.is_set():
                print("PLACEBET: cancel_event set; aborting.")
                return 0

            target_total = Decimal(str(current_price))   # table's current price (TOTAL)
            my_total     = Decimal(str(self.currentbet)) # my TOTAL on this street
            my_balance   = Decimal(str(self.balance))    # my stack
            allowed_to   = my_total + my_balance         # max TOTAL I can reach this street

            private_game_log_round = self.table.events_for_round(self, self.table.round)

            raw = await self.decide(private_game_log_round)
            print(f'LLM BOT BETTING (attempt {attempt}): ', raw)

            bet_total = None

            if isinstance(raw, dict):
                action = (raw.get("action") or "").strip().lower()
                norm = action.replace("-", "").replace(" ", "")
                if norm in ("allin", "shove", "jam"):
                    action = "all-in"

                try:
                    amt = _money(raw.get("amount") or "0")
                except Exception:
                    amt = Decimal("0")

                if action == "fold":
                    return Decimal("-1")

                if action == "check":
                    if _needs_total_for_check(self.table.preflop, my_total, target_total):
                        # Return the TOTAL-to-price, capped by what we can reach
                        return min(target_total, allowed_to)
                    return Decimal("0")

                if action == "call":
                    # If we can't fully cover, this returns our allowed_to (short all-in).
                    needed_total = max(target_total, my_total)
                    return min(needed_total, allowed_to)

                if action == "all-in":
                    return allowed_to  # TOTAL to-price

                if action == "raise":
                    bet_total = amt  # TOTAL (not delta)

            else:
                # legacy numeric path
                try:
                    bet_total = Decimal(str(raw))
                except (InvalidOperation, TypeError, ValueError):
                    self.table.private_ledger_event(
                        self,
                        'Invalid bet: expected {"action":"fold|check|call|raise|all-in","amount":"<total>"}'
                    )
                    await asyncio.sleep(SLEEP_BETWEEN)
                    continue

            # ---- Common normalization for numeric TOTAL (bet_total) ----
            if bet_total == Decimal("-1"):
                return Decimal("-1")

            if bet_total == Decimal("0"):
                if _needs_total_for_check(self.table.preflop, my_total, target_total):
                    return min(target_total, allowed_to)
                return Decimal("0")

            # Never go backwards
            if bet_total < my_total:
                bet_total = my_total

            # Snap under-calls up to the call size (if covered)
            if my_total < bet_total < target_total and allowed_to >= target_total:
                bet_total = target_total

            # Cap at ALL-IN if oversize
            if bet_total > allowed_to:
                bet_total = allowed_to

            return bet_total  # TOTAL to-price

        # Failsafe
        target_total = Decimal(str(current_price))
        my_total = Decimal(str(self.currentbet))
        if target_total <= my_total:
            return Decimal("0")
        return Decimal("-1")


from .utils_poker import *

def _normalize_decide_output(raw, target_total, my_total, allowed_to):
    try:
        if isinstance(raw, dict):
            a = (raw.get("action") or "").strip().lower().replace(" ", "").replace("-", "")
            amt = D(raw.get("amount") or "0")
            if a in ("allin","jam","shove"): return {"action":"all-in","amount": str(allowed_to)}
            if a == "fold":  return {"action":"fold","amount":"0"}
            if a == "check": return {"action":"check","amount":"0"} if target_total <= my_total else {"action":"call","amount": str(target_total)}
            if a == "call":  return {"action":"call","amount": str(max(target_total, my_total))}
            if a == "raise": return {"action":"raise","amount": str(amt)}
        else:
            amt = D(str(raw))
            if amt == D("-1"): return {"action":"fold","amount":"0"}
            if amt == D("0"):  return {"action":"check","amount":"0"} if target_total <= my_total else {"action":"call","amount": str(target_total)}
            return {"action":"raise","amount": str(amt)}
    except Exception:
        pass
    return None

def _snap_to_legal(act, gs):
    a = (act.get("action","") or "").lower().replace(" ", "").replace("-", "")
    amt = D(act.get("amount","0"))
    target = D(gs.current_price)
    my_total = D(gs.my_total)
    allowed_to = D(gs.my_total) + D(gs.my_balance)

    if a == "allin": return {"action":"all-in","amount": str(allowed_to)}
    if a == "fold":  return {"action":"fold","amount":"0"}
    if a == "check": return {"action":"check","amount":"0"} if target <= my_total else {"action":"call","amount": str(target)}
    if a == "call":  return {"action":"call","amount": str(max(target, my_total))}
    if a == "raise":
        min_total = min_raise_total(gs.current_price, gs.last_raise_delta, gs.my_total, allowed_to)
        total = max(amt, min_total)
        total = min(total, allowed_to)
        return {"action":"raise","amount": str(total)}
    return {"action":"check","amount":"0"}
