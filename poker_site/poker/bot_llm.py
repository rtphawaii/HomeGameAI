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
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from .equity_mc import *
import asyncio
import traceback
from decimal import Decimal as D



# Note on Architecture of Bot
# In order to speed up decisions ->
# Simple decisions (fold obvious trash) → Rule-based (instant)
# Standard plays (pot odds, equity) → Traditional poker solver (fast)
# Complex spots (multi-way, ICM, opponent reads) → LLM (slower but smart)

def avatar_from_id(pid: str) -> str:
    choices = [f"generic-{i}.svg" for i in range(1, 9)]
    idx = sum(ord(c) for c in str(pid)) % len(choices)
    return f"/static/avatars/{choices[idx]}"

def _q(x: Decimal) -> Decimal:
    # Quantize to cents for clean strings
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

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

ENGINE_WANTS_TOTAL_FOR_CHECK = True  # your engine wants total-to-price (e.g., 0.1) instead of 0 for a legal check

def _canon_cards(cards):
    """
    Turn [(Rank, Suit), ...] into a stable, hashable, sortable tuple of ints.
    Never compares Enum objects directly; always uses .value (or int fallback).
    """
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
    return (_canon_cards(hand), _canon_cards(board), int(players), int(trials))

@lru_cache(maxsize=10000)
def _equity_cache(key):
    # DEBUG breadcrumb — should always print on first unique key
    print("_equity_cache: starting for key", key[:2], "players/trials=", key[2:])
    hand_key, board_key, players, trials = key

    # Reconstruct (Rank, Suit) tuples from ints for the MC fn
    # Import enums locally to avoid circulars
    hand = [(Rank(r), Suit(s)) for (r, s) in hand_key]
    board = [(Rank(r), Suit(s)) for (r, s) in board_key]

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
        self.name = f'{self.player_id[-5:]}'
        self._bal_lock = asyncio.Lock()
        self.gs = None
        self.avatar_url = avatar_from_id(player_id)

    def __repr__(self):
        return (f'{self.name}')
    
    async def calc_equity(self, opponents: int = 1, trials: int = 3000) -> float:
        """
        Asynchronously estimate equity via Monte Carlo (cached).
        Use ~800–1500 trials for quick decision gates.
        """
        try:
            # Snapshot current state
            hand = tuple(self.hand)
            board = tuple(getattr(self.table, "board", ()))
            key = _eq_key(hand, board, opponents, trials)
        except Exception as e:
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

    async def decide(self, game_state,gs,) -> Dict[str, str]:
        async def decide_action(game_state):
            # Auto-Bet
            equity = max(0.0, min(1.0, float(gs.equity)))
            pot = D(str(gs.pot))
            target_total = D(str(gs.current_price))     # TOTAL price to match this street
            my_total = D(str(getattr(gs, "my_total", self.currentbet)))
            min_raise = max(D("0"), target_total)     # keep as Decimal
            stack = max(D("0"), D(str(gs.my_balance)))
            big_blind = D(str(self.table.bigblind))
            allowed_to = my_total + D(str(self.balance))
            to_call = max(D("0"), _money(gs.current_price) - _money(gs.my_total))
            can_check_local = (to_call == D("0"))  # use this instead of gs.can_check

            can_check = bool(gs.can_check)
            street = gs.street

            #Auto-Bet

            # Pot odds for calling: need equity >= to_call / (pot + to_call)
            if to_call > 0:
                pot_odds = float(to_call / (pot + to_call))
            else:
                pot_odds = 0.0
            edge = equity - pot_odds
            print('POT ODDS: ',pot_odds)
            print('EQUITY: ',equity)
            print('POT ODDS VS EQUITY EDGE: ', edge)

            if can_check_local:
                print('no previous bets - auto bet')
                print('CAN CHECK LOCAL',to_call,gs.current_price,gs.my_total,self.currentbet)
                strong_cut = 0.65 if street != "preflop" else 0.60
                if equity >= strong_cut and stack > 0:
                    if street == "preflop":
                        target = (big_blind * Decimal("2.5"))
                    else:
                        if edge>=0.8:
                            #target=(pot * Decimal("1"))
                            return await decide_action_llm(model_input='grok')
                        else:
                            #target = (pot * Decimal("0.75"))
                            return await decide_action_llm(model_input='deepseek')
                    raise_amount = max(min_raise, _q(target))
                    raise_amount = min(raise_amount, stack)
                    if raise_amount <= Decimal("0"):
                        return {"action": "check", "amount": "0"}
                    return {"action": "raise", "amount": str(_q(raise_amount))}
                return {"action": "check", "amount": "0"}
            else:
                # --- facing a bet ---
                denom = _money(gs.pot) + to_call
                pot_odds = float(to_call / denom) if denom > 0 else None

                # Very weak edge → fold
                if edge < -0.05:
                    return await decide_action_llm(model_input='grok')

                # Slightly negative edge (thin) → conservative call if we can cover
                if -0.05 <= edge < 0.0:
                    print('slightly negative edge')
                    # Ensure Decimals
                    target_total = D(str(gs.current_price))        # TOTAL needed to match
                    my_total     = D(str(getattr(gs, 'my_total', self.currentbet)))
                    allowed_to   = my_total + D(str(self.balance)) # max TOTAL we can reach
                    # If we can't add anything, nothing to do
                    if allowed_to <= my_total:
                        return {"action": "fold"}
                    # Short-stacked: can't reach target_total → shove (engine will convert TOTAL later)
                    if allowed_to < target_total:
                        return {"action": "all-in"}
                    # We can cover → normal call
                    return await decide_action_llm(model_input='deepseek')

                # "edge" in [0, 0.5] -> delegate to LLM
                if 0.0 <= edge <= 0.5:
                    print('edge is small')
                    return await decide_action_llm(model_input='deepseek')  # see below

                # Big edge → raise (or shove if short)
                if edge > 0.5 and edge <= 0.7:
                    print('edge is moderately large')
                    three_x = to_call * Decimal("3")
                    seventyfive_pot = (pot + to_call) * Decimal("0.75")
                    target = max(three_x, seventyfive_pot, min_raise)
                    raise_amt = min(_q(target), stack)
                    if raise_amt >= stack:
                        return {"action": "all-in"}
                    return {"action": "raise", "amount": str(_q(raise_amt))}
                
                if edge>0.7:
                    print('edge is very large')
                    return await decide_action_llm(model_input='grok')  # see below

        async def decide_action_llm(model_input='grok'):   
            print('decide_action_llm') 
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


            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key.strip(),
            ).with_options(timeout=40.0)

            user_content = prompt + flatten_to_string(game_state)
            print("LLM DECIDE payload length:", len(user_content))

            if model_input=='grok':
                model="x-ai/grok-4-fast"
                print('LLM DECIDE - GROK')
            else:
                model='deepseek/deepseek-chat-v3-0324'
                print('LLM DECIDE - DEEPSEEK')

            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
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
        gs=await self.equity_calculation(current_price,valid=True,cancel_event=None)

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

            # inside placebet(), each loop iteration BEFORE calling decide():
            target_total = D(str(getattr(self.table, "currentprice", self.table.currentprice)))  # LIVE total-to-price
            my_total     = D(str(self.currentbet))                                     # LIVE
            my_balance   = D(str(self.balance))
            allowed_to   = my_total + my_balance
            to_call_live = max(D("0"), target_total - my_total)

            print('CAN CHECK LOCAL',self.table.currentprice)

            gs = await self.equity_calculation(target_total, valid=True, cancel_event=None)  # LIVE gs

            private_game_log_round = self.table.events_for_round(self, self.table.round)

            raw = await self.decide(private_game_log_round,gs=gs)
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

                # handle dict action -> numeric TOTAL to return
                if action == "check":
                    if to_call_live == D("0"):
                        print('PLACEBET CHECK ATTEMPT',_needs_total_for_check(self.table.preflop, my_total, target_total),target_total)
                        # BB blind quirk only if truly unraised (strict equality)
                        if _needs_total_for_check(self.table.preflop, my_total, target_total):
                            return min(target_total, allowed_to)
                        return D("0")
                    print('PLACEBET MIN SNAP')
                    # Facing a price: snap "check" to CALL to avoid loops (or return D("-1") to fold)
                    return min(target_total, allowed_to)

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
        target_total = Decimal(str(self.table.currentprice))
        my_total = Decimal(str(self.currentbet))
        if target_total <= my_total:
            return Decimal("0")
        return Decimal("-1")
    
    async def equity_calculation(self, current_price, valid=True, cancel_event=None):
        class GS: pass
        gs=GS()
        gs.street = ("preflop" if self.table.preflop else ("river" if len(self.table.board) == 5 else "turn" if len(self.table.board) == 4 else "flop"))
        gs.pot = self.table.pot
        gs.current_price = current_price
        gs.my_total = self.currentbet
        gs.my_balance = self.balance
        gs.board = list(self.table.board)
        try:
            gs.in_position = (self.table.position_of_player(self) in ("BTN","CO","HJ","BTN/SB"))
        except Exception:
            gs.in_position = True
        gs.last_raise_delta = getattr(self.table, "last_raise_delta", D("0"))
        gs.spr = D(self.balance) / D(self.table.pot) if D(self.table.pot) > 0 else D("999")
        gs.players_left = max(1, len(self.table.order) - 1)
        gs.can_check = (D(current_price) <= D(self.currentbet))
        gs.can_bet = gs.can_check
        gs.can_raise = True
        gs.can_call = not gs.can_check
        gs.facing_raise = False
        gs.facing_checkraise = False
        gs.stack_after = D(self.balance)
        gs.equity = await self.calc_equity(opponents=1 if gs.players_left == 1 else 2, trials=1500)
        gs.multiway = (gs.players_left >= 2)
        return gs



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
