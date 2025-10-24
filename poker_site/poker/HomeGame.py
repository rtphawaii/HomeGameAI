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
from .bot_llm import LLMPokerBot
import string
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import copy
from .fold_range import *

def avatar_from_id(pid: str) -> str:
    choices = [f"generic-{i}.svg" for i in range(1, 9)]
    idx = sum(ord(c) for c in str(pid)) % len(choices)
    return f"/static/avatars/{choices[idx]}"

# Load LLM
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize client (once at startup)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def query_llm_deepseek(prompt_text: str) -> str:
    """
    Send a plain text message to the LLM and return its response.
    """
    try:
        completion = client.chat.completions.create(
            model='deepseek/deepseek-chat-v3-0324',
            #model="x-ai/grok-4-fast",
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            extra_headers={
                "HTTP-Referer": "https://yourapp.com",  # optional
                "X-Title": "HomeGame Poker",            # optional
            }
        )

        return completion.choices[0].message.content
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "⚠️ LLM request failed. Check your API key or network."
    
def query_llm_grok(prompt_text: str) -> str:
    """
    Send a plain text message to the LLM and return its response.
    """
    try:
        completion = client.chat.completions.create(
            model="x-ai/grok-4-fast",
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            extra_headers={
                "HTTP-Referer": "https://yourapp.com",  # optional
                "X-Title": "HomeGame Poker",            # optional
            }
        )

        return completion.choices[0].message.content
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "⚠️ LLM request failed. Check your API key or network."
    
# helper that runs the blocking call in a thread
async def query_llm_bg(prompt_text: str,model='grok') -> str:
    if model=='grok':
        return await asyncio.to_thread(query_llm_grok, prompt_text)
    if model=='deepseek':
        return await asyncio.to_thread(query_llm_deepseek, prompt_text)


#HELPER FUNCTIONS
from decimal import Decimal, ROUND_HALF_UP, getcontext
getcontext().prec = 28

EPS = Decimal("0.0000001")  # tiny epsilon for Decimal compares
CENT = Decimal("0.01")

def _money(x) -> Decimal:
    """Coerce to Decimal with 2dp."""
    if isinstance(x, Decimal):
        return x.quantize(CENT)
    return Decimal(str(x)).quantize(CENT)

def _to_float(d: Decimal) -> float:
    return float(d.quantize(CENT))

def flatten_to_string(data):
    """Safely flatten lists/dicts into a readable string."""
    if isinstance(data, (dict, list)):
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return str(data)
    else:
        return str(data)


class Table():
    def __init__(self,smallblind,bigblind,input,output,send_to_user,send_player_info,send_info_all):
        self.list=[]
        self.perma_list=[]
        self.players_by_id = {}
        self.order=[]
        self.startingorder=[]
        self.pot = Decimal("0.00")
        self.smallblind=smallblind
        self.bigblind=bigblind
        self.currentprice=self.bigblind
        self.bet=[]
        self.board=[]
        self.deck=[]
        self.rank=[]
        self.preflop=True
        self.rivercheck=False
        self.gameover=False
        self.round=1
        self.all_in=[]
        self.output=output
        self.input=input
        self.send_to_user=send_to_user
        self.send_player_info=send_player_info
        self.send_info_all=send_info_all
        self.contributed=defaultdict(Decimal)
        self.private_ledger = defaultdict(list)
        self.ledger=[]
        self.round_to = Decimal("0.00") 

    # ---- Players Bar: serialization + broadcasts ----
    def _serialize_player_brief(self, p) -> dict:
        return {
            "id": str(p.player_id),
            "name": getattr(p, "name", str(p)),
            "balance": float(getattr(p, "balance", 0)),
            "currentbet": float(getattr(p, "currentbet", 0)),
            "avatarUrl": getattr(p, "avatar_url", None),
        }

    def _order_ids(self) -> list[str]:
        return [str(p.player_id) for p in self.order]

    def _dealer_index_for_bar(self) -> int:
        # Your code treats self.order[0] as the Button (dealer)
        return 0

    async def broadcast_players_state(self, *, active_player_id: str | None = None):
        payload = {
            "type": "players_state",
            "players": [self._serialize_player_brief(p) for p in self.perma_list],
            "order": self._order_ids(),                  # order starting at BTN (index 0)
            "dealer_index": self._dealer_index_for_bar()
        }
        if active_player_id is not None:
            payload["active_player_id"] = str(active_player_id)
        await self.send_info_all(payload)

    async def broadcast_turn(self, player):
        await self.send_info_all({
            "type": "turn_update",
            "player_id": str(player.player_id),
        })

    async def broadcast_balance_update(self, players: list | None = None):
        if players is None:
            players = self.perma_list
        balances = {str(p.player_id): float(getattr(p, "balance", 0)) for p in players}
        await self.send_info_all({
            "type": "balance_update",
            "balances": balances
        })

    @property
    def currentprice(self) -> Decimal:
        # always return Decimal total-to-price for this street
        return _money(self.round_to)

    @currentprice.setter
    def currentprice(self, value) -> None:
        # allow legacy code to write currentprice; forward to round_to
        self.round_to = _money(value)

    def is_unopened_pot_preflop(self) -> bool:
        """
        True only if it's preflop, currentprice == BB (no raise),
        and nobody except the blind posters has voluntarily contributed > 0.
        Also treats a limp (non-blind contributing >= BB) as 'opened'.
        """
        if not getattr(self, "preflop", False):
            return False

        try:
            bb = Decimal(str(self.bigblind))
            cur_price = Decimal(str(self.currentprice))
        except Exception:
            return False

        # If someone raised pre, pot is opened
        if cur_price > bb:
            return False

        # Count any non-blind voluntary contribution
        for p in self.order:
            try:
                cb = Decimal(str(getattr(p, "currentbet", 0)))
            except Exception:
                cb = Decimal("0")

            if self._is_blind_poster(p):
                # SB is allowed to have smallblind; BB is allowed to have bigblind
                continue

            # Any non-blind chips in the middle means opened (including limps)
            if cb > 0:
                return False

        # Additionally, guard against limper exactly matching BB (edge engines)
        # If any non-blind has cb >= BB, it's definitely opened
        for p in self.order:
            if not self._is_blind_poster(p):
                cb = Decimal(str(getattr(p, "currentbet", 0)))
                if cb >= bb:
                    return False
                
        return True

    def _position_labels(self):
        """
        Returns a list of position labels aligned with self.order.
        Assumes self.order[0] is the Button (BTN). Supports 2..22 players.
        Heads-up: order[0] = BTN/SB, order[1] = BB.
        """
        n = len(self.order)
        if n < 2 or n > 22:
            raise ValueError(f"Supported players: 2..22 (got {n})")

        if n == 2:
            return ["BTN/SB", "BB"]

        labels = ["BTN", "SB", "BB"]
        m = n - 3  # middle seats
        if m <= 0:
            return labels
        if m == 1:
            middle = ["CO"]
        elif m == 2:
            middle = ["UTG", "CO"]
        else:
            utg_count = m - 2  # leave room for HJ, CO
            utgs = ["UTG"] + [f"UTG+{i}" for i in range(1, utg_count)]
            middle = utgs + ["HJ", "CO"]

        return labels + middle

    def position_of_index(self, idx: int) -> str:
        labels = self._position_labels()
        if idx < 0 or idx >= len(labels):
            raise IndexError("player index out of range")
        return labels[idx]

    def position_of_player(self, player) -> str:
        try:
            idx = self.order.index(player)
        except ValueError:
            raise ValueError("player not found in self.order")
        return self.position_of_index(idx)

    def positions_map(self):
        """Return {player_obj: 'POS', ...} for current order."""
        labels = self._position_labels()
        return {p: pos for p, pos in zip(self.order, labels)}
    
    def _is_blind_poster(self, player) -> bool:
        """True if player is SB or BB for the current hand."""
        labels = self._position_labels()
        try:
            pos = labels[self.order.index(player)]
        except ValueError:
            return False
        return pos in ("SB", "BB", "BTN/SB")  # HU BTN posts SB

    # hand over
    async def _signal_hand_over_safely(self):
        try:
            await self.send_info_all({"action": "hand_over"})
            # Optional: explicitly re-enable the start button
            await self.send_info_all({"action": "start_round_prompt"})
        except Exception:
            pass

    def _make_entry(self, event_data, *, etype: str = "event") -> Dict[str, Any]:
        return {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "round": self.round,
            "type": etype,          # "event" | "marker"
            "event": event_data,    # any shape you already use
        }

    def _add_to_private_ledger(self, player, entry: Dict[str, Any]):
        pid = getattr(player, "player_id", player)  # accepts Player or id
        # use per-player copy to avoid shared dict aliasing
        self.private_ledger[pid].append(copy.deepcopy(entry))

    def ledger_event(self, event_data: Dict[str, Any]):
        entry = self._make_entry(event_data, etype="event")
        self.ledger.append(entry)
        for p in self.perma_list:
            self._add_to_private_ledger(p, entry)
    
    def ledger_event_gen(self, event_data: Dict[str, Any]):
        entry = self._make_entry(event_data, etype="event")
        self.ledger.append(entry)

    def private_ledger_event(self, player, event_data: Dict[str, Any]):
        entry = self._make_entry(event_data, etype="event")
        self._add_to_private_ledger(player, entry)

    def events_for_round(self, player, round_no: int, include_markers: bool = False) -> List[Dict[str, Any]]:
        pid = getattr(player, "player_id", player)
        return [
            e for e in self.private_ledger.get(pid, [])
            if e.get("round") == round_no and (include_markers or e.get("type") != "marker")
        ]

    def all_events_for_round(self, round_no: int, include_markers: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        return {
            pid: [e for e in entries if e.get("round") == round_no and (include_markers or e.get("type") != "marker")]
            for pid, entries in self.private_ledger.items()
        }

    def all_events_from_round(self, round_no: int, include_markers: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        return {
            pid: [e for e in entries if e.get("round", 0) >= round_no and (include_markers or e.get("type") != "marker")]
            for pid, entries in self.private_ledger.items()
        }

    async def send_hint(self, player):
        prompt='you are a poker player and you are given the following information about a game of poker. Based on your hand, the betting order / positioning, the action, the board / run out, provide a strategy and analysis that is concise and 2 sentences long. Here is the game play log: ----> '
        combined = prompt + flatten_to_string(self.private_ledger[player.player_id])
        try:
            resp = await asyncio.wait_for(query_llm_bg(combined,model='grok'), timeout=12)
        except asyncio.TimeoutError:
            resp = "(hint delayed)"
        await self.send_to_user(player.player_id, resp)

    async def game_summary(self):
        if not self.ledger:
            print("[SUMMARY] skipped (empty ledger)")
            return

        prompt = (
            "You are a poker commentator and you are given a game script. "
            "Summarize the highlights of the game, say what players did well and "
            "point out where they made mistakes. Your summary should be concise and "
            "2-4 sentences long, kept to a minimum but expanded if interesting action. "
            "Here is the game script: ---->"
        )
        combined = prompt + flatten_to_string(self.ledger)

        try:
            resp = await asyncio.wait_for(query_llm_bg(combined,model='grok'), timeout=15)
        except asyncio.TimeoutError:
            resp = "(summary delayed)"
        except Exception as e:
            print("[SUMMARY ERROR]", e)
            resp = "(summary unavailable)"

        # ✅ await send directly, don’t fire and forget
        try:
            await self.output(f"[Game Commentator]: {resp}")
        except Exception as e:
            print(f"[SUMMARY SEND ERROR] {e}")

    def createdeck(self):
        '''create the deck'''
        self.deck = [(rank, suit) for rank in Rank for suit in Suit]
        self.shuffledeck()

    def shuffledeck(self):
        random.shuffle(self.deck)

    async def addplayer(self, player):
        '''add a player to the game'''
        self.list.append(player)
        self.perma_list.append(player)
        self.players_by_id[str(player.player_id)] = player   # ✅ index by string key
        self.private_ledger_event(player,f'Your identity is {player}, you will soon receive a hand, if it has your player id then that is your hand')
        await self.send_to_user(player.player_id, f"you are {player}")

    def pickdealer(self):
        '''pick a dealer'''
        self.order=self.list
        random.shuffle(self.order)
        return self.order[0].name

    def deal(self):
        '''deal hands'''
        for x in self.list:
            x.hand.append(self.deck.pop())
            x.hand.append(self.deck.pop())
    
    def _iter_seats(self, start_idx: int, end_idx: int):
        """Yield seats from start_idx up to (but not including) end_idx, wrapping once."""
        n = len(self.order)
        i = start_idx % n
        stop = end_idx % n
        while True:
            yield self.order[i]
            if i == (stop - 1) % n:    # stop right before end_idx
                break
            i = (i + 1) % n
    
    CENT = Decimal("0.01")

    @staticmethod
    def _money(x):
        return Decimal(str(x)).quantize(Table.CENT, rounding=ROUND_HALF_UP)

    def _apply_delta(self, player, delta_dec: Decimal) -> Decimal:
        """Apply a chip delta in Decimal to all ledgers + UI list."""
        delta_dec = _money(delta_dec)
        if delta_dec <= Decimal("0.00"):
            return Decimal("0.00")

        # Decimal ledgers
        self.contributed[player] += delta_dec
        self.pot += delta_dec

        # Update balance
        new_bal = _money(player.balance) - delta_dec
        if new_bal <= Decimal("0.00"):
            new_bal = Decimal("0.00")
            if player not in self.all_in:
                self.all_in.append(player)
        player.balance = _to_float(new_bal)

        # UI history expects floats (your bet_info_update reads self.bet[-1][-1])
        self.bet.append((player, _to_float(delta_dec)))
        return delta_dec

    
    #helper functions - allinmech
    def live_opponents(self, actor):
        """Opponents still in the hand (not folded)."""
        return [
            q for q in self.order
            if q is not actor and not getattr(q, "folded", False)
        ]

    def opponents_with_chips(self, actor):
        """Opponents who can still put chips in (balance > 0)."""
        return [q for q in self.live_opponents(actor) if _money(q.balance) > EPS]

    def max_effective_total_commit(self, actor):
        """
        Maximum *total to* amount this actor can legally commit
        without creating an uncallable raise.
        """
        actor_to_cap = _money(actor.currentbet) + _money(actor.balance)
        opps = self.opponents_with_chips(actor)
        if not opps:
            # Nobody left with chips → actor can only match/check in the main loop
            return actor_to_cap
        opp_caps = [_money(o.currentbet) + _money(o.balance) for o in opps]
        # You can't exceed what the largest remaining opponent could ever call.
        return min(max(opp_caps), actor_to_cap)

    def betting_round_done(self):
        """
        True if all non-folded players are either:
        - matched to current price, or
        - all-in (or out of chips)
        """
        target = _money(self.round_to)
        for p in self.order:
            if getattr(p, "folded", False):
                continue
            # All-in players don't need to match
            if p in self.all_in or _money(p.balance) <= EPS:
                continue
            if _money(p.currentbet) != target:
                return False
        return True
    
    def _is_fold(raw) -> bool:
        try:
            s = str(raw).strip()
            if s == "-1":
                return True
            # also accept numeric -1 / -1.0
            return float(raw) <= -1.0 + 1e-9
        except Exception:
            return False

    async def flop(self):
        '''deals flop'''
        print('flop')
        #await self.output(f"now dealing the flop...")
        #burn one card
        self.deck.pop()
        #deal 3 cards to the board
        self.board.append(self.deck.pop())
        self.board.append(self.deck.pop())
        self.board.append(self.deck.pop())
        print(self.board)
        rank1, suit1 = self.board[0]
        rank2, suit2 = self.board[1]
        rank3, suit3 = self.board[2]
        self.ledger_event_gen(f"the flop is: {rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}, {rank3.name} of {suit3.name}")
        await self.output(f"the flop is: {rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}, {rank3.name} of {suit3.name}")
        

        #send the update to the board
        board_flop=f"{rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}, {rank3.name} of {suit3.name}"
        await self.send_info_all({
                "board": {
                    'board':board_flop
                }})
        
        #update handscores
        await self.update_handscore()

    async def turn(self):
        '''deals turn'''
        print('turn')
        #await self.output(f"now dealing the turn...")
        #burn one card
        self.deck.pop()
        #deal 1 card to the board
        self.board.append(self.deck.pop())
        print(self.board)
        rank1, suit1 = self.board[0]
        rank2, suit2 = self.board[1]
        rank3, suit3 = self.board[2]
        rank4, suit4 = self.board[3]
        self.ledger_event_gen(f"the turn is: {rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}, {rank3.name} of {suit3.name}, {rank4.name} of {suit4.name}")
        await self.output(f"the turn is: {rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}, {rank3.name} of {suit3.name}, {rank4.name} of {suit4.name}")

        #send the update to the board
        board_turn=f"{rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}, {rank3.name} of {suit3.name}, {rank4.name} of {suit4.name}"
        await self.send_info_all({
                "board": {
                    'board':board_turn
                }})
        
        #update hand score
        await self.update_handscore()
        
    async def river(self):
        '''deals river'''
        print('~river')
        #await self.output(f"now dealing the river...")
        #burn one card
        self.deck.pop()
        #deal 1 card to the board
        self.board.append(self.deck.pop())
        print(self.board)
        rank1, suit1 = self.board[0]
        rank2, suit2 = self.board[1]
        rank3, suit3 = self.board[2]
        rank4, suit4 = self.board[3]
        rank5, suit5 = self.board[4]
        print(rank5.name,suit5.name)
        self.ledger_event_gen(f"the river is: {rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}, {rank3.name} of {suit3.name}, {rank4.name} of {suit4.name}, {rank5.name} of {suit5.name}")
        await self.output(f"the river is: {rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}, {rank3.name} of {suit3.name}, {rank4.name} of {suit4.name}, {rank5.name} of {suit5.name}")

        #send the update to the board
        board_river=f"{rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}, {rank3.name} of {suit3.name}, {rank4.name} of {suit4.name}, {rank5.name} of {suit5.name}"
        await self.send_info_all({
                "board": {
                    'board':board_river
                }})
        
        #update hand score
        await self.update_handscore()

    #get player
    def get_player(self, player_id: str):
        """Find a player by exact id; if not found, try base part before '-'."""
        if player_id is None:
            return None
        pid = str(player_id)

        # Fast path via index
        p = self.players_by_id.get(pid)
        if p:
            return p

        # Fallback scan (exact)
        for pool in (self.perma_list, self.list):
            for p in pool:
                if str(p.player_id) == pid:
                    self.players_by_id[pid] = p
                    return p

        # Base-id fallback to handle client ids like "user123-<rnd>"
        if "-" in pid:
            base = pid.split("-", 1)[0]

            p = self.players_by_id.get(base)
            if p:
                # index both for future hits
                self.players_by_id[pid] = p
                return p

            for pool in (self.perma_list, self.list):
                for p in pool:
                    if str(p.player_id) == base:
                        self.players_by_id[base] = p
                        self.players_by_id[pid] = p
                        return p

        return None


    
    def positions(self):
        """
        Returns (dealer_idx, sb_idx, bb_idx) for current self.order.
        Heads-up: dealer IS small blind; other player is big blind.
        3+ players: dealer at 0, SB at 1, BB at 2 (as before).
        """
        n = len(self.order)
        if n < 2:
            raise RuntimeError("Not enough players to compute positions")

        dealer_idx = 0
        if n == 2:
            sb_idx = dealer_idx
            bb_idx = (dealer_idx + 1) % n
        else:
            sb_idx = (dealer_idx + 1) % n
            bb_idx = (dealer_idx + 2) % n
        return dealer_idx, sb_idx, bb_idx

    #bets
    async def bets(self, preflop: bool = False):
        print('betting started')
        self.ledger_event(f'betting has started for this street, the players are {self.order}')

        if self.cancel_event and self.cancel_event.is_set():
            raise asyncio.CancelledError

        # --- tiny helpers (local so it's 100% drop-in) ---
        def _is_fold(raw) -> bool:
            try:
                s = str(raw).strip()
                if s == "-1":
                    return True
                return float(raw) <= -1.0 + 1e-9
            except Exception:
                return False

        # Reset per-street meters
        for p in self.perma_list:
            if not preflop or (p is not getattr(self, "small_blind_player", None)
                            and p is not getattr(self, "big_blind_player", None)):
                p.currentbet = Decimal("0.00")

        # Street target (float for UI, Decimal for math)
        if preflop:
            self.round_to = float(self.bigblind)
            if hasattr(self, "small_blind_player"):
                self.small_blind_player.currentbet = float(self.smallblind)
            if hasattr(self, "big_blind_player"):
                self.big_blind_player.currentbet = float(self.bigblind)
            await self.bet_info_update(blinds_start=True)
        else:
            self.round_to = 0.0
            await self.bet_info_update(new_bets=True)

        # Find starting index
        n = len(self.order)
        dealer_idx, sb_idx, bb_idx = self.positions()
        if preflop:
            found_index = (bb_idx + 1) % n
            end_index   = found_index
        else:
            found_index = (dealer_idx + 1) % n
            end_index   = found_index

        if found_index is None:
            return

        continue_loop = True
        while continue_loop:
            if self.cancel_event and self.cancel_event.is_set():
                raise asyncio.CancelledError

            continue_loop = False
            restart = False
            seats = list(self._iter_seats(found_index, end_index))

            for player in seats:
                # hand ended by folds
                if len(self.order) <= 1:
                    self.ledger_event('all players have folded except 1')
                    return

                # no action for all-in players
                if player in self.all_in:
                    self.ledger_event(f'{player} is all-in and has no action')
                    await self.output(f'{player} is all-in and has no action')
                    continue

                # ---------- NORMAL INPUT LOOP PER PLAYER ----------
                while True:
                    self.ledger_event(f'{player} is now starting to bet')
                    
                    if self.cancel_event and self.cancel_event.is_set():
                        raise asyncio.CancelledError

                    target_dec     = _money(self.round_to)            # price to call (street target)
                    current_to_dec = _money(player.currentbet)        # player's current total on this street
                    balance_dec    = _money(player.balance)
                    allowed_to     = current_to_dec + balance_dec     # max this player can reach (to-price)
                    to_call_dec    = target_dec - current_to_dec

                    # let UI know it's their turn
                    await self.broadcast_turn(player)
                    await self.send_to_user(player.player_id, {"action": "your_turn", "target": float(target_dec)})

                    raw = await player.placebet(float(target_dec), cancel_event=self.cancel_event)

                    # --- FOLD (first) ---
                    if _is_fold(raw):
                        self.order.remove(player)
                        self.ledger_event(f'{player} has folded')
                        await self.send_to_user(player.player_id, {"action": "turn_end"})
                        await self.output(f"{player} has folded")
                        await self.broadcast_players_state()
                        if len(self.order) == 1:
                            self.gameover = True
                            return "hand_over"
                        break  # next player

                    # --- parse numeric if not fold; keep original errors clean ---
                    try:
                        playerbet_dec = _money(raw)
                    except Exception:
                        # allow 'all-in' keywords via placebet() (it returns a to-price number already)
                        if isinstance(raw, str) and raw.strip().lower() in ("all-in", "all in", "allin"):
                            # treat as all-in TO-PRICE
                            playerbet_dec = allowed_to
                        else:
                            await self.send_to_user(player.player_id, '❌ Invalid bet. Please enter another bet.')
                            continue

                    # recompute caps after parsing
                    current_to_dec = _money(player.currentbet)
                    balance_dec    = _money(player.balance)
                    allowed_to     = current_to_dec + balance_dec
                    to_call_dec    = target_dec - current_to_dec

                    # ---------- SHORT-STACK GATE ----------
                    # If player cannot cover the call, they must fold or go EXACT all-in (no partials).
                    if to_call_dec > EPS and allowed_to + EPS < target_dec:
                        # Only exact all-in-to-price accepted (or fold handled above)
                        # If they typed something other than EXACT allowed_to, reject & reprompt.
                        if abs(playerbet_dec - allowed_to) > EPS:
                            await self.send_to_user(
                                player.player_id,
                                "❌ You’re covered: enter **-1** to fold, or **all-in** (your full stack amount)."
                            )
                            continue

                    # Clamp intent to legal "to" amount (never exceed own stack)
                    bet_to = playerbet_dec if playerbet_dec <= allowed_to else allowed_to
                    delta  = bet_to - current_to_dec

                    # --- enforce minimum call for covered players ---
                    # If player CAN cover the call, they may not bet below the target (no “partial calls”).
                    if to_call_dec > EPS and allowed_to + EPS >= target_dec:
                        if bet_to + EPS < target_dec:
                            await self.send_to_user(
                                player.player_id,
                                f"❌ You must at least call {float(target_dec):.2f} or fold (-1)."
                            )
                            continue

                    # nothing added and not an exact check -> invalid
                    if delta <= Decimal("0.00") and bet_to != target_dec:
                        await self.send_to_user(player.player_id, '❌ Invalid bet. Please enter another bet.')
                        continue

                    # apply the delta to ledgers (Decimal) and update UI meters
                    if delta > Decimal("0.00"):
                        self._apply_delta(player, delta)
                        player.currentbet = _money(bet_to)
                        await self.pot_info_update()
                        await self.player_info_update_all()
                        await self.bet_info_update()
                        await self.broadcast_balance_update()
                    else:
                        # check / already-in (no chips added)
                        player.currentbet = _money(bet_to)

                    # mark all-in if they hit their cap
                    if abs(bet_to - (current_to_dec + balance_dec)) <= EPS and player not in self.all_in:
                        self.all_in.append(player)
                        self.ledger_event(f"player {player.name} is ALL-IN for {bet_to:.2f}")
                        await self.output(f"{player.name} is ALL-IN for {bet_to:.2f}")

                    # ---- decide outcome: raise / call / check / short-all-in display ----
                    round_to_dec = _money(self.round_to)

                    # RAISE (includes first-action all-in > target)
                    if bet_to > round_to_dec + EPS:
                        self.round_to = float(bet_to)
                        self.ledger_event(f'{player} raises to {self.round_to}')
                        await self.output(f'{player} raises to {self.round_to}')
                        await self.send_to_user(player.player_id, {"action": "turn_end"})

                        # Move action to the next seat; action closes on the raiser
                        raiser_idx  = next(i for i, p2 in enumerate(self.order) if p2 is player)
                        found_index = (raiser_idx + 1) % len(self.order)
                        end_index   = raiser_idx

                        # Restart outer loop to process next player naturally
                        continue_loop = True
                        restart = True
                        break

                    # Not a raise:
                    if round_to_dec <= EPS:
                        # price == 0 → check
                        self.ledger_event(f'{player} checks')
                        await self.output(f'{player} checks')
                        await self.send_to_user(player.player_id, {"action": "turn_end"})
                        break

                    # Positive price: exactly matched the call
                    if abs(_money(player.currentbet) - round_to_dec) <= EPS:
                        if delta <= EPS:
                            if balance_dec > EPS:
                                self.ledger_event(f'{player} is already in for {self.round_to}')
                                await self.output(f'{player} is already in for {self.round_to}')
                            else:
                                self.ledger_event(f'{player} is all-in and has no action')
                                await self.output(f'{player} is all-in and has no action')
                        else:
                            self.ledger_event(f'{player} calls to {self.round_to}')
                            await self.output(f'{player} calls to {self.round_to}')
                        await self.send_to_user(player.player_id, {"action": "turn_end"})
                        break

                    # short all-in message (cannot cover full call) — reach here only if we allowed exact all-in
                    if bet_to + EPS < round_to_dec:
                        self.ledger_event(f'{player} is all-in short for {bet_to:.2f} (cannot cover full call)')
                        await self.output(f'{player} is all-in short for {bet_to:.2f} (cannot cover full call)')
                        await self.send_to_user(player.player_id, {"action": "turn_end"})
                        break

                    # fallback
                    await self.output(f'{player} action registered')
                    await self.send_to_user(player.player_id, {"action": "turn_end"})
                    break  # end inner while; move to next player

                if restart:
                    break  # out of for seats; outer while restarts

            # If nobody raised, keep iterating to next player (don’t end after first action)
            if not restart:
                self.ledger_event('the betting is now done for this street')
                # We finished a full pass with no raise → street done
                return "street_done"


    async def evaluate(self):
        print('EVALUATE')

        # 1) If only one player remains -> instant win (fold-out)
        if len(self.order) == 1:
            winner = self.order[0]
            self._add_to_balance(winner, _money(self.pot))
            self.ledger_event(f"everyone else folded... {winner.name} wins {float(self.pot):.2f}")
            await self.output(f"everyone else folded... {winner.name} wins {float(self.pot):.2f}")
            self.pot = Decimal("0.00")
            await self.player_info_update_all()

            # clear board
            await self.send_info_all({"board": {"clear": True}})
            return

        # ---------- Parse hands WITHOUT mutating player.hand ----------
        board = list(self.board)
        results = []  # list[(player, hp)]
        for player in self.order:
            hole = list(player.hand)               # copy
            hp = HandParser(hole + board)          # eval on fresh snapshot
            player.hand_forscore = hp              # optional: keep for UI/debug
            player.handscore = hp.handenum.name
            results.append((player, hp))

        # ---------- Tie handling helpers ----------
        def hands_equal(a, b):
            return not (a > b) and not (b > a)

        def tier_results_with_ties(pairs):
            """
            pairs: list[(player, hp)]
            returns: list[list[(player, hp)]]  # tiers; each inner list is a tie group
            """
            pool = {p: hp for p, hp in pairs}
            tiers = []
            while pool:
                # find best in pool
                best_p, best_hp = next(iter(pool.items()))
                for p, hp in pool.items():
                    if hp > best_hp:
                        best_p, best_hp = p, hp
                # collect ties with best
                tied = [(p, hp) for p, hp in list(pool.items()) if hands_equal(hp, best_hp)]
                tiers.append(tied)
                for p, _ in tied:
                    del pool[p]
            return tiers

        ranked_tiers = tier_results_with_ties(results)  # best tier first
        seat_index = {p: i for i, p in enumerate(self.order)}  # deterministic remainders

        # ---------- Build pots correctly from total contributions ----------
        bets_by_player = dict(self.contributed)  # {player: Decimal total_contributed}
        print('[EVAL] contributed:', {p.name: float(a) for p, a in bets_by_player.items()})

        # money helpers (Decimal-safe)
        getcontext().prec = 28
        CENT = Decimal("0.01")

        def money(x) -> Decimal:
            if isinstance(x, Decimal):
                return x.quantize(CENT, rounding=ROUND_HALF_UP)
            return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)

        def split_even(amount: Decimal, n: int):
            total_cents = int((amount / CENT).to_integral_value(rounding=ROUND_HALF_UP))
            q, r = divmod(total_cents, n)
            return [(Decimal(q + (1 if i < r else 0)) * CENT) for i in range(n)]

        def build_side_pots(bets_by_player, active_players):
            """
            Canonical side-pot construction using contribution caps.
            Returns: list[(pot_amount: Decimal, eligible_set: set(player))]
            """
            contrib = {p: money(a) for p, a in bets_by_player.items() if money(a) > 0}
            if not contrib:
                return []

            levels = sorted(set(contrib.values()))
            actives = set(active_players)

            def capped_sum(cap):
                return sum(min(a, cap) for a in contrib.values())

            pots, prev = [], Decimal("0.00")
            for cap in levels:
                layer_amount = money(capped_sum(cap) - capped_sum(prev))
                if layer_amount > 0:
                    elig = {p for p, a in contrib.items() if a >= cap} & actives
                    if elig:
                        pots.append((layer_amount, elig))
                prev = cap
            return pots

        pots = build_side_pots(bets_by_player, self.order)
        print('[EVAL] pots:', [(float(a), {p.name for p in s}) for a, s in pots])

        # ---------- Award pots (tier-aware, Decimal-safe) ----------
        for pot_amount, elig in pots:
            pot_amount = money(pot_amount)
            if pot_amount <= 0 or not elig:
                continue

            # Find the best eligible tier (highest hand among elig set)
            winners = None
            for tier in ranked_tiers:  # best → worse
                tier_players = [p for (p, _hp) in tier if p in elig]
                if tier_players:
                    winners = tier_players
                    break
            if not winners:
                # No eligible winners (shouldn't happen if elig non-empty)
                continue

            winners_sorted = sorted(winners, key=lambda p: seat_index.get(p, 0))

            if len(winners_sorted) == 1:
                # ✅ Single winner takes THIS pot_amount
                w = winners_sorted[0]
                self._add_to_balance(w, pot_amount)
                self.ledger_event(f"{w.name} wins {float(pot_amount):.2f}")
                await self.output(f"{w.name} wins {float(pot_amount):.2f}")
                continue

            # Tie: split into exact cents
            shares = split_even(pot_amount, len(winners_sorted))
            for share, p in zip(shares, winners_sorted):
                self._add_to_balance(p, share)

            names = " & ".join(p.name for p in winners_sorted)
            if len(set(shares)) == 1:
                self.ledger_event(f"{names} split {float(pot_amount):.2f} ({float(shares[0]):.2f} each)")
                await self.output(f"{names} split {float(pot_amount):.2f} ({float(shares[0]):.2f} each)")
            else:
                self.ledger_event(f"{names} split {float(pot_amount):.2f} "
                    f"({', '.join(f'{float(s):.2f}' for s in shares)})")
                await self.output(
                    f"{names} split {float(pot_amount):.2f} "
                    f"({', '.join(f'{float(s):.2f}' for s in shares)})"
                )

        # ---------- Cleanup & broadcast AFTER paying ----------
        self.pot = Decimal("0.00")
        self.all_in.clear()
        self.round_to = 0
        self.bet.clear()
        self.contributed.clear()  # start next hand fresh
        for pl in self.perma_list:
            pl.currentbet = 0.0

        await self.pot_info_update()
        await self.bet_info_update(new_bets=True)
        await self.player_info_update_all()
        await self.broadcast_balance_update()

        #clear board
        await self.send_info_all({"board": {"clear": True}})

        print("[ALL-IN STATUS]", [p.name for p in self.all_in])
        print("[BETS]", [(p.name, b) for p, b in self.bet])
        print("[POTS BUILT]", [(float(a), {p.name for p in s}) for a, s in pots])

    async def fold_check(self):
        if len(self.order)==1:
            # self.order[0].balance+=self.pot
            print(f'everyone else folded... {self.order[0].name} wins {self.pot}')
            await self.output(f"everyone else folded... {self.order[0].name} wins {self.pot}")
            # asyncio.create_task(self.game_summary())
            self.gameover=True
  
    def potcalc(self):
        #takes the latest bets from each player unless the player bet then folded
        latest_bets = {}
        print(self.bet)

        #go through the list of bets in reverse
        for player, bet in reversed(self.bet):
            if player not in latest_bets and bet != -1:
                latest_bets[player] = bet
            elif player not in latest_bets:
                latest_bets[player] = 0
            elif bet != -1:
                continue  # Skip if player already has a non-negative bet
            elif latest_bets[player] != -1:
                latest_bets[player] = 0  # Treat -1 bet as 0 if there's no previous non-negative bet
            #subtract the latest bet for each player from their balance
            print('subtracting bet from balance')
            player.balance-=latest_bets[player]
            print(player,player.balance)
        
        #if all the latest bets are greater than or equal to big blind then the two blinds have bet or checked
        if all(value >= self.bigblind for value in latest_bets.values()) and self.preflop==True:
            #remove small blind and big blind bets
            self.bet.pop(0)
            self.bet.pop(1)
            print('preflop bets list after removing blinds:',self.bet)

        print(latest_bets)
        sum_pot=sum(value for value in latest_bets.values())
        print('potcalc sum_pot: ',sum_pot)
        return sum_pot
    
    def _add_to_balance(self, player, amount_dec: Decimal):
        player.balance = _to_float(_money(player.balance) + _money(amount_dec))
        self.all_in = [p for p in self.perma_list if _money(p.balance) <= EPS]
        for p in self.perma_list:
            p.currentbet = 0.0

    async def pot_info_update(self):
        data={
            "pot": {
                "pot": float(self.pot)  # ← cast to float
            }
        }
        self.ledger_event(data)
        await self.send_info_all(data)
    
    async def bet_info_update(self, blinds_start=False, new_bets=False):
        if blinds_start:
            data={"bet": {"amount": float(self.bigblind)}}
            ledger_msg={"bet": {"amount": float(self.bigblind)},'type':'blind'}
            self.ledger_event(data)
            await self.send_info_all(data)
            return
        if new_bets:
            data={"bet": {"amount": 0.0}}
            self.ledger_event(data)
            await self.send_info_all(data)
            return
        if self.bet:
            last = float(self.bet[-1][-1])  # ✅ ensure JSON-serializable
            data={"bet": {"amount": last}}
            self.ledger_event(data)
            await self.send_info_all(data)

    
    async def player_info_update(self):
        for player in self.order:
            print(f"Sending stats to {player.player_id}:", player.balance)

            hand_pairs = []
            if len(player.hand) >= 2:
                (rank1, suit1), (rank2, suit2) = player.hand[:2]
                hand_pairs = [(rank1.name, suit1.name), (rank2.name, suit2.name)]

            data={
                "player": {
                    "user_id": player.player_id,
                    "name": player.name,
                    "balance": float(player.balance),
                    "currentbet": float(player.currentbet),
                    "handscore": getattr(player, "handscore", None),
                    "hand": hand_pairs,
                }
            }
            self.private_ledger_event(player,data)
            await self.send_player_info(player.player_id,data)



    async def player_info_update_all(self):
        for player in self.perma_list:
            print(f"Sending stats to {player.player_id}:", player.balance)

            hand_pairs = []
            if len(player.hand) >= 2:
                (rank1, suit1), (rank2, suit2) = player.hand[:2]
                hand_pairs = [(rank1.name, suit1.name), (rank2.name, suit2.name)]
            data={
                "player": {
                    "user_id": player.player_id,
                    "name": player.name,
                    "balance": float(player.balance),
                    "currentbet": float(player.currentbet),
                    "handscore": getattr(player, "handscore", None),
                    "hand": hand_pairs,
                }
            }
            self.private_ledger_event(player,data)
            await self.send_player_info(player.player_id, data)


    async def update_handscore(self):
        for player in self.perma_list:
            # 1) Build a fresh snapshot; never pass references you might mutate later
            hole = list(player.hand)      # copy
            board = list(self.board)      # copy
            seven = hole + board          # 2 to 7 cards, depending on street

            # 2) Create a NEW parser on the combined snapshot
            hp = HandParser(seven)
            player.hand_forscore = hp
            player.handscore = hp.handenum.name

            await self.player_info_update_all()


    async def Round(self,bet=0):
        if self.cancel_event and self.cancel_event.is_set():
            raise asyncio.CancelledError
        
        marker = self._make_entry({"marker": "ROUND_START"}, etype="marker")
        self.ledger.append(marker)

        for p in self.perma_list:
            self._add_to_private_ledger(p, marker)

        # --- per-hand reset ---
        self.gameover = False
        self.pot = Decimal("0.00")
        self.bet = []
        self.contributed = defaultdict(Decimal)
        self.board = []
        self.preflop = True
        self.rivercheck = False
        self.all_in = []

        # per-hand resets:
        st = getattr(self, "send_info_all", None)
        # access via consumer.state if you keep a ref there
        if hasattr(self, "send_to_user"):  # we have the consumer bound
            try:
                st = self.input.__self__.state
                for k, f in list(st.get("pending_inputs_all", {}).items()):
                    if not f.done():
                        f.set_result("round starting")
                st.get("pending_inputs_all", {}).clear()
            except Exception:
                pass

        # refresh active players list (keep everyone with chips)
        self.list = [p for p in self.perma_list if not getattr(p, "out_of_balance", False) and p.balance > 0]

        # seed/rotate order BEFORE dealing
        if not getattr(self, "startingorder", []):
            # first hand in a match: pick dealer & freeze starting order
            self.pickdealer()
            self.startingorder = self.order[:]  # snapshot
        else:
            # subsequent hands: rotate against startingorder + round counter
            if getattr(self, "round", None) is None:
                self.round = 1
            n = (self.round - 1) % len(self.startingorder)
            self.order = self.startingorder[n:] + self.startingorder[:n]

        # safety: if order accidentally collapsed last hand, repopulate from startingorder
        if len(self.order) < len(self.list):
            # Re-synchronize order with active players while preserving rotation
            active_ids = {p.player_id for p in self.list}
            self.order = [p for p in self.order if p.player_id in active_ids]
            # append any missing active players in startingorder order
            for p in self.startingorder:
                if p.player_id in active_ids and p not in self.order:
                    self.order.append(p)

        # still not enough players? bail gracefully
        if len(self.order) < 2:
            await self.output("Not enough players to continue.")
            return

        if self.cancel_event and self.cancel_event.is_set():
                raise asyncio.CancelledError
        
        await self.output("Game round begins")
        print(self.list)

        #remove player from game if they are out of balance
        for player in self.list[:]:  # iterate over a copy
            if player.balance == 0:
                player.out_of_balance = True
                self.list.remove(player)

        #reset hands
        for x in self.list:
            x.hand=[]
        if self.round==1:
            #pick a dealer
            self.pickdealer()
            self.startingorder=copy.copy(self.order)
        await self.output(f"{self.order[0]} is the dealer - the order is {self.order}")
        self.ledger_event(f"{self.order[0]} is the dealer - the order is {self.order}")
        #create a new deck
        self.createdeck()
        #shuffle the deck
        self.shuffledeck()
        #deal each player 2 cards 
        self.deal()
        for player in self.order:
            print(f'{player} hand is: ',player.hand)
            rank1, suit1 = player.hand[0]
            rank2, suit2 = player.hand[1]
            msg = f"your hand is {rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}"
            ledger_msg=f'{player}, you have the hand {rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}'
            ledger_msg_hand=f'{player} has the hand {rank1.name} of {suit1.name}, {rank2.name} of {suit2.name}'
            self.private_ledger_event(player,ledger_msg)
            self.ledger_event_gen(ledger_msg_hand)
            #print('LEDGER TEST',ledger_msg,flatten_to_string(self.private_ledger))
            await self.send_to_user(player.player_id, msg)
        

        #sends an update to player info
        await self.player_info_update()

        #update hand score
        await self.update_handscore()
        
        #preflop
        # at the start of the hand (before posting blinds)
        self.contributed.clear()
        self.bet.clear()
        self.pot = 0

        # ... after dealing, before preflop betting ...
        dealer_idx, sb_idx, bb_idx = self.positions()
        sbp = self.order[sb_idx]
        bbp = self.order[bb_idx]

        self.ledger_event(f'the dealer is {dealer_idx}, the small blind player is {sbp}, the big blind player is {bbp}, the value of both blinds is 0.1 each and the blinds have been posted')

        # Post blinds using Decimal-safe helper
        sb_amt = _money(self.smallblind)
        bb_amt = _money(self.bigblind)

        self._apply_delta(sbp, sb_amt)  # logs (player, delta), bumps pot, contributed, balance
        self._apply_delta(bbp, bb_amt)

        # UI-facing per-street meters remain floats
        sbp.currentbet = _money(sb_amt)   # Decimal
        bbp.currentbet = _money(bb_amt)   # Decimal
        self.round_to  = _money(bb_amt)   # Decimal (source of truth!)

        # Remember who posted blinds (bets() reads these safely)
        self.small_blind_player = sbp
        self.big_blind_player   = bbp

        await self.pot_info_update()
        await self.bet_info_update(blinds_start=True)

        # === Players Bar init ===
        n = len(self.order)
        # Preflop, first to act is after the big blind
        first_to_act_idx = (bb_idx + 1) % n
        first_to_act_pid = self.order[first_to_act_idx].player_id

        await self.broadcast_players_state(active_player_id=first_to_act_pid)


        #reset
        if self.cancel_event and self.cancel_event.is_set():
                raise asyncio.CancelledError
        
        self.ledger_event('Preflop betting begins.')

        # Preflop betting begins vs target == BB (bets() will set self.round_to accordingly)
        res = await self.bets(preflop=True)

        if res == "hand_over" or self.gameover:
            await self.pot_info_update()          # optional: refresh UI before paying
            await self.player_info_update_all()   # optional
            await self.evaluate()                 # ✅ pay once here
            await self._signal_hand_over_safely()   
            # ✅ rotate for the NEXT hand now
            self.round = getattr(self, "round", 1) + 1
            n = (self.round - 1) % len(self.startingorder)
            self.order = self.startingorder[n:] + self.startingorder[:n]
            await self.broadcast_players_state()
            return


        #send pot info
        await self.pot_info_update()
        await self.player_info_update_all()


        await self.fold_check()
        
        if self.gameover==False:
            if self.cancel_event and self.cancel_event.is_set():
                raise asyncio.CancelledError

            #flop
            await self.flop()
            self.preflop=False

            if self.cancel_event and self.cancel_event.is_set():
                raise asyncio.CancelledError
            
            self.ledger_event(f'the flop has been dealt, the flop is: {self.board}')

            #self.bet=[]
            res = await self.bets()
            if res == "hand_over" or self.gameover:
                await self.pot_info_update()
                await self.player_info_update_all()
                await self.evaluate()
                await self._signal_hand_over_safely()
                self.round = getattr(self, "round", 1) + 1
                n = (self.round - 1) % len(self.startingorder)
                self.order = self.startingorder[n:] + self.startingorder[:n]
                await self.broadcast_players_state()                
                return

            #update
            await self.player_info_update()
            await self.output(f"the pot is: {self.pot}")
            await self.pot_info_update()
            await self.player_info_update()
            await self.fold_check()

        if self.gameover==False:
            #turn

            if self.cancel_event and self.cancel_event.is_set():
                raise asyncio.CancelledError
            
            await self.turn()
            self.preflop=False

            self.ledger_event(f'the turn has been dealt, the turn is: {self.board}')

            if self.cancel_event and self.cancel_event.is_set():
                raise asyncio.CancelledError
            
            res = await self.bets()
            if res == "hand_over" or self.gameover:
                await self.pot_info_update()
                await self.player_info_update_all()
                await self.evaluate()
                await self._signal_hand_over_safely()
                self.round = getattr(self, "round", 1) + 1
                n = (self.round - 1) % len(self.startingorder)
                self.order = self.startingorder[n:] + self.startingorder[:n]
                await self.broadcast_players_state()
                return

            #update
            await self.player_info_update()
            await self.output(f"the pot is: {self.pot}")
            await self.pot_info_update()
            await self.player_info_update()
            await self.fold_check()

        if self.gameover==False:
            if self.cancel_event and self.cancel_event.is_set():
                raise asyncio.CancelledError
            #river
            await self.river()
            self.preflop=False

            self.ledger_event(f'the river has been dealt, the river is: {self.board}')
            if self.cancel_event and self.cancel_event.is_set():
                raise asyncio.CancelledError

            res = await self.bets()
            if res == "hand_over" or self.gameover:
                await self.pot_info_update()
                await self.player_info_update_all()
                await self.evaluate()
                self.round = getattr(self, "round", 1) + 1
                n = (self.round - 1) % len(self.startingorder)
                self.order = self.startingorder[n:] + self.startingorder[:n]
                await self.broadcast_players_state()
                return

            #update
            await self.player_info_update()
            await self.output(f"the pot is: {self.pot}")
            await self.pot_info_update()
            await self.player_info_update()

            self.rivercheck=True

            if self.cancel_event and self.cancel_event.is_set():
                raise asyncio.CancelledError
            
            await self.fold_check()
            await self.evaluate()
            await self._signal_hand_over_safely()     

        #changed from list to order 
        for x in self.order:
            print(x,'  ','balance:',x.balance)
            await self.output(f"{x} has the balance: {x.balance}")
            print(x, x.hand)
            hole  = list(x.hand)          # [('SEVEN','DIAMOND'), ('QUEEN','SPADE')] via enums
            board = list(self.board)      # 0..5 board cards

            # parser for scoring (7-card eval when board is complete enough)
            hp = HandParser(hole + board)

            # readable strings — choose what you want to show:
            hole_readable  = [f"{r.name} of {s.name}" for r, s in hole]
            board_readable = [f"{r.name} of {s.name}" for r, s in board]
            all_readable   = hole_readable + board_readable

            score_name = hp.handenum.name  # e.g., "TWO_PAIR", "FLUSH", etc.

            # If x is a player object, use x.name (str(x) may be an object repr)
            await self.output(
                f"{x.name} has a {score_name} with the cards {', '.join(all_readable)}"
            )

        for x in self.list:
            #reset hand for next round
            x.hand=[]
        
        #sends an update to player info
        for player in self.list:
            await self.send_player_info(player.player_id, {
                "player": {
                    "name": player.name,
                    "balance": player.balance,
                    "currentbet": player.currentbet,
                    "handscore": player.handscore,
                    "hand": ['']
                }
            })

        self.rivercheck=False

        try:
            await self.pot_info_update()
        except Exception as e:
            pass

        try:
            await self.send_info_all({"board": {"board": "new round"}})
        except Exception as e:
            pass
  
        #change order for next round
        self.round+=1
        print('THE ROUND IS: ',self.round)
        self.gameover=False

        #rotate button clockwise and wrap
        n = (self.round - 1) % len(self.startingorder)
        self.order = self.startingorder[n:] + self.startingorder[:n]






        
class Player():
    def __init__(self,player_id,balance,table):
        self.player_id=player_id
        self.balance=balance
        self.out_of_balance=False
        self.hand=[]
        self.currentbet=0
        self.handscore=None
        self.hand_forscore=None
        self.table=table
        self.name=f'{self.player_id[-5:]}'
        self._bal_lock = asyncio.Lock()
        self.avatar_url = avatar_from_id(player_id)

    def __repr__(self):
        return (f'{self.name}')
    
    # MANAGE ALL-IN
    async def placebet(self, current_price, valid=True, cancel_event=None):
        # print(f"[placebet] prompt for {self.player_id[:5]} target={current_price} bal={self.balance} curbet={self.currentbet}")
        while True:
            try:
                await asyncio.wait_for(self.table.send_hint(self), timeout=10)
            except asyncio.TimeoutError:
                await self.table.send_to_user(self.player_id, "(hint delayed)")
            except Exception as e:
                pass
            raw = await self.table.input(
                self.player_id,
                f'{self.name}, price is {current_price}, place your bet (0 for check, -1 for fold): ',
                cancel_event=cancel_event
            )

            # --- interpret ALL-IN keyword(s) ---
            if isinstance(raw, str) and raw.strip().lower() in ("all-in", "all in", "allin"):
                # RETURN A TO-PRICE, NOT THE REMAINING AMOUNT
                to_price = float(self.currentbet) + float(self.balance)
                if self not in self.table.all_in:
                    self.table.all_in.append(self)
                await self.table.output(f"{self.name} goes ALL-IN for {self.balance}")
                return to_price  # <-- was: return float(self.balance)

            # --- otherwise parse numeric ---
            try:
                bet = float(raw)
            except (TypeError, ValueError):
                await self.table.send_to_user(self.player_id, "❌ Invalid input. Enter a number, -1 to fold, or 'all-in'.")
                continue

            if bet < -1:
                await self.table.send_to_user(self.player_id, "❌ Invalid bet.")
                continue

            # If they typed exactly their remaining amount, treat that as ALL-IN TO-PRICE as well.
            # (Prevents ‘short’ all-in when they already have chips in this street.)
            if abs(bet - float(self.balance)) < 1e-9:
                to_price = float(self.currentbet) + float(self.balance)
                if self not in self.table.all_in:
                    self.table.all_in.append(self)
                await self.table.output(f"{self.name} goes ALL-IN for {self.balance}")
                return to_price  # <-- was: return bet

            if bet > float(self.balance):
                await self.table.send_to_user(self.player_id, "❌ Bet exceeds your balance.")
                continue

            return bet

    async def add_balance(self, amount):
        try:
            inc = _money(amount)
            if inc <= 0:
                return
        except Exception:
            return

        # Lazy-init lock in case of older objects
        if not hasattr(self, "_bal_lock"):
            import asyncio
            self._bal_lock = asyncio.Lock()

        async with self._bal_lock:
            self.balance = _to_float(_money(self.balance) + inc)
            if getattr(self, "out_of_balance", False) and self.balance > 0:
                self.out_of_balance = False

        await self.table.player_info_update_all()

import asyncio

async def run_game(player_ids, consumer, smallblind=.10, bigblind=.10, room_name=None, cancel_event=None):
    table = Table(
        smallblind, bigblind,
        output=consumer.broadcast_system,
        input=consumer.get_input,
        send_to_user=consumer.send_to_user,
        send_player_info=consumer.send_player_info,
        send_info_all=consumer.send_info_all,
    )
    table.cancel_event = cancel_event

    for pid in player_ids:
        player = Player(player_id=pid, balance=20, table=table)
        await table.addplayer(player)

    consumer.table = table
    if isinstance(getattr(consumer, "state", None), dict):
        consumer.state["table"] = table

    def _clear_all_futures():
        state = getattr(consumer, "state", {})
        # per-user
        for fut in list(state.get("pending_inputs", {}).values()):
            if not fut.done():
                fut.set_result("hand over")
        state.get("pending_inputs", {}).clear()
        # group: clear ALL keys
        for k, f in list(state.get("pending_inputs_all", {}).items()):
            if not f.done():
                f.set_result("hand over")
        state.get("pending_inputs_all", {}).clear()

    try:
        await consumer.broadcast_system(f"🎮 Room ready in {room_name or 'room'} — players: {len(player_ids)}")

        while True:
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError

            _clear_all_futures()

            # 🟢 Tell clients the next round can be started (enable button)
            await consumer.send_info_all({"action": "start_round_prompt"})

            rsp = await consumer.get_input_all(
                '✅ Game is ready. Type or press **Start New Round** to begin',
                cancel_event=cancel_event,
            )

            if str(rsp).strip().lower() != "start new round":
                await consumer.broadcast_system('[INVALID ENTRY] Please type **start new round**')
                await asyncio.sleep(0)
                continue

            # 🔒 Round is starting: disable the button on clients
            await consumer.send_info_all({"action": "round_started"})

            await table.Round()
            _clear_all_futures()

            # 🗣️ Show commentator summary before prompting next round
            try:
                await asyncio.wait_for(table.game_summary(), timeout=15)
                table.ledger=[]
            except asyncio.TimeoutError:
                await consumer.broadcast_system("[SYSTEM]: Summary delayed.")
            except Exception as e:
                print(f"[SUMMARY ERROR] {e}")

            # 🟢 Now show the next-round prompt
            # await consumer.broadcast_system("🟢 Hand finished. Type **start new round** or press the button.")
            await consumer.send_info_all({"action": "start_round_prompt"})
            await asyncio.sleep(0)


    except asyncio.CancelledError:
        return {"status": "cancelled", "players": player_ids}
    except Exception:
        tb = traceback.format_exc()
        print("[GAME CRASH]\n", tb)
        try:
            await consumer.broadcast_system(f"💥 Game crashed:\n{tb}")
        except Exception:
            pass
        return {"status": "error", "players": player_ids}


import random
import string
from decimal import Decimal

async def run_game_cpu(cpu_count, player_ids, consumer, smallblind=.10, bigblind=.10, room_name=None, cancel_event=None):
    # 0) Early cancel if restart hit before we even begin
    if cancel_event and cancel_event.is_set():
        return {"status": "cancelled", "players": player_ids}

    # 1) Build table and players
    table = Table(
        Decimal(str(smallblind)),
        Decimal(str(bigblind)),
        output=consumer.broadcast_system,
        input=consumer.get_input,          # supports cancel_event
        send_to_user=consumer.send_to_user,
        send_player_info=consumer.send_player_info,
        send_info_all=consumer.send_info_all,
    )
    table.cancel_event = cancel_event

    for pid in player_ids:
        if cancel_event and cancel_event.is_set():
            return {"status": "cancelled", "players": player_ids}
        player = Player(player_id=pid, balance=20, table=table)
        await table.addplayer(player)
    
    for _ in range(cpu_count):
        if cancel_event and cancel_event.is_set():
            return {"status": "cancelled", "players": player_ids}
        cpu_id = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))
        new_cpu = LLMPokerBot(cpu_id, balance=20, table=table)
        await table.addplayer(new_cpu)

    # 2) Bind table back to consumer + shared state
    consumer.table = table
    if isinstance(getattr(consumer, "state", None), dict):
        consumer.state["table"] = table

    # 3) Announce readiness
    try:

        def _clear_all_futures():
            state = getattr(consumer, "state", {})
            # per-user
            for fut in list(state.get("pending_inputs", {}).values()):
                if not fut.done():
                    fut.set_result("hand over")
            state.get("pending_inputs", {}).clear()
            # group: clear ALL keys
            for k, f in list(state.get("pending_inputs_all", {}).items()):
                if not f.done():
                    f.set_result("hand over")
            state.get("pending_inputs_all", {}).clear()

        while True:
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError

            _clear_all_futures()

            # 🟢 Tell clients the next round can be started (enable button)
            await consumer.send_info_all({"action": "start_round_prompt"})

            rsp = await consumer.get_input_all(
                '✅ Game is ready. Type or press **Start New Round** to begin',
                cancel_event=cancel_event,
            )

            if str(rsp).strip().lower() != "start new round":
                await consumer.broadcast_system('[INVALID ENTRY] Please type **start new round**')
                await asyncio.sleep(0)
                continue

            # 🔒 Round is starting: disable the button on clients
            await consumer.send_info_all({"action": "round_started"})

            await table.Round()
            _clear_all_futures()

            # 🗣️ Show commentator summary before prompting next round
            try:
                await asyncio.wait_for(table.game_summary(), timeout=15)
                table.ledger=[]
            except asyncio.TimeoutError:
                await consumer.broadcast_system("[SYSTEM]: Summary delayed.")
            except Exception as e:
                print(f"[SUMMARY ERROR] {e}")

            # 🟢 Now show the next-round prompt
            await consumer.send_info_all({"action": "start_round_prompt"})
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        return {"status": "cancelled", "players": player_ids}
    except Exception:
        tb = traceback.format_exc()
        print("[GAME CRASH]\n", tb)
        try:
            await consumer.broadcast_system(f"💥 Game crashed:\n{tb}")
        except Exception:
            pass
        return {"status": "error", "players": player_ids}




         