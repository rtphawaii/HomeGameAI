# equity_mc.py
import random
from functools import lru_cache
from typing import Iterable, List, Tuple, Callable
from pokerlib.enums import Rank, Suit
from pokerlib import HandParser as HandParser  # <-- your class expects (Rank, Suit) tuples

Card = Tuple[Rank, Suit]

def full_deck() -> List[Card]:
    return [(r, s) for r in Rank for s in Suit]

# ----- Cached 7-card evaluator using your HandParser -----

def _cards_key(cards7: List[Card]) -> Tuple[Tuple[int, int], ...]:
    """
    Order-insensitive, hashable key for 7-card sets.
    Convert Rank/Suit enums to ints so sorting/keys are stable.
    """
    return tuple(sorted(((int(r), int(s)) for (r, s) in cards7)))

@lru_cache(maxsize=200_000)
def _rank7_cached(key: Tuple[Tuple[int, int], ...]) -> HandParser:
    """
    Cached evaluation: build HandParser once for the 7-card set.
    Returns a HandParser object; comparisons (__gt__/__eq__) are defined.
    """
    # Rebuild (Rank, Suit) tuples from ints:
    cards7 = [(Rank(r), Suit(s)) for (r, s) in key]
    return HandParser(cards7)

def rank7_with_handparser(cards7: List[Card]) -> HandParser:
    """Public wrapper used by the MC loop."""
    return _rank7_cached(_cards_key(cards7))

# ----- Monte Carlo equity -----

def estimate_equity_mc(
    hero_hand: Iterable[Card],
    board: Iterable[Card],
    num_opponents: int = 1,
    trials: int = 1000,
    rank7_fn: Callable[[List[Card]], HandParser] = rank7_with_handparser,
) -> float:
    """
    Monte Carlo equity vs `num_opponents` random hands on the given `board`.
    Uses your HandParser objects directly for comparisons (higher = stronger).
    """
    print('estimate_equity_mc')
    hero = list(hero_hand)
    brd = list(board)

    # Build remaining deck (exclude dead cards)
    dead = set(hero + brd)
    deck = [c for c in full_deck() if c not in dead]

    need_board = 5 - len(brd)
    if len(hero) != 2 or need_board < 0:
        return 0.0

    draw_needed = 2 * num_opponents + need_board
    if draw_needed > len(deck):
        # Not enough cards to draw (shouldn't happen in normal play)
        return 0.0

    wins = ties = total = 0.0

    for _ in range(max(1, trials)):
        draw = random.sample(deck, draw_needed)

        opps_raw = draw[: 2 * num_opponents]
        bfill = draw[2 * num_opponents :]

        opps = [opps_raw[i:i+2] for i in range(0, len(opps_raw), 2)]
        seven_board = brd + bfill

        hero_rank = rank7_fn(hero + seven_board)
        opp_ranks = [rank7_fn(h + seven_board) for h in opps]

        best_opp = max(opp_ranks)                      # uses HandParser.__gt__
        n_best = sum(1 for r in opp_ranks if r == best_opp)  # HandParser.__eq__

        total += 1.0
        if hero_rank > best_opp:
            wins += 1.0
        elif hero_rank == best_opp:
            ties += 1.0 / (n_best + 1)

    return (wins + ties) / total if total else 0.0
