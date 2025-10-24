# utils_poker.py
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

def D(x):
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def amount_to_call(current_price, my_total) -> Decimal:
    t, m = D(current_price), D(my_total)
    return max(Decimal("0"), t - m)

def pot_odds_e(to_call, pot) -> Decimal:
    to_call, pot = D(to_call), D(pot)
    return to_call / (pot + to_call) if to_call > 0 else D("0")

def break_even_fe(bet_size, pot_before_bet) -> Decimal:
    b, p = D(bet_size), D(pot_before_bet)
    return b / (p + b) if b > 0 else D("0")

def bet_for_target_fe(target_fe, pot_before_bet) -> Decimal:
    fe, p = D(target_fe), D(pot_before_bet)
    if fe <= 0 or fe >= 1:
        return D("0")
    return (fe * p) / (Decimal("1") - fe)

def min_raise_total(current_price, last_raise_delta, my_total, stack_total_cap) -> Decimal:
    """
    Minimum legal TOTAL to-price for a raise:
    - current_price: table's current total-to-price to call
    - last_raise_delta: previous raise increment on this street
    - my_total: my current total committed on this street
    - stack_total_cap: my_total + balance (max total I can reach = all-in)
    """
    cp = D(current_price)
    delta = D(last_raise_delta)
    target = cp + max(delta, D("0"))   # call + last delta
    target = max(target, D(my_total))  # never go backwards
    cap = D(stack_total_cap)
    return min(target, cap)

def quantize_to_chips(x, denom="0.01"):
    return D(x).quantize(Decimal(denom), rounding=ROUND_HALF_UP)
