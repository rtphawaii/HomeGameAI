# --- realistic GTO-inspired preflop folds by position (6-max, first-in) ---
from pokerlib.enums import Rank, Suit
import itertools

# Helpers
RANKS = [
    Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
    Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK,
    Rank.QUEEN, Rank.KING, Rank.ACE
]
SUITS = [Suit.CLUB, Suit.DIAMOND, Suit.HEART, Suit.SPADE]
rank_index = {r: i for i, r in enumerate(RANKS)}

def normalize_combo(c):
    (r1, s1), (r2, s2) = c
    from pokerlib.enums import Rank  # for .value
    rank_index = {r: i for i, r in enumerate([
        Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE,
        Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE
    ])}
    return tuple(sorted(((r1, s1), (r2, s2)), key=lambda x: (rank_index[x[0]], x[1].value)))


def is_pair(r1, r2): return r1 == r2
def is_suited(c1, c2): return c1[1] == c2[1]

def canonical_label(c1, c2):
    """Return a hand-class label like 'AKs', 'QJo', '76s', '22'."""
    (r1, s1), (r2, s2) = c1, c2
    # order by rank high→low for label
    if rank_index[r2] > rank_index[r1]:
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return f"{rank_str(r1)}{rank_str(r2)}"
    suffix = "s" if s1 == s2 else "o"
    return f"{rank_str(r1)}{rank_str(r2)}{suffix}"

def rank_str(r):
    return {
        Rank.ACE: "A", Rank.KING: "K", Rank.QUEEN: "Q", Rank.JACK: "J",
        Rank.TEN: "T", Rank.NINE: "9", Rank.EIGHT: "8", Rank.SEVEN: "7",
        Rank.SIX: "6", Rank.FIVE: "5", Rank.FOUR: "4", Rank.THREE: "3",
        Rank.TWO: "2"
    }[r]

def expand_pair(token):
    """'22+' or '77' → set of pair labels like {'22','33',...}"""
    out = set()
    if token.endswith("+") and len(token) == 3 and token[0] == token[1]:
        start = token[0]
        start_idx = "23456789TJQKA".index(start)
        for ch in "23456789TJQKA"[start_idx:]:
            out.add(ch + ch)
    elif len(token) == 2 and token[0] == token[1]:
        out.add(token)
    return out

def expand_suited_off(token):
    """
    Expand e.g. 'A2s+' → {'A2s','A3s',...,'ATs','AJs','AQs','AKs'}
         or 'K9s+' → {'K9s','KTs','KJs','KQs','AKs' not included}
         or exact like 'QJo' → {'QJo'}
    """
    out = set()
    ranks = "23456789TJQKA"
    def idx(ch): return ranks.index(ch)

    if len(token) in (3,4):
        hi = token[0]
        lo = token[1]
        typ = token[2]  # 's' or 'o'
        plus = token.endswith("+")
        if is_pair_text(token): return out  # ignore; handled elsewhere

        if plus:
            # e.g. A2s+
            # advance low rank upwards but below hi
            for ch in ranks[:idx(hi)]:
                # only include lows >= given lo
                if idx(ch) >= idx(lo):
                    out.add(f"{hi}{ch}{typ}")
        else:
            out.add(token)
    return out

def is_pair_text(t): return len(t) >= 2 and t[0] == t[1]

def expand_range_tokens(tokens):
    """Turn tokens like ['22+','A2s+','KQo','JTo','T9s','98s'] into hand-class set."""
    classes = set()
    for t in tokens:
        t = t.strip().upper()
        if not t: continue
        if is_pair_text(t):
            classes |= expand_pair(t)
        else:
            # suited/off tokens (with or without '+')
            classes |= expand_suited_off(t)
    return classes

def combos_for_class(label):
    """Return the concrete 2-card combos for a hand-class label."""
    ranks_map = {v:k for k,v in {
        Rank.ACE:"A", Rank.KING:"K", Rank.QUEEN:"Q", Rank.JACK:"J",
        Rank.TEN:"T", Rank.NINE:"9", Rank.EIGHT:"8", Rank.SEVEN:"7",
        Rank.SIX:"6", Rank.FIVE:"5", Rank.FOUR:"4", Rank.THREE:"3",
        Rank.TWO:"2"
    }.items()}
    def r(ch): return ranks_map[ch]
    ranks = "23456789TJQKA"

    if len(label) == 2 and label[0] == label[1]:
        # pair: 6 combos
        rr = r(label[0])
        cards = [(rr, s) for s in SUITS]
        out = []
        for c1, c2 in itertools.combinations(cards, 2):
            out.append((c1, c2))
        return out

    hi, lo, typ = label[0], label[1], label[2]
    r_hi, r_lo = r(hi), r(lo)

    if typ == 's':
        # suited: 4 combos (one per suit)
        return [((r_hi, s), (r_lo, s)) for s in SUITS]
    else:
        # offsuit: 12 combos (all suit pairs except same suit)
        out = []
        for s1 in SUITS:
            for s2 in SUITS:
                if s1 != s2:
                    out.append(((r_hi, s1), (r_lo, s2)))
        return out

# Build the set of all 1326 combos (unordered)
ALL_COMBOS = []
for i, c1 in enumerate([(r, s) for r in RANKS for s in SUITS]):
    for c2 in [(r, s) for r in RANKS for s in SUITS][i+1:]:
        ALL_COMBOS.append((c1, c2))

# --- Solver-inspired first-in OPEN ranges (approximate) ---
# Tight UTG (~18-20%), HJ (~23-25%), CO (~28-30%), BTN (~45-50%), SB (~40-45%)
UTG_OPEN_TOKENS = [
    "22+",          # all pairs
    "A2s+",         # all suited aces
    "KTs+", "QTs+", "JTs",
    "T9s", "98s", "87s",
    "AJo+", "KQo"
]
HJ_OPEN_TOKENS = UTG_OPEN_TOKENS + [
    "K9s+", "Q9s+", "J9s+", "T8s+", "97s+", "76s", "65s",
    "ATo+", "KJo+", "QJo"
]
CO_OPEN_TOKENS = [
    "22+",
    "A2s+", "K7s+", "Q8s+", "J8s+", "T8s+", "97s+", "86s+", "75s+", "65s", "54s",
    "A9o+", "KTo+", "QTo+", "JTo"
]
BTN_OPEN_TOKENS = [
    "22+",
    "A2s+", "K2s+", "Q5s+", "J6s+", "T6s+", "96s+", "86s+", "75s+", "65s", "54s",
    "A2o+", "K8o+", "Q9o+", "J9o+", "T9o"
]
SB_OPEN_TOKENS = [
    "22+",
    "A2s+", "K5s+", "Q7s+", "J7s+", "T7s+", "97s+", "86s+", "75s+", "65s", "54s",
    "A2o+", "K9o+", "QTo+", "JTo", "T9o"
]

# Heads-up BTN/SB opens wider — roughly 80-85% of all combos
BTN_SB_OPEN_TOKENS = [
    "22+",
    "A2s+", "K2s+", "Q2s+", "J2s+", "T2s+", "93s+", "84s+", "74s+", "64s+", "53s+", "43s",
    "A2o+", "K2o+", "Q5o+", "J6o+", "T6o+", "96o+", "86o+", "76o+", "65o", "54o"
]

def expand_open(tokens):
    classes = expand_range_tokens(tokens)
    combos = set()
    for cl in classes:
        for c in combos_for_class(cl):
            combos.add(normalize_combo(c))  # ✅ store normalized
    return combos

OPEN_BY_POS = {
    "UTG": expand_open(UTG_OPEN_TOKENS),
    "HJ":  expand_open(HJ_OPEN_TOKENS),
    "CO":  expand_open(CO_OPEN_TOKENS),
    "BTN": expand_open(BTN_OPEN_TOKENS),
    "SB":  expand_open(SB_OPEN_TOKENS),
    "BTN/SB": expand_open(BTN_SB_OPEN_TOKENS),  # ✅ New heads-up position
    "BB":  set(),   # unopened to BB → check, not fold
}

# Compute fold lists (all combos minus opens)  ➜ return normalized combos
def fold_list_for(pos):
    opens = OPEN_BY_POS[pos]  # set of normalized combos
    folds = []
    for c in ALL_COMBOS:
        norm = normalize_combo(c)
        if norm not in opens:
            folds.append(norm)  # ✅ append normalized, not original
    return folds

# Optional: fast set lookup + public helper
FOLD_SET_BY_POS = {pos: set(fold_list_for(pos)) for pos in OPEN_BY_POS.keys()}

def is_fold_combo(pos, hand):
    """Return True if the specific 2-card combo should be folded first-in at pos."""
    return normalize_combo(hand) in FOLD_SET_BY_POS[pos]

FOLD_RANGES = {pos: fold_list_for(pos) for pos in ["UTG","HJ","CO","BTN","SB","BB"]}

# --- Optional: quick sanity checks / previews ---
# Example: show 10 sample fold combos from UTG in your exact tuple format
preview_utg = FOLD_RANGES["UTG"][:10]
# print(preview_utg)
