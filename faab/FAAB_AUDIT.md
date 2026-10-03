# FAAB Bid Audit — Fantasy 101 (2020–2026, league-specific)

Dataset: 2404 bids across 1453 auctions, 2020–2026. Losing bids included (status=failed).
Processing weekday (Denver): Wednesday accounts for 83% of completed claims — the league's waiver day, as expected.

## Bid-digit fingerprints

League-wide: only 7.7% of all bids end in 4 or 9 ('one less than a multiple of 5') — BELOW the 20% you'd get from uniform digits, because half the league's bids are $0/$1 claims (digit 0/1 trivially). The league's real house style is the round number: 49.3% of bids end in 0/5. The 4/9 habit is a per-manager fingerprint, not a league trait — see below.

| Manager | Bids | % ending 4/9 | p vs league rate | % ending 0/5 | % $0/$1 | Median bid |
|---|---:|---:|---:|---:|---:|---:|
| 11kaplandh | 161 | 16% | 0.000 | 32% | 37% | $3 |
| BillFroemming | 304 | 11% | 0.066 | 49% | 55% | $1 |
| Jgersowsky | 213 | 3% | 0.014 | 54% | 54% | $1 |
| OGJonnyB | 162 | 11% | 0.103 | 31% | 29% | $4 |
| bgabrielsen | 216 | 4% | 0.054 | 50% | 32% | $3 |
| blinton2 | 364 | 5% | 0.093 | 71% | 74% | $0 |
| gizzle4 | 441 | 4% | 0.003 | 55% | 69% | $0 |
| jcarney3344 | 209 | 6% | 0.362 | 51% | 47% | $2 |
| mlguagliardo | 199 | 18% | 0.000 | 16% | 12% | $4 |
| nsaed | 135 | 5% | 0.333 | 55% | 32% | $5 |

Strongest 4/9 habit (all bids): **mlguagliardo** ends 18% of bids in 4/9 (league 8%, p=0.0000) — a real fingerprint, not noise. Conditioning on real-money bids ($5+) sharpens it: 11kaplandh ends 23% of $5+ bids in 4/9 (p=0.001 vs the $5+ league rate of 9.5%), while bgabrielsen (1%, p=0.004) and nsaed (1%, p=0.02) significantly AVOID 4/9 endings — they bid round or off numbers. Eric (gizzle4): 4% of all bids end 4/9 (11% of $5+ bids, p=0.66 — no detectable digit habit; his tell is volume: 69% of his bids are $0/$1 shots, the league's most trigger-happy profile).

## Position tastes (median bid / share of manager's bids)

| Manager | QB med / share | RB med / share | WR med / share | TE med / share | DEF med / share |
|---|---:|---:|---:|---:|---:|
| 11kaplandh | $9 / 8% | $3 / 42% | $4 / 22% | $4 / 6% | $2 / 23% |
| BillFroemming | $0 / 8% | $2 / 28% | $1 / 37% | $0 / 10% | $1 / 17% |
| Jgersowsky | $2 / 18% | $4 / 20% | $1 / 38% | $1 / 9% | $1 / 15% |
| OGJonnyB | $5 / 9% | $4 / 30% | $3 / 35% | $4 / 11% | $4 / 15% |
| bgabrielsen | $5 / 10% | $5 / 24% | $5 / 13% | $3 / 13% | $2 / 41% |
| blinton2 | $0 / 16% | $0 / 20% | $0 / 24% | $0 / 9% | $0 / 32% |
| gizzle4 | $0 / 10% | $1 / 25% | $1 / 24% | $0 / 14% | $0 / 27% |
| jcarney3344 | $5 / 10% | $2 / 28% | $3 / 28% | $2 / 14% | $0 / 20% |
| mlguagliardo | $6 / 14% | $6 / 20% | $4 / 23% | $5 / 12% | $2 / 32% |
| nsaed | $8 / 10% | $12 / 32% | $4 / 25% | $1 / 10% | $1 / 23% |

League bid-share baseline: WR 27%, RB 26%, DEF 25%, QB 11%, TE 11%

Median bid (all) / median winning bid by position: DEF $1/$2, QB $2/$2, RB $3/$4, TE $2/$2, WR $2/$3

## Winner's-curse margins (winning bid minus second-highest bid)

Median margin $1; mean $3.4. 10% of wins overpaid by $10+; 2% by $20+. 55% of wins were uncontested ($-bid races with one bidder).

| Manager | Wins | Median margin | Mean margin | % wins by $10+ |
|---|---:|---:|---:|---:|
| 11kaplandh | 87 | $3 | $4.1 | 13% |
| BillFroemming | 149 | $0 | $2.4 | 7% |
| Jgersowsky | 78 | $1 | $3.7 | 12% |
| OGJonnyB | 91 | $2 | $4.0 | 13% |
| bgabrielsen | 95 | $3 | $4.3 | 13% |
| blinton2 | 119 | $0 | $3.2 | 12% |
| gizzle4 | 166 | $0 | $2.1 | 7% |
| jcarney3344 | 112 | $1 | $3.3 | 9% |
| mlguagliardo | 112 | $3 | $4.0 | 10% |
| nsaed | 61 | $3 | $5.3 | 20% |

## Aggression across the season

             median_bid  median_win   bidders
phase                                        
early (1-5)         2.0         6.0  2.768281
mid (6-10)          1.0         4.0  2.217391
late (11+)          0.0         2.0  1.934974

Median winning bid by phase (same table): early-season claims command the premium; late FAAB is cheap unless a playoff-push RB emerges.

Mean/max FAAB spent per season (of $100):
                    mean  max
manager                      
gizzle4        95.714286  132
11kaplandh     87.571429  133
BillFroemming  86.857143  100
mlguagliardo   83.428571  120
bgabrielsen    82.571429  100
blinton2       80.571429  100
jcarney3344    80.428571  100
OGJonnyB       79.857143  116
nsaed          79.000000  100
Jgersowsky     68.285714   92

## Bid size vs FAAB remaining

Correlation between a manager's bid and their FAAB remaining before the claim: 0.26 (n=2192). Median bid as a share of remaining FAAB: 3%.

## What drives the winning bid

               n  med_win  med_margin
num_bidders                          
1            584      1.0         1.0
2            255      4.0         2.0
3            126      7.0         3.0
4             54     10.0         3.0
5             30     14.5         2.0
6             15     26.0        11.0
7              4     13.0         2.5
8              2     48.0         6.0

More bidders → higher clearing price, but the second bid is what sets the price; uncontested claims clear at whatever the lone bidder offered (often $0–$5).


## Predictive model (frozen approach, cold-tested)

**Approach, frozen before testing:** predict the *price to beat* (max rival
bid) with quantile gradient boosting (q10–q90) on features knowable before
waivers run: position, week/phase, the target's prior-week points and
season-to-date PPG (demand), and the league's FAAB-remaining landscape.
Trained on auctions not won by Eric, 2020–2024; cold-tested on 2025–2026
(n=173). Eric's needed bid = predicted quantile + $1.

**Results (2025–26 cold test):**
- Actual winning bid fell inside the predicted 50% band 71% of the time
  (conservative) and inside the 80% band 89% (nominal 80%).
- Upper tail — the part Eric bids against — is calibrated: actuals exceeded
  q75 28% of the time (ideal 25%) and q90 11% (ideal 10%).
- Median-bid MAE $4.22 vs $4.88 (position median) and $4.81 (position ×
  phase): the model beats simple baselines, modestly.
- Weakest cell: QB (50%-band coverage 48%, n=21) — treat QB numbers gently.

**What failed, plainly:** naming *who* wins. Five rival-scoring variants
(historical position propensity × typical bid, etc.) were selected on a
2023–24 validation split and tested once on 2025–26: best top-3 hit rates
34–36% vs 33% for picking three names at random. Generic manager habits do
not reveal who wants this week's player — roster need (injuries, byes)
drives that, and it isn't in generic propensity. Salvage: when the winner
*was* in the model's top 3, their actual bid landed in their predicted
range 81% of the time. Bid amounts per manager are predictable; winner
identity is not. The tool labels this honestly.

**Data nugget:** 383 of 1,453 waiver targets (26%) drew only failed claims —
nobody landed the player in that run (invalid/conditional claim chains).

## Caveats

- FAAB-remaining is reconstructed from completed waiver spend plus recorded
  trade FAAB moves; unrecorded adjustments would make it approximate. Treat
  it as "knowable-ish", per the dataset column name.
- 2026 contributes only 44 auctions so far; era drift is handled by the
  train/test split, not eliminated.
- League-specific only. Nothing here generalizes to other leagues.
