#!/usr/bin/env python3
"""
Run one improvement-loop experiment against the current champion.

Usage: python run_experiment.py <exp_id>
Experiments are single changes relative to the champion spec in
loop_state.json. On acceptance the champion spec advances (hill-climbing).
Every run appends to experiment_log.md and checkpoints/progress.md, and
saves full metrics to loop_runs/<exp_id>.json.
"""
import copy, json, os, sys, datetime
import loop_lib as L

OUT = L.OUT
LOG_MD = os.path.join(OUT, 'experiment_log.md')
PROG = os.path.join(OUT, 'checkpoints', 'progress.md')

DROP_OFF = ['prior_off_rank', 'prio_top', 'prio_mid', 'prio_bottom']
DROP_MISS = ['missed_prior', 'missed5']

def map_feats(spec, fn):
    """Apply a feature-list transform to the shared list AND any
    per-target overrides (feats_by_target shadows feats in eval)."""
    spec['feats'] = fn(list(spec['feats']))
    fbt = spec.get('feats_by_target')
    if fbt:
        spec['feats_by_target'] = {t: fn(list(v)) for t, v in fbt.items()}
    return spec

def variant(champ, exp_id):
    spec = copy.deepcopy(champ)
    kind = 'clf'
    desc = ''
    if exp_id == 'e01_val':
        spec = map_feats(spec, lambda f: f + L.GROUPS['VAL'])
        desc = 'add VAL relative-value/interaction features ' \
               '(ppg_per_dollar, price_x_age, ecr_vs_prior_rank, price_per_ecr)'
    elif exp_id == 'e02_eff':
        spec = map_feats(spec, lambda f: f + L.GROUPS['EFF'])
        desc = 'add EFF prior-season efficiency-rate features (pts/opp, ' \
               'pts/snap, catch rate, aDOT, RZ share, QB pass/rush rates)'
    elif exp_id == 'e02b_eff_starter':
        fbt = {t: list(spec['feats']) for t in ('top5', 'starter')}
        fbt['starter'] = spec['feats'] + L.GROUPS['EFF']
        spec['feats_by_target'] = fbt
        desc = 'EFF efficiency features applied to the STARTER head only ' \
               '(post-hoc scoped variant of rejected e02: EFF helped starter ' \
               '+0.004 AUC / hurt top-5 -0.016 as a global add; gated identically)'
    elif exp_id == 'e03_dropoff':
        spec = map_feats(spec, lambda f: [x for x in f if x not in DROP_OFF])
        desc = 'drop prior-year offense features (prior_off_rank, prio_*) - ' \
               'effect measured ~0 post-2023 in v2 report s.11'
    elif exp_id == 'e04_dropmissed':
        spec = map_feats(spec, lambda f: [x for x in f if x not in DROP_MISS])
        desc = 'drop missed_prior/missed5 (injury effect flipped sign in ' \
               '2024-25 test actuals; model still applies stale discount)'
    elif exp_id == 'e05_ensemble':
        spec['ensemble_w'] = 0.5
        desc = 'fixed 50/50 probability average of XGBoost with the ' \
               'LR-extended head (LR-ext beat XGB on holdout top-5)'
    elif exp_id == 'e06_cluster':
        spec['weight_mode'] = 'cluster'
        desc = 'cluster-normalized sample weights: recency decay / times ' \
               'player-season appears across leagues, renormalized'
    elif exp_id == 'e07_team':
        spec = map_feats(spec, lambda f: f + L.GROUPS['TEAM'])
        desc = 'add TEAM prior-year team pass-volume context (pass att/game, ' \
               'pass share, pass rank, plays/game)'
    elif exp_id == 'e08_traj':
        spec = map_feats(spec, lambda f: f + L.GROUPS['TRAJ'])
        desc = 'add TRAJ two-year trajectory (prior2 PPG/rank, ppg_delta, ' \
               'rank_improve)'
    elif exp_id == 'e09_wk':
        spec = map_feats(spec, lambda f: f + L.GROUPS['WK'])
        desc = 'add WK prior-season weekly consistency (points CV, ' \
               'floor-week share, 90th-pct week)'
    elif exp_id == 'e10_hp_d3':
        spec['params'] = 'B_d3_reg'
        desc = 'hyperparameters: max_depth 3 (min_child_weight 50) instead ' \
               'of champion depth 2'
    elif exp_id == 'e11_posblend':
        spec['pos_blend'] = {'top5|WR': 0.4, 'starter|TE': 0.4}
        desc = 'per-position blend for weak cells: 0.6 pooled + 0.4 ' \
               'per-position model for WR top-5 and TE starter (fixed weights)'
    elif exp_id == 'e12_cal':
        spec['calibrator'] = 'shrink_oof'
        kind = 'cal'
        desc = 'train-fold-only shrink calibration: s in {1,.95,...,.7} ' \
               'toward train base rate chosen by 5-fold OOF Brier inside ' \
               'each training fold; transport to both test folds'
    else:
        raise SystemExit(f'unknown experiment {exp_id}')
    return spec, kind, desc

def fmt_delta(base, new):
    out = []
    for t in ('top5', 'starter'):
        for fold in ('wf', 'holdout'):
            out.append(f"{t}/{fold} AUC {new[t][fold]['auc']-base[t][fold]['auc']:+.4f} "
                       f"Brier {new[t][fold]['brier']-base[t][fold]['brier']:+.4f}")
    return '; '.join(out)

def main():
    exp_id = sys.argv[1]
    state = L.load_state()
    if exp_id == 'baseline':
        res = L.run_spec(copy.deepcopy(L.CHAMPION_SPEC), 'champion_v2')
        state = {'champion_spec': copy.deepcopy(L.CHAMPION_SPEC),
                 'champion_metrics': L.metric_table(res),
                 'history': [{'exp_id': 'baseline',
                              'decision': 'reference',
                              'when': datetime.datetime.now().isoformat()}]}
        json.dump(state, open(L.STATE_F, 'w'), indent=1, default=str)
        L.save_run('baseline', res, {'accept': None, 'reasons': ['reference']})
        print(L.summarize(res))
        print('BASELINE SAVED')
        return
    if state is None:
        raise SystemExit('run baseline first')
    champ = state['champion_spec']
    # reconstruct champion metrics object from stored table where possible;
    # we keep full champion results in loop_runs/_champion.json
    champ_full_f = os.path.join(L.RUNS_D, '_champion.json')
    if os.path.exists(champ_full_f):
        champ_res = json.load(open(champ_full_f))['results']
    else:
        champ_res = None

    spec, kind, desc = variant(champ, exp_id)
    res = L.run_spec(spec, exp_id)
    if champ_res is None:
        # first experiment: evaluate champion for comparison (and cache it)
        cres = L.run_spec(copy.deepcopy(champ), 'champion_cmp')
        L.save_run('_champion', cres, {'accept': None})
        champ_res = {k: v for k, v in cres.items() if k != '_store'}

    accept, reasons = L.decide(champ_res, res, kind=kind)
    base_tbl = L.metric_table(champ_res)
    new_tbl = L.metric_table(res)
    rec = L.save_run(exp_id, res,
                     {'accept': bool(accept), 'reasons': reasons, 'kind': kind},
                     extra={'description': desc, 'spec': spec,
                            'champion_spec_before': champ})
    # log
    with open(LOG_MD, 'a') as f:
        f.write(f"\n## {exp_id} — {desc}\n\n")
        f.write(f"- decision: **{'ACCEPT' if accept else 'REJECT'}** "
                f"({'; '.join(reasons)})\n")
        f.write(f"- deltas vs champion: {fmt_delta(champ_res, res)}\n")
        f.write(f"- new metrics: `{json.dumps(new_tbl)}`\n")
    with open(PROG, 'a') as f:
        f.write(f"- [{datetime.datetime.now():%H:%M}] {exp_id}: "
                f"{'ACCEPT' if accept else 'REJECT'} — {desc}\n")
    state['history'].append({'exp_id': exp_id, 'decision':
                             'accept' if accept else 'reject',
                             'when': datetime.datetime.now().isoformat(),
                             'description': desc, 'reasons': reasons})
    if accept:
        state['champion_spec'] = spec
        state['champion_metrics'] = new_tbl
        L.save_run('_champion', res, {'accept': None,
                                      'reasons': [f'champion after {exp_id}']})
    json.dump(state, open(L.STATE_F, 'w'), indent=1, default=str)
    print(L.summarize(res))
    print(f"DECISION: {'ACCEPT' if accept else 'REJECT'} — {reasons}")

if __name__ == '__main__':
    main()
