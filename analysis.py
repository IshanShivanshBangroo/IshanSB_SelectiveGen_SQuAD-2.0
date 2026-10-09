"""Frozen-cache selective-generation analysis. No model loading or network calls."""
from pathlib import Path

import argparse, ast, base64, collections, csv, gzip, hashlib, json, math, re, string

import numpy as np

from sklearn.metrics import roc_auc_score

SEED = 597

SIGNALS = {"Token probability": "tokprob", "Verbalized confidence": "verbal",
           "Sample agreement": "agree", "Lexical cluster entropy": "neg_entropy",
           "Context-sufficiency self-check": "support"}

NO_ANSWER_RE = re.compile(
    r"\bunanswerable\b|\bnot (?:mentioned|stated|specified|provided|given|included|present|found|available|known|clear)\b"
    r"|\b(?:does|do|did)\s*n[o']t (?:say|mention|specify|state|provide|contain|include|indicate|give|tell)\b"
    r"|\bno (?:information|answer|mention|data)\b|\bcan(?:not|'t| not) be (?:determined|answered|found)\b"
    r"|^\s*(?:unknown|none|n/?a)\s*\.?\s*$", re.I)

def sha(raw): return hashlib.sha256(raw).hexdigest()

def normalize_answer(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())

def token_f1(a, b):
    a, b = normalize_answer(a).split(), normalize_answer(b).split()
    if not a or not b: return float(a == b)
    common = sum((collections.Counter(a) & collections.Counter(b)).values())
    return 2*common/(len(a)+len(b))

def best_f1(a, golds): return max((token_f1(a,g) for g in golds), default=0.)

def exact_match(a, golds): return any(normalize_answer(a)==normalize_answer(g) for g in golds)

def is_no_answer(s): return bool(NO_ANSWER_RE.search(s or ''))

def canon_tokens(s):
    return [(x.lstrip('0') or '0') if x.isdigit() else x for x in normalize_answer(s).split()]

def same_answer(a, b, mode):
    if mode == 'public_f1': return token_f1(a,b)>=.5
    ta,tb = canon_tokens(a),canon_tokens(b)
    if not ta or not tb: return ta==tb
    common=sum((collections.Counter(ta)&collections.Counter(tb)).values())
    numbers=lambda ts:{x for x in ts if any(c.isdigit() for c in x)}
    return 2*common/(len(ta)+len(tb))>=.5 and numbers(ta)==numbers(tb)

def clusters(texts, mode):
    reps,labels=[],[]
    for t in texts:
        key='<declined>' if is_no_answer(t) else t
        for i,rep in enumerate(reps):
            if key==rep=='<declined>' or ('<declined>' not in (key,rep) and same_answer(key,rep,mode)):
                labels.append(i);break
        else: reps.append(key);labels.append(len(reps)-1)
    return labels,reps

def entropy(counts):
    x=np.asarray(counts,float);x=x[x>0]/x.sum()
    return max(0.,float(-(x*np.log2(x)).sum()))

def sample_entropy(samples,mode): return entropy(list(collections.Counter(clusters(samples,mode)[0]).values()))

def item_row(r,mode):
    f=r['forced'];imp=bool(r['is_impossible']);eligible=not is_no_answer(f)
    wrong=imp or best_f1(f,r['golds'])<.5
    ntok=r.get('forced_ntok') or 0
    return dict(set=r['set'],id=r['id'],split=r.get('split'),answerable=not imp,
                eligible=eligible,wrong=wrong,correct=eligible and not wrong,
                wrong_em=imp or not exact_match(f,r['golds']),
                tokprob=math.exp(r['forced_lp_sum']/ntok) if ntok else np.nan,
                verbal=r['verbal'] if r['verbal'] is not None else np.nan,
                agree=float(np.mean([same_answer(s,f,mode) for s in r['samples']])),
                neg_entropy=-sample_entropy(r['samples'],mode),support=r['support_p_yes'],
                abst_released=not is_no_answer(r['abstain']),
                abst_wrong=imp or best_f1(r['abstain'],r['golds'])<.5,
                abst_wrong_em=imp or not exact_match(r['abstain'],r['golds']),
                abst_correct_em=(not is_no_answer(r['abstain'])) and not(imp or not exact_match(r['abstain'],r['golds'])),
                abst_correct=(not is_no_answer(r['abstain'])) and not(imp or best_f1(r['abstain'],r['golds'])<.5))

def arrays(rows):
    return {k:np.asarray([x[k] for x in rows]) for k in rows[0]}

def sorted_groups(scores,eligible):
    s=np.nan_to_num(np.asarray(scores,float),nan=-np.inf)
    idx=np.flatnonzero(eligible);idx=idx[np.argsort(-s[idx],kind='stable')]
    v=s[idx]   # equal scores, including several missing (-inf) ones, stay in one tie group
    cuts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1,len(idx)]
    return [idx[cuts[i]:cuts[i+1]] for i in range(len(cuts)-1)]

def expected_release(scores,eligible,k):
    rel=np.zeros(len(eligible));taken=0
    for g in sorted_groups(scores,eligible):
        if taken>=k:break
        take=min(len(g),k-taken);rel[g]=take/len(g);taken+=take
    return rel

def metrics(rel,wrong,answerable,correct):
    rel=np.asarray(rel,float);n=len(rel);released=rel.sum();errors=float((rel*wrong).sum())
    return dict(n_total=n,expected_released=float(released),expected_errors=errors,
                coverage=float(released/n),risk=errors/released if released else np.nan,
                answerable_declined=float(1-(rel*answerable).sum()/answerable.sum()),
                forced_correct_withheld=float(1-(rel*correct).sum()/correct.sum()) if correct.sum() else np.nan)

def policies(a):
    n=len(a['wrong']);elig=a['eligible'];k=round(.5*n)
    out={'Forced-answer prompt':metrics(elig,a['wrong'],a['answerable'],a['correct'])}
    for name,col in SIGNALS.items():out[name+' gate']=metrics(expected_release(a[col],elig,k),a['wrong'],a['answerable'],a['correct'])
    out["Prompt allows unanswerable"]=metrics(a['abst_released'],a['abst_wrong'],a['answerable'],a['correct'])
    out['Answerability oracle (diagnostic only)']=metrics(elig&a['answerable'],a['wrong'],a['answerable'],a['correct'])
    out['Random eligible gate (expected 50%)']=metrics(elig.astype(float)*min(1.,k/elig.sum()),a['wrong'],a['answerable'],a['correct'])
    return out

def bootstrap(a,n_boot=2000):
    rng=np.random.default_rng(SEED);draws=collections.defaultdict(list)
    for _ in range(n_boot):
        ids=rng.integers(0,len(a['wrong']),len(a['wrong']))
        for name,m in policies({k:v[ids] for k,v in a.items()}).items():draws[name].append(m['risk'])
    return {k:np.asarray(v) for k,v in draws.items()}

def threshold(scores,eligible,n_total,target=.5):
    s=np.sort(np.nan_to_num(np.asarray(scores,float),nan=-np.inf)[eligible])[::-1];k=round(target*n_total)
    if not len(s) or not k:return np.inf,0.
    tau=s[min(k,len(s))-1];gt=int((s>tau).sum());eq=int((s==tau).sum())
    return float(tau),(min(k,len(s))-gt)/eq

def apply_threshold(scores,eligible,tau,q):
    s=np.nan_to_num(np.asarray(scores,float),nan=-np.inf)
    return np.where(s>tau,1.,np.where(s==tau,q,0.))*eligible

def rc_curve(scores,eligible,wrong,n_total):
    cov=[];risk=[];taken=0;wsum=0.
    for g in sorted_groups(scores,eligible):
        for t in range(1,len(g)+1):
            cov.append((taken+t)/n_total);risk.append((wsum+t*wrong[g].sum()/len(g))/(taken+t))
        taken+=len(g);wsum+=wrong[g].sum()
    return np.asarray(cov),np.asarray(risk)

def subset(a,mask): return {k:v[mask] for k,v in a.items()}

def reading_of(text,readings):
    hits=[r['label'] for r in readings if any(f' {normalize_answer(g)} ' in f' {normalize_answer(text)} ' for g in r['golds'] if normalize_answer(g))]
    return 'declined' if is_no_answer(text) else ('both' if len(hits)==2 else hits[0] if hits else 'neither')

def audit_mode(recs,mode,n_boot):
    rs=[r for r in recs if r['set']=='squad'];rows=[item_row(r,mode) for r in rs];a=arrays(rows)
    point=policies(a);boot=bootstrap(a,n_boot)
    for name,m in point.items():m['risk_ci_question_bootstrap']=np.nanpercentile(boot[name],[2.5,97.5]).tolist()
    gates=[name+' gate' for name in SIGNALS];pairs=[]
    for i,g in enumerate(gates):
        for h in ['Forced-answer prompt','Random eligible gate (expected 50%)']+gates[i+1:]:
            pairs.append(dict(a=g,b=h,risk_difference=point[g]['risk']-point[h]['risk'],
                              ci=np.nanpercentile(boot[g]-boot[h],[2.5,97.5]).tolist()))
    cal=subset(a,a['split']=='calib');tst=subset(a,a['split']=='test');held={};rng=np.random.default_rng(SEED)
    held_boot_ids=rng.integers(0,len(tst['wrong']),(n_boot,len(tst['wrong'])))
    for name,col in SIGNALS.items():
        tau,q=threshold(cal[col],cal['eligible'],len(cal['wrong']));rel=apply_threshold(tst[col],tst['eligible'],tau,q)
        m=metrics(rel,tst['wrong'],tst['answerable'],tst['correct']);draw=[]
        for ids in held_boot_ids:draw.append(metrics(rel[ids],tst['wrong'][ids],tst['answerable'][ids],tst['correct'][ids])['risk'])
        held[name]=dict(threshold=tau,tie_release_probability=q,calibration=metrics(apply_threshold(cal[col],cal['eligible'],tau,q),cal['wrong'],cal['answerable'],cal['correct']),test=m,
                        test_risk_ci_conditional_on_calibrated_threshold=np.nanpercentile(draw,[2.5,97.5]).tolist())
    qualities={}
    for name,col in SIGNALS.items():
        cov,risk=rc_curve(a[col],a['eligible'],a['wrong'],len(a['wrong']));mask=a['eligible']&np.isfinite(a[col]);y=~a['wrong'][mask]
        qualities[name]=dict(aurc_full=float(risk.mean()),aurc_from_5_percent=float(risk[cov>=.05].mean()),
                             max_coverage=float(cov[-1]),auroc=float(roc_auc_score(y,a[col][mask])))
    vmask=a['eligible']&np.isfinite(a['verbal']);scores=a['verbal'][vmask];correct=a['correct'][vmask];band=np.clip(np.digitize(scores,np.linspace(0,1,11))-1,0,9)
    bins=[]
    for b in range(10):
        m=band==b
        if m.any():bins.append(dict(bin=b,count=int(m.sum()),mean_confidence=float(scores[m].mean()),accuracy=float(correct[m].mean())))
    ece=sum(x['count']/len(scores)*abs(x['mean_confidence']-x['accuracy']) for x in bins)
    amb=[]
    for r in (r for r in recs if r['set']=='ambig'):
        row=item_row(r,mode);sa,sb=[x['samples'] for x in r['readings']];labs,_=clusters(sa+sb,mode);la,lb=labs[:len(sa)],labs[len(sa):]
        total=entropy(list(collections.Counter(labs).values()));within=.5*entropy(list(collections.Counter(la).values()))+.5*entropy(list(collections.Counter(lb).values()))
        x=dict(id=r['id'],question=r['question'],forced=r['forced'],reading_picked=reading_of(r['forced'],r['readings']),
               original_entropy=sample_entropy(r['samples'],mode),pooled_total=total,within=within,between=total-within,
               clarified_correct=sum(best_f1(rd['forced'],rd['golds'])>=.5 for rd in r['readings']),gates={})
        for name,col in SIGNALS.items():
            tau,q=threshold(a[col],a['eligible'],len(a['wrong']));ctau,cq=threshold(cal[col],cal['eligible'],len(cal['wrong']))
            val=row[col]
            x['gates'][name]=dict(old_full_sample_all_ties_release=bool(row['eligible'] and np.isfinite(val) and val>=tau),
                                 full_sample_randomized_release_probability=float(apply_threshold(np.array([val]),np.array([row['eligible']]),tau,q)[0]),
                                 calibration_only_release_probability=float(apply_threshold(np.array([val]),np.array([row['eligible']]),ctau,cq)[0]))
        amb.append(x)
    base=point['Forced-answer prompt']; ab=point['Prompt allows unanswerable']
    return dict(mode=mode,points=point,paired_differences_exploratory=pairs,held_out=held,ranking=qualities,
                verbal_ece=ece,verbal_bins=bins,ambiguous=amb,
                base_counts=dict(n_total=len(rs),released=int(a['eligible'].sum()),wrong_released=int((a['wrong']&a['eligible']).sum()),
                                 answerable=int(a['answerable'].sum()),correct_answerable=int(a['correct'].sum()),
                                 exact_match_answerable=int((~a['wrong_em']&a['answerable']).sum()),
                                 spontaneous_declines=[dict(id=r['id'],answer=r['forced']) for r,x in zip(rs,rows) if not x['eligible']],
                                 prompt_abstain_released=int(a['abst_released'].sum()),prompt_abstain_wrong_released=int((a['abst_wrong']&a['abst_released']).sum()),
                                 prompt_abstain_correct=int(a['abst_correct'].sum())),
                exact_match_policies=policies({**a,'wrong':a['wrong_em'],'correct':a['eligible']&~a['wrong_em'],
                                              'abst_wrong':a['abst_wrong_em'],'abst_correct':a['abst_correct_em']}))

CACHE_SHA256 = "149f421f930223a53d163afa3186968bc8482497dfb0498e3af20b6e94cfae6a"

def load_verified_cache(path="results_cache.jsonl"):
    raw = Path(path).read_bytes()
    assert sha(raw) == CACHE_SHA256, "This file differs from the frozen run; do not mix provenance."
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    meta = [r for r in records if r.get("set") == "meta"]
    items = [r for r in records if r.get("set") != "meta"]
    assert len(meta) == 1 and len(items) == 418
    assert len({(r["set"], r["id"]) for r in items}) == 418
    assert collections.Counter(r["set"] for r in items) == {"squad":400,"workshop":6,"ambig":12}
    assert all(len(r["samples"]) == 10 for r in items)
    return meta[0], items

def run_analysis(path="results_cache.jsonl", bootstrap_resamples=2000):
    meta, items = load_verified_cache(path)
    corrected = audit_mode(items,"number_guard",bootstrap_resamples)
    legacy = policies(arrays([item_row(r,"public_f1") for r in items if r["set"]=="squad"]))
    return dict(cache_sha256=CACHE_SHA256, model=meta["model"], seed=SEED,
                bootstrap_resamples=bootstrap_resamples, bootstrap_unit="question",
                no_new_inference=True, corrected=corrected, legacy_point_sensitivity=legacy)

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Replay the frozen cache; no GPU, model loading or downloads.")
    parser.add_argument("--cache",default="results_cache.jsonl")
    parser.add_argument("--output",default="results.json")
    parser.add_argument("--bootstrap",type=int,default=2000)
    args=parser.parse_args()
    result=run_analysis(args.cache,args.bootstrap)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False))
    print("Saved",args.output,"from cache",CACHE_SHA256)
