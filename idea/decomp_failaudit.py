#!/usr/bin/env python3
"""Negative-gain subgroup systematic failure audit (red-team item #7).

For Expression-Decomposed CRS (frozen GroundingDINO grounding, 2nd main result):
the matched recall@size~3 subgroup analysis found decomposition helps overall but
HURTS testA n_gt==1 (-0.148). The prior claim attributed this to "gRefCOCO
annotation ambiguity" based on eyeballing 12 cases. Red-team verdict: sample
systematically and report failure-type proportions, don't trust 12 cases.

Definition of a NEGATIVE-GAIN case (this is what decomposition actually harms):
  among truly-decomposed keys present in both orig+decomp dumps, the decomp pool's
  ORACLE recall (fraction of GT covered by ANY pool box at IoU>=0.5, ignoring
  scores) is STRICTLY LESS than the full-expression pool's oracle recall.

Heuristic failure-type rules (auto-classification; honest about its limits).
Applied IN ORDER, first match wins:

  1. detector_miss
     decomp pool oracle recall < 0.5 (i.e. even with the union of all decomposed
     prompts' boxes, the detector simply did not box the GT). The harm is a
     detection failure, not a parsing/annotation problem.
     Confidence: HIGH. This is measured directly from coverage, no language guess.

  2. annotation_ambiguity   (only meaningful when n_gt==1)
     The expression has an EXPLICIT coordinate/list structure that names two or
     more distinct referents, yet GT labels only one box. Trigger when the
     expression matches a coordination pattern:
        - " and " / " & " joining two NOUN-bearing sides, OR
        - a comma joining two noun phrases, OR
        - " with " / " next to " / " behind " / " in front of " linking two
          distinct objects (a relational two-object phrase).
     AND the two sides do NOT share a common head noun (to separate from the
     "big and red car" attribute-stacking case, which is over_decomposition).
     Confidence: MEDIUM. Language is genuinely dual-referent; supports the
     "label ambiguity" reading. False positives possible when " and " joins two
     attributes of one object whose head noun differs lexically.

  3. over_decomposition      (only meaningful when n_gt==1)
     n_parts > n_gt (==1) AND it's NOT clean coordination of distinct objects,
     i.e. the parser split a single-object phrase into >1 prompt. Coarse proxy:
     n_parts>1 with n_gt==1 and the two sides SHARE a head noun, or one side has
     no detectable noun (attribute stacking like "big and red", "tall and thin").
     Confidence: LOW. Cannot truly tell apart from annotation_ambiguity without
     human reading; we approximate with shared-head-noun / no-noun-on-a-side.
     This is the rule most likely to be wrong; treat its count as an upper-ish
     estimate of "parser split one object".

  4. parser_error
     n_parts structure inconsistent with the surface language: the expression has
     NO coordination/relation cue at all (no " and ", no comma, no relational
     preposition) yet n_parts>1. The parser invented a split with no linguistic
     basis. Confidence: LOW-MEDIUM (surface-cue based).

  5. other
     everything else (e.g. n_parts==1 negative-gain, or n_gt>1 cases that still
     lost coverage). Catch-all / not auto-classifiable.

Limits stated plainly: rules 2/3/4 are surface-string heuristics with a tiny
noun lexicon, NOT a parser or human judgement. They are meant to give a defensible
PROPORTION with a known confidence ordering (detector_miss HIGH > ambiguity MED >
over_decomp/parser LOW), not a per-case ground truth. The headline question is
answered by comparing the MED-confidence annotation_ambiguity share against the
combined LOW-confidence over_decomposition + parser_error share.
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crs_protocol as P

SPLITS = ["val", "testA", "testB"]

# ---- tiny surface-noun lexicon (object head-noun cues) ----
NOUNS = {
    "man","woman","men","women","guy","girl","boy","kid","child","person","people",
    "lady","player","baby","dude","human","child","children",
    "car","truck","bus","train","plane","boat","bike","motorcycle","vehicle",
    "dog","cat","horse","cow","sheep","bird","elephant","bear","zebra","giraffe",
    "animal","cake","pizza","sandwich","banana","apple","orange","donut","food",
    "chair","table","couch","sofa","bed","bench","desk","bottle","cup","glass",
    "bowl","plate","fork","knife","spoon","laptop","phone","tv","book","clock",
    "ball","racket","bat","glove","frisbee","kite","umbrella","bag","suitcase",
    "shirt","hat","jacket","tie","shoe","helmet","plant","flower","tree","sign",
    "light","window","door","wall","building","house","sky","ground","field",
    "head","hand","face","leg","arm","hair","one","thing","object","piece","item",
}
# attribute-ish words that often stack with " and " on a single object
ATTRS = {
    "big","small","large","tiny","tall","short","long","wide","red","blue","green",
    "yellow","black","white","orange","purple","pink","brown","gray","grey","dark",
    "light","bright","old","young","new","front","back","left","right","top","bottom",
    "middle","center","far","near","close","first","second","third","striped","plaid",
    "round","square","empty","full","open","closed","clean","dirty","wet","dry",
}
REL_CUES = [" with ", " next to ", " behind ", " in front of ", " on top of ",
            " under ", " beside ", " near ", " holding ", " wearing "]
COORD_AND = [" and ", " & "]


def has_noun(s):
    toks = re.findall(r"[a-z]+", s.lower())
    return any(t in NOUNS for t in toks)


def head_nouns(s):
    toks = re.findall(r"[a-z]+", s.lower())
    return set(t for t in toks if t in NOUNS)


def is_attr_side(s):
    """side looks like pure attribute (no noun, has an attribute word)."""
    toks = re.findall(r"[a-z]+", s.lower())
    nouns = [t for t in toks if t in NOUNS]
    attrs = [t for t in toks if t in ATTRS]
    return len(nouns) == 0 and len(attrs) >= 1


def split_coord(expr):
    """split on first ' and '/'&'; return (left,right) or None."""
    low = expr.lower()
    for c in COORD_AND:
        i = low.find(c)
        if i >= 0:
            return expr[:i], expr[i + len(c):]
    return None


def has_comma_two_np(expr):
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    return len(parts) >= 2 and sum(1 for p in parts if has_noun(p)) >= 2


def has_rel_two_obj(expr):
    low = expr.lower()
    for c in REL_CUES:
        i = low.find(c)
        if i >= 0:
            left, right = expr[:i], expr[i + len(c):]
            if has_noun(left) and has_noun(right):
                return True
    return False


def ling_pattern(expr, n_gt, n_parts):
    """LANGUAGE-only diagnosis, independent of coverage. Returns one of
    annotation_ambiguity / over_decomposition / parser_error / none.
    Only meaningful for n_gt==1 (single GT box but possibly multi-referent text)."""
    if n_gt != 1:
        return "none"
    coord = split_coord(expr)
    has_and = coord is not None
    distinct_objects = shared_or_attr = False
    if coord:
        l, r = coord
        ln, rn = head_nouns(l), head_nouns(r)
        if ln and rn and not (ln & rn):
            distinct_objects = True
        if (ln and rn and (ln & rn)) or is_attr_side(l) or is_attr_side(r):
            shared_or_attr = True
    if (has_and and distinct_objects) or has_comma_two_np(expr) or has_rel_two_obj(expr):
        return "annotation_ambiguity"
    if n_parts and n_parts > 1 and (shared_or_attr or (has_and and not distinct_objects)):
        return "over_decomposition"
    low = expr.lower()
    no_cue = (not has_and) and ("," not in expr) and not any(c in low for c in REL_CUES)
    if n_parts and n_parts > 1 and no_cue:
        return "parser_error"
    return "none"


def classify(expr, n_gt, n_parts, dec_recall):
    """return (label, confidence) per the documented ordered rules.

    NOTE on ordering subtlety (important for interpretation):
    when n_gt==1 and the expression is clearly dual-referent coordination, the
    decomp pool can score dec_recall==0.0 precisely BECAUSE decomposition split
    the dual referent into per-object prompts that each box a DIFFERENT object
    than the single GT. That is annotation_ambiguity expressed as zero coverage,
    not a generic detector blindness. So for n_gt==1 we let the LANGUAGE pattern
    win over the coverage-based detector_miss; detector_miss is reserved for
    n_gt==1 with NO multi-referent language cue (genuinely failed to box one
    object), and for the n_gt>1 cases."""
    if n_gt == 1:
        lp = ling_pattern(expr, n_gt, n_parts)
        if lp == "annotation_ambiguity":
            return "annotation_ambiguity", "MED"
        if lp == "over_decomposition":
            return "over_decomposition", "LOW"
        if lp == "parser_error":
            return "parser_error", "LOW-MED"
        # no language cue: a real single-object detection failure
        if dec_recall < 0.5:
            return "detector_miss", "HIGH"
        return "other", "NA"

    # n_gt>1 path: coverage-driven. detector_miss if pool lost the GT badly,
    # else 'other' (partial coverage loss on a genuine multi-target case).
    if dec_recall < 0.5:
        return "detector_miss", "HIGH"
    return "other", "NA"


def pool_recall(rec):
    gt = rec["gt_boxes_xyxy"]
    if not gt:
        return None
    boxes = rec["pred_boxes_xyxy"]
    cov = sum(1 for q in gt if any(P.iou(p, q) >= 0.5 for p in boxes))
    return cov / len(gt)


def run_split(sp):
    orig = P._read(f"{P.B}/gdino_gref_{sp}.jsonl")
    dec = P._read(f"{P.B}/gdino_gref_{sp}_decomp.jsonl")
    dkeys = [k for k, v in dec.items() if v.get("decomposed") and k in orig]

    neg = []  # negative-gain cases
    for k in dkeys:
        ro, rd = orig[k], dec[k]
        pr_o, pr_d = pool_recall(ro), pool_recall(rd)
        if pr_o is None or pr_d is None:
            continue
        if pr_d < pr_o:  # decomposition strictly reduced oracle coverage
            expr = rd.get("expression", ro.get("expression", ""))
            n_gt = rd.get("n_gt")
            n_parts = rd.get("n_parts")
            label, conf = classify(expr, n_gt, n_parts, pr_d)
            neg.append(dict(key=k, expr=expr, n_gt=n_gt, n_parts=n_parts,
                            full_rec=pr_o, dec_rec=pr_d, label=label, conf=conf))
    return dkeys, neg


def tally(rows):
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    n = len(rows)
    out = []
    for lbl in ["annotation_ambiguity", "over_decomposition", "detector_miss",
                "parser_error", "other"]:
        cnt = c.get(lbl, 0)
        out.append((lbl, cnt, (cnt / n * 100 if n else 0.0)))
    return out


def main():
    all_neg = {}
    for sp in SPLITS:
        dkeys, neg = run_split(sp)
        all_neg[sp] = neg
        print(f"\n{'='*70}\n### split={sp}")
        print(f"truly-decomposed keys (in orig+decomp): {len(dkeys)}")
        print(f"NEGATIVE-GAIN cases (decomp oracle recall < full): {len(neg)}")
        print(f"  ({len(neg)/len(dkeys)*100:.1f}% of decomposed keys)" if dkeys else "")
        print(f"\n-- failure-type breakdown (all negative-gain) --")
        for lbl, cnt, pct in tally(neg):
            print(f"   {lbl:22s} {cnt:4d}  {pct:5.1f}%")

        sub = [r for r in neg if r["n_gt"] == 1]
        print(f"\n-- n_gt==1 subgroup (the headline -0.148 driver), n={len(sub)} --")
        for lbl, cnt, pct in tally(sub):
            print(f"   {lbl:22s} {cnt:4d}  {pct:5.1f}%")

    # focused testA n_gt==1 verdict
    print(f"\n{'='*70}\n### HEADLINE: testA n_gt==1 negative-gain verdict")
    sub = [r for r in all_neg["testA"] if r["n_gt"] == 1]
    n = len(sub)
    from collections import Counter
    c = Counter(r["label"] for r in sub)
    amb = c.get("annotation_ambiguity", 0)
    over = c.get("over_decomposition", 0)
    parse = c.get("parser_error", 0)
    miss = c.get("detector_miss", 0)
    print(f"total testA n_gt==1 negative-gain cases: {n}")
    if n:
        print(f"  annotation_ambiguity (MED conf, supports orig claim): {amb} ({amb/n*100:.1f}%)")
        print(f"  over_decomposition    (LOW conf, refutes orig claim): {over} ({over/n*100:.1f}%)")
        print(f"  parser_error          (LOW-MED conf, refutes):        {parse} ({parse/n*100:.1f}%)")
        print(f"  detector_miss         (HIGH conf, orthogonal):        {miss} ({miss/n*100:.1f}%)")
        print(f"  --> ambiguity share vs (over_decomp+parser) share: "
              f"{amb/n*100:.1f}% vs {(over+parse)/n*100:.1f}%")

    # cross-tab: pure LANGUAGE pattern for ALL testA n_gt==1 negative-gain,
    # independent of coverage -- shows entanglement when dec_recall==0
    print(f"\n-- LANGUAGE-only pattern (coverage-independent) for the same {n} cases --")
    lc = Counter(ling_pattern(r["expr"], 1, r["n_parts"]) for r in sub)
    for lbl in ["annotation_ambiguity", "over_decomposition", "parser_error", "none"]:
        cnt = lc.get(lbl, 0)
        print(f"   {lbl:22s} {cnt:4d}  {cnt/n*100:5.1f}%" if n else f"   {lbl}: 0")
    zero_cov = sum(1 for r in sub if r["dec_rec"] == 0.0)
    zero_amb = sum(1 for r in sub if r["dec_rec"] == 0.0
                   and ling_pattern(r["expr"], 1, r["n_parts"]) == "annotation_ambiguity")
    print(f"   of these, dec_recall==0.0: {zero_cov}; "
          f"and ALSO multi-referent language: {zero_amb} "
          f"(decomp split dual referent -> each part boxes the wrong single object)")

    # sample display: prefer testA n_gt==1, then fill from other negatives
    print(f"\n{'='*70}\n### SAMPLE (eyeball-verify the heuristic), up to 30 cases")
    show = sorted([r for r in all_neg["testA"] if r["n_gt"] == 1],
                  key=lambda r: (r["label"], r["dec_rec"]))
    pool_extra = [r for sp in SPLITS for r in all_neg[sp]
                  if not (sp == "testA" and r["n_gt"] == 1)]
    show = show + pool_extra
    print(f"{'label':22s} {'conf':7s} {'n_gt':4s} {'np':3s} {'full':5s} {'dec':5s}  expression")
    for r in show[:30]:
        print(f"{r['label']:22s} {r['conf']:7s} {str(r['n_gt']):4s} "
              f"{str(r['n_parts']):3s} {r['full_rec']:.2f}  {r['dec_rec']:.2f}  "
              f"{r['expr'][:70]}")


if __name__ == "__main__":
    main()
