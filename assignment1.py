# =============================================================================
# DSAI 582 — Agentic AI | Assignment 1
# Model : Qwen2.5-0.5B-Instruct  (frozen weights)
# Data  : GSM8K main config
# Runs  : CPU-optimised (int8 quant via bitsandbytes if available)
# =============================================================================

import re, json, ast, math, operator, sys, time, io
from collections import Counter

# Force UTF-8 output on Windows to avoid cp1254 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

PYTHON = sys.executable

# ── 1. Model ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {MODEL_NAME} …", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,         # CPU-safe
    low_cpu_mem_usage=True,
)
model.eval()
for p in model.parameters():
    p.requires_grad = False

DEVICE = torch.device("cpu")
print(f"Model ready on {DEVICE}", flush=True)

# ── 2. Dataset ────────────────────────────────────────────────────────────────
print("Loading GSM8K …", flush=True)
ds        = load_dataset("openai/gsm8k", "main")
train_all = ds["train"]
test_all  = ds["test"]

dev_examples  = [train_all[i] for i in range(0, 10)]   # train 1-10
shot_examples = [train_all[i] for i in range(10, 14)]  # train 11-14
test_examples = [test_all[i]  for i in range(0, 30)]   # test  1-30
print(f"dev={len(dev_examples)}  shots={len(shot_examples)}  test={len(test_examples)}", flush=True)

# ── 3. Helpers ────────────────────────────────────────────────────────────────
def extract_answer(text):
    for pat in [
        r'####\s*(-?[\d,]+(?:\.\d+)?)',
        r'[Aa]nswer\s*(?:is|:)\s*(-?[\d,]+(?:\.\d+)?)',
    ]:
        m = re.search(pat, text)
        if m: return m.group(1).replace(',','').strip()
    nums = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    return nums[-1].replace(',','') if nums else None

def ref_answer(sol):
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', sol)
    return m.group(1).replace(',','').strip() if m else None

def match(pred, ref):
    if pred is None or ref is None: return False
    try:    return abs(float(pred)-float(ref)) < 1e-6
    except: return str(pred).strip()==str(ref).strip()

def generate(prompt, max_new=300, temp=1.0, do_sample=False, n=1):
    msgs  = [{"role":"user","content":prompt}]
    text  = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp   = tokenizer(text, return_tensors="pt")
    in_len= inp["input_ids"].shape[1]
    kw    = dict(max_new_tokens=max_new, do_sample=do_sample,
                 num_return_sequences=n, pad_token_id=tokenizer.eos_token_id)
    if do_sample: kw["temperature"] = temp
    with torch.no_grad():
        out = model.generate(**inp, **kw)
    gens, total = [], 0
    for o in out:
        toks = o[in_len:]
        total += len(toks)
        gens.append(tokenizer.decode(toks, skip_special_tokens=True))
    return gens, total

# ── 4. Few-shot prefix ────────────────────────────────────────────────────────
def build_prefix():
    p = ""
    for ex in shot_examples:
        p += f"Question: {ex['question']}\nSolution: {ex['answer']}\n\n"
    return p

PREFIX = build_prefix()

# ── 5. Q1 Methods ─────────────────────────────────────────────────────────────
SOLVE_INSTR = (
    "Solve the math problem step by step. "
    "Write your final answer as: #### <number>\n\n"
)

def zero_shot(q):
    prompt = SOLVE_INSTR + f"Question: {q}\nSolution:"
    r, tok = generate(prompt)
    return extract_answer(r[0]), tok, 1, r[0]

def four_shot(q):
    prompt = PREFIX + f"Question: {q}\nSolution:"
    r, tok = generate(prompt)
    return extract_answer(r[0]), tok, 1, r[0]

def self_consistency(q, n=4):
    prompt = PREFIX + f"Question: {q}\nSolution:"
    rs, tok = generate(prompt, temp=0.7, do_sample=True, n=n)
    answers = [extract_answer(r) for r in rs]
    valid   = [a for a in answers if a]
    best    = Counter(valid).most_common(1)[0][0] if valid else None
    return best, tok, n, rs

def reflection(q):
    p1 = PREFIX + f"Question: {q}\nSolution:"
    r1, t1 = generate(p1)
    p2 = (f"Review your solution and fix any mistakes.\n"
          f"Question: {q}\nYour solution:\n{r1[0]}\n"
          f"Revised solution (end with #### <number>):")
    r2, t2 = generate(p2)
    ans = extract_answer(r2[0]) or extract_answer(r1[0])
    return ans, t1+t2, 2, (r1[0], r2[0])

# ── 6. Q1 Evaluation ─────────────────────────────────────────────────────────
def eval_method(fn, name, examples):
    rows = []
    ttok = tcall = ok = 0
    print(f"\n{'-'*55}\n  {name}\n{'-'*55}", flush=True)
    for i, ex in enumerate(examples):
        q   = ex["question"]
        ref = ref_answer(ex["answer"])
        ans, tok, calls, _ = fn(q)
        correct = match(ans, ref)
        ok    += int(correct)
        ttok  += tok
        tcall += calls
        sym = "✓" if correct else "✗"
        print(f"  [{i+1:2d}/30] ref={str(ref):>8}  pred={str(ans):>8}  {sym}  tok={tok}", flush=True)
        rows.append({"idx":i+1,"q":q,"ref":ref,"pred":ans,"ok":correct,"tok":tok,"calls":calls})
    acc = ok/len(examples)
    print(f"\n  Accuracy {ok}/{len(examples)} = {acc:.1%}  |  tokens={ttok:,}  calls={tcall:,}", flush=True)
    return {"method":name,"accuracy":acc,"correct":ok,
            "total_tokens":ttok,"total_calls":tcall,"results":rows}

# ── 7. Q2 Safe Calculator ─────────────────────────────────────────────────────
_OPS = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
        ast.Div:operator.truediv, ast.FloorDiv:operator.floordiv,
        ast.Mod:operator.mod, ast.Pow:operator.pow,
        ast.USub:operator.neg, ast.UAdd:operator.pos}
_FUNCS = {"sqrt":math.sqrt,"ceil":math.ceil,"floor":math.floor,
          "abs":abs,"round":round}

def safe_calc(expr):
    """AST-based evaluator — no eval(), no exec()."""
    try: tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e: return None, f"SyntaxError: {e}"
    def ev(node):
        if isinstance(node, ast.Expression): return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value,(int,float)): return float(node.value)
            raise ValueError(f"bad constant {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in _OPS: raise ValueError(f"bad op {op.__name__}")
            l,r = ev(node.left), ev(node.right)
            if op in (ast.Div,ast.FloorDiv,ast.Mod) and r==0: raise ZeroDivisionError
            return _OPS[op](l,r)
        if isinstance(node, ast.UnaryOp):
            op = type(node.op)
            if op not in _OPS: raise ValueError(f"bad unary {op.__name__}")
            return _OPS[op](ev(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func,ast.Name): raise ValueError("only simple calls")
            fn = node.func.id
            if fn not in _FUNCS: raise ValueError(f"bad func {fn}")
            return _FUNCS[fn](*[ev(a) for a in node.args])
        raise ValueError(f"bad node {type(node).__name__}")
    try:
        res = ev(tree)
        return (int(res) if res==int(res) else res), None
    except Exception as e:
        return None, str(e)

# ── 8. Q2 Agent ───────────────────────────────────────────────────────────────
_SYS = (
    "You are a math agent. Use the calculator by writing:\n"
    "  CALC: <arithmetic expression>\n"
    "Give the final answer as:\n"
    "  ANSWER: <number>\n"
    "Rules: max 3 CALC calls; pure arithmetic only; no variables.\n\n"
)

def _parse(text):
    m = re.search(r'CALC\s*:\s*(.+?)(?:\n|$)', text)
    if m: return "calc", m.group(1).strip()
    m = re.search(r'ANSWER\s*:\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m: return "answer", m.group(1).replace(',','').strip()
    return "none", None

def tool_agent(q, max_calc=3):
    conv = _SYS + f"Problem: {q}\n\nSolve step by step:\n"
    total_tok = mcalls = ccalls = 0
    traj, ans = [], None
    for _ in range(max_calc+2):
        rs, tok = generate(conv, max_new=300)
        total_tok += tok; mcalls += 1
        resp = rs[0]; traj.append(("model", resp))
        action, val = _parse(resp)
        if action == "answer": ans = val; break
        elif action == "calc" and ccalls < max_calc:
            ccalls += 1
            res, err = safe_calc(val)
            obs = f"Result: {res}" if err is None else f"Error: {err}"
            traj.append(("tool", f"CALC: {val} → {obs}"))
            conv += resp + f"\n[Tool] {obs}\n"
        else:
            ans = extract_answer(resp); break
    return ans, total_tok, mcalls, ccalls, traj

def eval_agent(examples):
    rows = []
    ttok = tmcall = tccall = ok = 0
    print(f"\n{'-'*55}\n  Q2: Tool-Using Agent\n{'-'*55}", flush=True)
    for i, ex in enumerate(examples):
        q   = ex["question"]
        ref = ref_answer(ex["answer"])
        ans, tok, mc, cc, _ = tool_agent(q)
        correct = match(ans, ref)
        ok += int(correct); ttok += tok; tmcall += mc; tccall += cc
        sym = "✓" if correct else "✗"
        print(f"  [{i+1:2d}/30] ref={str(ref):>8}  pred={str(ans):>8}  {sym}  tok={tok}  calcs={cc}", flush=True)
        rows.append({"idx":i+1,"q":q,"ref":ref,"pred":ans,"ok":correct,
                     "tok":tok,"model_calls":mc,"calc_calls":cc})
    acc = ok/len(examples)
    print(f"\n  Accuracy {ok}/{len(examples)} = {acc:.1%}  |  tokens={ttok:,}  model_calls={tmcall:,}  calc_calls={tccall:,}", flush=True)
    return {"method":"Tool-Using Agent","accuracy":acc,"correct":ok,
            "total_tokens":ttok,"total_model_calls":tmcall,
            "total_calc_calls":tccall,"results":rows}

# ── 9. Run All ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Q1: Prompting Methods")
print("=" * 55, flush=True)

t0 = time.time()
q1 = []
q1.append(eval_method(zero_shot,        "Zero-Shot",                    test_examples))
q1.append(eval_method(four_shot,        "Four-Shot",                    test_examples))
q1.append(eval_method(self_consistency, "Self-Consistency (4 samples)", test_examples))
q1.append(eval_method(reflection,       "Reflection / Revision",         test_examples))

print("\n" + "=" * 55)
print("  Q2: Tool-Using Agent")
print("=" * 55, flush=True)
q2 = eval_agent(test_examples)

# ── 10. Summary Table ─────────────────────────────────────────────────────────
print("=" * 72)
print("  FINAL RESULTS")
print("=" * 72)
print(f"{'Method':<34} {'Accuracy':>10} {'Tokens':>10} {'Calls':>8}")
print("-" * 72)
for r in q1:
    print(f"{r['method']:<34} {r['accuracy']:>9.1%} {r['total_tokens']:>10,} {r['total_calls']:>8,}")
r = q2
print(f"{'Tool-Using Agent':<34} {r['accuracy']:>9.1%} {r['total_tokens']:>10,} {r['total_model_calls']:>8,}")
print("="*72)
print(f"\nTotal time: {(time.time()-t0)/60:.1f} min", flush=True)

# ── 11. Save JSON ─────────────────────────────────────────────────────────────
out_path = r"c:\Users\Htc\Desktop\DSAI-AGENTIC AI\assignment1_results.json"
payload  = {
    "q1_summary": [{k:v for k,v in r.items() if k!="results"} for r in q1],
    "q2_summary": {k:v for k,v in q2.items() if k!="results"},
    "q1_detail":  [r["results"] for r in q1],
    "q2_detail":  q2["results"],
}
with open(out_path,"w",encoding="utf-8") as f:
    json.dump(payload, f, indent=2, default=str)
print(f"Results saved → {out_path}", flush=True)
