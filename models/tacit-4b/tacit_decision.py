"""Tacit decisions: one forward pass per decision, optionally escalating the least confident ones to
the model's own reasoning. Runs on transformers (default) or vLLM.

    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    Tacit = get_class_from_dynamic_module("tacit_decision.Tacit", "morriszjm/Tacit-9B")
    tacit = Tacit.from_pretrained("morriszjm/Tacit-9B", adaptive=True, tau=0.5, max_cot_share=0.2)
    # tacit = Tacit.from_pretrained("morriszjm/Tacit-9B", engine="vllm", adaptive=True)
    d = tacit.decide(state="Order #4471 ...", question="What does the customer want?",
                     options=["track_order", "cancel_order", "refund"])
    d["answer"], d["probs"], d["route"]          # "cancel_order", {...}, "one_forward" or "cot"

One forward: the prompt Tacit was trained on, the next-token distribution read over the option labels
(A, B, C... / Yes, No / 1, 2, 3...) and renormalised over them. No tokens are generated.

Adaptive (`adaptive=True`): a decision whose margin (log-probability gap between its top two
options) is below `tau` is asked again with the model's thinking on; the reasoning ends, "Answer:"
is appended, and the answer is read the same way, so an escalated decision also comes with
probabilities and never fails to parse. `max_cot_share` caps the share of the last `cot_window`
decisions that may be escalated (rounded up), so traffic cannot drift into reasoning, also on a model
that serves for days; `cot_window=None` counts every decision since loading, `max_cot_share=None`
removes the cap.

vLLM (`engine="vllm"`): the labels are read from the top `logprobs` (default 20) log-probabilities of
the full vocabulary, a label outside them counting as probability 0; escalated decisions of one
batch reason in parallel. `vllm_kwargs` go to `vllm.LLM` (e.g. gpu_memory_utilization,
max_model_len, tensor_parallel_size). Qwen3.5-based checkpoints on Hopper need nvcc for vLLM's
default gated-delta-rule kernel; without it (`native_gdn=None` detects this) the Triton kernel is
used and the engine runs in this process.

A vLLM server (`engine="server"`): `vllm serve morriszjm/Tacit-9B`, then
`Tacit.from_pretrained("morriszjm/Tacit-9B", engine="server", base_url="http://127.0.0.1:8000/v1")`.
Only the tokenizer loads on the client; vLLM is not needed there. Or put an HTTP gateway in front of
the server, so every client shares one cap:

    python tacit_decision.py serve --model morriszjm/Tacit-9B --upstream http://127.0.0.1:8000/v1 --adaptive
    curl -s 127.0.0.1:8100/v1/decide -d '{"state": "...", "question": "...", "options": ["a", "b"]}'
"""
from __future__ import annotations

import collections
import importlib
import importlib.util
import json
import math
import os
import shutil
import string
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

LETTERS = string.ascii_uppercase
ONE_FORWARD_SYSTEM = ("You are a decision function. You will be given a state and one question. "
                      "Reply with the answer label only: no words, no punctuation, no explanation.")
COT_SYSTEM = "You are a careful assistant that answers questions about a given context."
KINDS = {"choice": "choice", "yes_no": "yes_no", "noul": "yes_no", "score": "score"}


def _labels(kind: str, k: int) -> List[str]:
    if kind == "yes_no":
        return ["Yes", "No"]
    if kind == "score":
        return [str(i + 1) for i in range(k)]
    return list(LETTERS[:k])


def _one_forward_user(state: str, question: str, kind: str, options: Sequence[str], labels: Sequence[str]) -> str:
    prefix = ["State:", state if state else "(empty)", "", "Question: %s" % question]
    suffix = []
    if kind == "yes_no":
        suffix.append("Answer Yes or No.")
    elif kind == "score":
        prefix.append("Pick the level that applies (the levels are ordered):")
        suffix += ["%s. %s" % (labels[j], o) for j, o in enumerate(options)]
        suffix.append("Answer with the number only.")
    else:
        prefix.append("Options:")
        suffix += ["%s. %s" % (labels[j], o) for j, o in enumerate(options)]
        suffix.append("Answer with the letter only.")
    return "\n".join(prefix) + "\n" + "\n".join(suffix)


def _cot_user(state: str, question: str, kind: str, options: Sequence[str], labels: Sequence[str]) -> str:
    parts = [state.strip(), "", "Question: " + question.strip(), ""]
    if kind == "yes_no":
        parts.append("Answer Yes or No.")
        form = "Answer: Yes  or  Answer: No"
    elif kind == "score":
        parts += ["Levels:"] + ["%s. %s" % (labels[j], o) for j, o in enumerate(options)]
        form = "Answer: <level number>"
    else:
        parts += ["Options:"] + ["%s. %s" % (labels[j], o) for j, o in enumerate(options)]
        form = "Answer: <option letter>"
    parts += ["", "Think through the question, then give your final answer on the last line in "
                  "exactly this form:", form]
    return "\n".join(parts)


def _restricted(values: Sequence[float]) -> torch.Tensor:
    """Log-softmax over the labels only (a label with no value counts as probability 0)."""
    t = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(t).any():                      # no label among the returned log-probabilities
        t = torch.zeros_like(t)
    return torch.log_softmax(t, dim=-1).float()


# ---------------------------------------------------------------------- engines
class _TransformersEngine:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok

    @torch.no_grad()
    def read(self, texts: List[str], label_ids: List[List[int]]) -> List[torch.Tensor]:
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        try:                                             # only the last position is read
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        except TypeError:
            logits = self.model(**enc).logits[:, -1, :].float()
        return [torch.log_softmax(logits[b, ids], dim=-1).cpu() for b, ids in enumerate(label_ids)]

    @torch.no_grad()
    def reason_read(self, prompts: List[str], label_ids: List[List[int]], max_tokens: int,
                    sampling: Dict[str, Any]) -> List[Tuple[torch.Tensor, int]]:
        out = []
        for prompt, ids in zip(prompts, label_ids):
            enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            gen = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=True, stop_strings=["</think>"],
                                      tokenizer=self.tok, pad_token_id=self.tok.pad_token_id, **sampling)
            new = gen[0, enc["input_ids"].shape[1]:]
            thought = self.tok.decode(new, skip_special_tokens=False)
            for eos in (self.tok.eos_token or "", "<|im_end|>", "<|endoftext|>"):
                if eos and thought.endswith(eos):
                    thought = thought[: -len(eos)]
            if "</think>" in thought:
                thought = thought[: thought.index("</think>") + len("</think>")]
            else:                                        # out of budget: close the reasoning here
                thought = thought + "\n</think>"
            out.append((self.read([prompt + thought + "\n\nAnswer:"], [ids])[0], int(new.shape[0])))
        return out


class _VLLMEngine:
    def __init__(self, llm, tok, logprobs: int = 20):
        self.llm, self.tok, self.k = llm, tok, logprobs

    def _ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def _read_ids(self, prompts_ids: List[List[int]], label_ids: List[List[int]]) -> List[torch.Tensor]:
        SamplingParams = importlib.import_module("vllm").SamplingParams
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=self.k)
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts_ids], sp, use_tqdm=False)
        res = []
        for o, ids in zip(outs, label_ids):
            lp = o.outputs[0].logprobs[0]
            res.append(_restricted([lp[t].logprob if t in lp else -math.inf for t in ids]))
        return res

    def read(self, texts: List[str], label_ids: List[List[int]]) -> List[torch.Tensor]:
        return self._read_ids([self._ids(t) for t in texts], label_ids)

    def reason_read(self, prompts: List[str], label_ids: List[List[int]], max_tokens: int,
                    sampling: Dict[str, Any]) -> List[Tuple[torch.Tensor, int]]:
        SamplingParams = importlib.import_module("vllm").SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, stop=["</think>"], include_stop_str_in_output=True, **sampling)
        pids = [self._ids(p) for p in prompts]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in pids], sp, use_tqdm=False)
        tail, close = self._ids("\n\nAnswer:"), self._ids("\n</think>")
        reads, ntok = [], []
        for p, o in zip(pids, outs):
            gen = list(o.outputs[0].token_ids)
            closed = "</think>" in o.outputs[0].text
            reads.append(p + gen + ([] if closed else close) + tail)    # out of budget: close it here
            ntok.append(len(gen))
        return list(zip(self._read_ids(reads, label_ids), ntok))


class _ServerEngine:
    """A running `vllm serve` (OpenAI-compatible /v1/completions). Prompts go as token ids, the label
    log-probabilities come back keyed by token id (`return_tokens_as_token_ids`), and one request
    carries a whole batch, which the server schedules in parallel."""

    def __init__(self, base_url: str, model: str, tok, api_key: Optional[str] = None, logprobs: int = 20,
                 timeout: float = 3600.0):
        self.url = base_url.rstrip("/") + "/completions"
        self.model, self.tok, self.k, self.timeout = model, tok, logprobs, timeout
        self.api_key = api_key or os.environ.get("TACIT_API_KEY") or os.environ.get("OPENAI_API_KEY")

    def _ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def _complete(self, prompts_ids: List[List[int]], **params) -> List[Dict[str, Any]]:
        body = json.dumps(dict(model=self.model, prompt=prompts_ids, **params)).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                choices = json.loads(r.read())["choices"]
        except urllib.error.HTTPError as e:
            raise RuntimeError("vLLM server %s: %s %s" % (self.url, e.code, e.read()[:500])) from None
        return sorted(choices, key=lambda c: c["index"])

    def _read_ids(self, prompts_ids: List[List[int]], label_ids: List[List[int]]) -> List[torch.Tensor]:
        choices = self._complete(prompts_ids, max_tokens=1, temperature=0.0, logprobs=self.k,
                                 return_tokens_as_token_ids=True)
        res = []
        for c, ids in zip(choices, label_ids):
            top = c["logprobs"]["top_logprobs"][0] or {}
            lp = {int(k.split(":", 1)[1]): v for k, v in top.items() if k.startswith("token_id:")}
            res.append(_restricted([lp.get(t, -math.inf) for t in ids]))
        return res

    def read(self, texts: List[str], label_ids: List[List[int]]) -> List[torch.Tensor]:
        return self._read_ids([self._ids(t) for t in texts], label_ids)

    def reason_read(self, prompts: List[str], label_ids: List[List[int]], max_tokens: int,
                    sampling: Dict[str, Any]) -> List[Tuple[torch.Tensor, int]]:
        choices = self._complete([self._ids(p) for p in prompts], max_tokens=max_tokens, stop=["</think>"],
                                 include_stop_str_in_output=True, skip_special_tokens=False, **sampling)
        texts, ntok = [], []
        for p, c in zip(prompts, choices):
            thought = c["text"]
            if "</think>" in thought:
                thought = thought[: thought.index("</think>") + len("</think>")]
            else:                                        # out of budget: close the reasoning here
                thought = thought + "\n</think>"
            texts.append(p + thought + "\n\nAnswer:")
            ntok.append(len(self._ids(c["text"])))
        return list(zip(self.read(texts, label_ids), ntok))


def _needs_native_gdn(repo: str) -> bool:
    """Qwen3.5 (gated delta rule) on a machine where vLLM's default kernel cannot be compiled."""
    try:                                                 # read config.json itself: older transformers do not know qwen3_5
        if os.path.isdir(repo):
            path = os.path.join(repo, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo, "config.json")
        cfg = json.load(open(path))
        mt = str(cfg.get("model_type", "")) + str((cfg.get("text_config") or {}).get("model_type", ""))
    except Exception:
        return False
    has_nvcc = shutil.which("nvcc") or os.path.exists(os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"))
    return "qwen3_5" in mt and not has_nvcc


def _patch_native_gdn():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # the patch must reach the engine: run it here
    ChunkGatedDeltaRule = importlib.import_module("vllm.model_executor.models.qwen3_next").ChunkGatedDeltaRule
    if getattr(ChunkGatedDeltaRule, "_tacit_native", False):
        return
    init = ChunkGatedDeltaRule.__init__

    def native_init(self, *a, **kw):
        init(self, *a, **kw)
        self._forward_method = self.forward_native
    ChunkGatedDeltaRule.__init__ = native_init
    ChunkGatedDeltaRule._tacit_native = True


# ---------------------------------------------------------------------- the decision API
class Tacit:
    def __init__(self, engine, tokenizer, *, adaptive: bool = False, tau: float = 0.5,
                 max_cot_share: Optional[float] = 0.2, cot_window: Optional[int] = 1000, cot_max_tokens: int = 8192,
                 temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20, batch_size: int = 8):
        self.engine, self.tok = engine, tokenizer
        self.adaptive, self.tau, self.max_cot_share = adaptive, tau, max_cot_share
        self.cot_window = cot_window
        self._recent = collections.deque(maxlen=cot_window) if cot_window else None
        self.cot_max_tokens, self.batch_size = cot_max_tokens, batch_size
        self.sampling = dict(temperature=temperature, top_p=top_p, top_k=top_k)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.decisions = 0
        self.escalated = 0
        self._lock = threading.Lock()                    # the cap's bookkeeping, shared by concurrent callers

    @classmethod
    def from_pretrained(cls, repo: str, *, engine: str = "transformers", dtype=torch.bfloat16, device_map="auto",
                        vllm_kwargs: Optional[Dict[str, Any]] = None, logprobs: int = 20,
                        native_gdn: Optional[bool] = None, base_url: Optional[str] = None,
                        api_key: Optional[str] = None, served_model: Optional[str] = None, **kwargs) -> "Tacit":
        if engine == "server":
            # a running `vllm serve <repo>`; only the tokenizer (prompt and label tokens) is loaded here
            from transformers import AutoTokenizer
            if not base_url:
                raise ValueError('engine="server" needs base_url, e.g. "http://127.0.0.1:8000/v1"')
            tok = AutoTokenizer.from_pretrained(repo)
            kwargs.setdefault("batch_size", 1 << 30)
            return cls(_ServerEngine(base_url, served_model or repo, tok, api_key, logprobs), tok, **kwargs)
        if engine == "vllm":
            # the engine runs in this process unless the caller says otherwise: a spawned engine
            # re-imports the calling script, which breaks scripts without a __main__ guard and notebooks
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            if native_gdn or (native_gdn is None and _needs_native_gdn(repo)):
                _patch_native_gdn()
            LLM = importlib.import_module("vllm").LLM
            vk = dict(dtype="bfloat16", max_model_len=32768, max_logprobs=max(20, logprobs))
            vk.update(vllm_kwargs or {})
            llm = LLM(model=repo, **vk)
            tok = llm.get_tokenizer()
            # vLLM batches internally; send everything at once
            kwargs.setdefault("batch_size", 1 << 30)
            return cls(_VLLMEngine(llm, tok, logprobs), tok, **kwargs)
        if engine != "transformers":
            raise ValueError('engine must be "transformers", "vllm" or "server"')
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo)
        if device_map is not None and importlib.util.find_spec("accelerate") is None:
            device_map = None                            # device_map needs accelerate; place the model by hand
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, device_map=device_map)
        if device_map is None and torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return cls(_TransformersEngine(model, tok), tok, **kwargs)

    # ------------------------------------------------------------------ public API
    def decide(self, state: str, question: str, options: Optional[Sequence[str]] = None,
               kind: str = "choice") -> Dict[str, Any]:
        """One decision. kind: "choice" (pick one of `options`), "yes_no", or "score" (`options` are
        ordered levels, lowest first)."""
        return self.decide_batch([dict(state=state, question=question, options=options, kind=kind)])[0]

    def decide_batch(self, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Each item: dict(state=..., question=..., options=[...], kind=...). Within a batch the least
        confident decisions are escalated first."""
        norm = [self._normalise(it) for it in items]
        out = []
        for i in range(0, len(norm), self.batch_size):
            chunk = norm[i:i + self.batch_size]
            texts = [self._render(ONE_FORWARD_SYSTEM, _one_forward_user(it["state"], it["question"], it["kind"],
                                                                        it["options"], it["labels"]), False)
                     for it in chunk]
            logps = self.engine.read(texts, [self._label_ids(it["labels"], spaced_first=False) for it in chunk])
            out += [self._result(it, lp, "one_forward") for it, lp in zip(chunk, logps)]
        # which decisions escalate is settled and booked under the lock; the reasoning itself runs
        # outside it, so concurrent callers (a gateway's threads) share one cap without waiting on each other
        with self._lock:
            chosen = []
            if self.adaptive:
                want = sorted((i for i, d in enumerate(out) if d["margin"] < self.tau), key=lambda i: out[i]["margin"])
                chosen = want[:self._budget(len(out), len(want))]
                self.escalated += len(chosen)
            self.decisions += len(out)
            if self._recent is not None:
                picked = set(chosen)
                self._recent.extend(i in picked for i in range(len(out)))
        if chosen:
            prompts = [self._render(COT_SYSTEM, _cot_user(norm[i]["state"], norm[i]["question"], norm[i]["kind"],
                                                          norm[i]["options"], norm[i]["labels"]), True) for i in chosen]
            reads = self.engine.reason_read(prompts, [self._label_ids(norm[i]["labels"], spaced_first=True)
                                                      for i in chosen], self.cot_max_tokens, self.sampling)
            for i, (lp, n) in zip(chosen, reads):
                out[i] = self._result(norm[i], lp, "cot", first_pass=out[i], reasoning_tokens=n)
        return out

    def _budget(self, n_new: int, n_want: int) -> int:
        """How many of `n_new` new decisions may be escalated.

        With `cot_window`, escalations among the last `cot_window` decisions (these included) stay at
        or below `max_cot_share` of them, rounded up: unused budget does not pile up over a long run,
        and a burst of hard traffic gets the same share as any other stretch. Without it, the share
        counts every decision since loading."""
        if self.max_cot_share is None:
            return n_want
        if self._recent is None:
            return max(0, math.ceil(self.max_cot_share * (self.decisions + n_new) - 1e-9) - self.escalated)
        W = self._recent.maxlen
        if n_new >= W:                                   # the batch alone fills the window
            return math.ceil(self.max_cot_share * n_new - 1e-9)
        old = list(self._recent)[max(0, len(self._recent) + n_new - W):]   # what stays in the window
        size = len(old) + n_new
        return max(0, math.ceil(self.max_cot_share * size - 1e-9) - sum(old))

    @property
    def stats(self) -> Dict[str, Any]:
        s = {"decisions": self.decisions, "escalated": self.escalated,
             "cot_share": self.escalated / self.decisions if self.decisions else 0.0}
        if self._recent is not None:
            s["window_cot_share"] = sum(self._recent) / len(self._recent) if self._recent else 0.0
        return s

    # ------------------------------------------------------------------ internals
    def _normalise(self, it):
        kind = KINDS.get(it.get("kind", "choice"))
        if kind is None:
            raise ValueError("kind must be one of %s" % sorted(KINDS))
        options = ["Yes", "No"] if kind == "yes_no" else [str(o) for o in (it.get("options") or [])]
        if len(options) < 2 or len(set(options)) != len(options):
            raise ValueError("need at least two distinct options")
        if kind != "yes_no" and len(options) > len(LETTERS):
            raise ValueError("at most %d options" % len(LETTERS))
        return dict(state=str(it.get("state", "")), question=str(it["question"]), kind=kind,
                    options=options, labels=_labels(kind, len(options)))

    def _render(self, system: str, user: str, thinking: bool) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=thinking)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if text.endswith("<|start|>assistant"):          # harmony-format models: read the final channel
            text += "<|channel|>final<|message|>"
        return text

    def _label_ids(self, labels, spaced_first: bool):
        ids = []
        for lab in labels:
            cands = (" " + lab, lab) if spaced_first else (lab, " " + lab)
            for c in cands:
                t = self.tok.encode(c, add_special_tokens=False)
                if len(t) == 1:
                    ids.append(t[0])
                    break
            else:
                raise ValueError("label %r is not a single token for this tokenizer" % lab)
        if len(set(ids)) != len(ids):
            raise ValueError("label tokens collide")
        return ids

    def _result(self, it, logp: torch.Tensor, route: str, **extra) -> Dict[str, Any]:
        p = logp.exp()
        top = torch.topk(logp, 2).values
        j = int(torch.argmax(logp))
        margin = float(top[0] - top[1]) if math.isfinite(float(top[1])) else float("inf")
        return dict(answer=it["options"][j], index=j, probs={o: float(p[i]) for i, o in enumerate(it["options"])},
                    margin=margin, route=route, **extra)


# ---------------------------------------------------------------------- an HTTP gateway
def serve_gateway(repo: str, upstream: str, host: str = "127.0.0.1", port: int = 8100, **tacit_kwargs):
    """POST /v1/decide in front of `vllm serve <repo>`: one shared Tacit, so every client counts
    against one cap. Body: {"state", "question", "options", "kind"} or {"items": [...]};
    GET /v1/stats; GET /health."""
    import http.server

    tacit = Tacit.from_pretrained(repo, engine="server", base_url=upstream, **tacit_kwargs)

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True})
            if self.path == "/v1/stats":
                return self._send(200, tacit.stats)
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/decide":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if "items" in req:
                    return self._send(200, {"decisions": tacit.decide_batch(req["items"])})
                return self._send(200, tacit.decide_batch([req])[0])
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            except RuntimeError as e:                    # the vLLM server failed or is unreachable
                return self._send(502, {"error": str(e)})

        def log_message(self, fmt, *args):
            pass

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    print("Tacit gateway: http://%s:%d/v1/decide -> %s (%s)" % (host, port, upstream, repo), flush=True)
    server.serve_forever()


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Tacit decisions in front of a running `vllm serve`.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="HTTP gateway: POST /v1/decide, GET /v1/stats")
    s.add_argument("--model", required=True, help="the repo vLLM serves, e.g. morriszjm/Tacit-9B")
    s.add_argument("--upstream", default="http://127.0.0.1:8000/v1", help="the vLLM server's /v1 URL")
    s.add_argument("--served-model", default=None, help="vLLM's --served-model-name, if it was set")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8100)
    s.add_argument("--adaptive", action="store_true", help="send low-confidence decisions to reasoning")
    s.add_argument("--tau", type=float, default=0.5)
    s.add_argument("--max-cot-share", type=float, default=0.2, help="a negative value removes the cap")
    s.add_argument("--cot-window", type=int, default=1000, help="0 counts every decision since start")
    s.add_argument("--cot-max-tokens", type=int, default=8192)
    a = ap.parse_args(argv)
    serve_gateway(a.model, a.upstream, a.host, a.port, served_model=a.served_model, adaptive=a.adaptive,
                  tau=a.tau, max_cot_share=None if a.max_cot_share < 0 else a.max_cot_share,
                  cot_window=a.cot_window or None, cot_max_tokens=a.cot_max_tokens)


if __name__ == "__main__":
    _main()
