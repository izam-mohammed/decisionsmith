"""`decisionsmith finetune | bench | status | export | doctor | mcp`. Every command takes `--json`.

Exit codes: 0 ok · 1 error · 2 done but not ready (no-go / advice pending) · 3 invalid input · 4 engine unavailable.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from typing import Any

from .engines.base import EngineError

OK, ERROR, NOT_READY, INVALID, ENGINE = 0, 1, 2, 3, 4


def load_schema(spec: str) -> Any:
    """`app.py:Ticket` or `package.module:Ticket`."""
    path, sep, name = spec.rpartition(":")
    if not sep or not path or not name:
        raise ValueError("--schema must look like app.py:Ticket or package.module:Ticket, got %r" % spec)
    if path.endswith(".py") or os.path.sep in path:
        if not os.path.exists(path):
            raise ValueError("schema file not found: %s" % path)
        folder = os.path.dirname(os.path.abspath(path))
        if folder not in sys.path:
            sys.path.insert(0, folder)
        mod_name = "_ds_schema_%s" % os.path.splitext(os.path.basename(path))[0]
        spec_obj = importlib.util.spec_from_file_location(mod_name, path)
        assert spec_obj is not None and spec_obj.loader is not None
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[mod_name] = module
        spec_obj.loader.exec_module(module)
    else:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        module = importlib.import_module(path)
    try:
        return getattr(module, name)
    except AttributeError:
        raise ValueError("%s has no %r" % (path, name)) from None


def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(text)


def _given(args: argparse.Namespace, *keys: str) -> dict[str, Any]:
    return {k: getattr(args, k) for k in keys if getattr(args, k) is not None}


def open_log(path: str) -> Any:
    from .log import Log

    if not os.path.exists(path):
        raise ValueError("no log at %s" % path)
    return Log(path)


def _finetune(args: argparse.Namespace) -> int:
    from .training.finetuning import finetune

    schema = load_schema(args.schema) if args.schema else None
    options = _given(args, "epochs", "batch", "accum", "lr", "loss", "seed", "device", "max_steps")
    report = finetune(
        args.data,
        schema,
        base=args.base,
        out=args.out,
        train=args.train,
        group_by=args.group_by,
        resume=args.resume,
        verbose=not args.json,
        **options,
    )
    _emit(args, report.to_dict(), str(report))
    return OK if report.go or report.go is None else NOT_READY


def _model(args: argparse.Namespace) -> Any:
    from .predictor import Model

    if bool(args.labels) == bool(args.schema):
        raise ValueError("give either --labels a,b,c or --schema app.py:Model")
    spec = [x.strip() for x in args.labels.split(",") if x.strip()] if args.labels else load_schema(args.schema)
    return Model(
        spec, getattr(args, "base", "laya") or "laya", question=args.question, device=getattr(args, "device", None)
    )


def _teacher(args: argparse.Namespace) -> Any:
    if not args.teacher:
        return None
    if args.teacher_url:
        from .engines import LLMEngine

        return LLMEngine(args.teacher, url=args.teacher_url, api_key=os.environ.get("DECISIONSMITH_TEACHER_KEY"))
    return args.teacher


def read_texts(path: str) -> list[str]:
    """Plain texts from a .txt (one per line), a CSV `text` column, or JSONL `text` fields."""
    if not os.path.exists(path):
        raise ValueError("no such file: %s" % path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        if path.endswith(".csv"):
            import csv

            rows = list(csv.DictReader(f))
            if rows and "text" not in rows[0]:
                raise ValueError("%s needs a 'text' column" % path)
            return [r["text"] for r in rows if r.get("text", "").strip()]
        if path.endswith(".jsonl"):
            return [json.loads(line)["text"] for line in f if line.strip()]
        return [line.strip() for line in f if line.strip()]


def _train(args: argparse.Namespace) -> int:
    m = _model(args)
    teacher = _teacher(args)
    options = _given(args, "epochs", "lr", "seed", "max_steps")
    data = read_texts(args.data) if args.data and teacher is not None else args.data
    report = m.train(
        data,
        teacher=teacher,
        generate=args.generate,
        about=args.about,
        out=args.out,
        verbose=not args.json,
        train=args.train,
        **options,
    )
    if args.json:
        _emit(args, {**report.to_dict(), "using": m.trained}, "")
    return OK if m.trained else NOT_READY


def _generate(args: argparse.Namespace) -> int:
    from .predictor import write_rows

    m = _model(args)
    teacher = _teacher(args)
    if teacher is None:
        raise ValueError("generate needs --teacher, e.g. --teacher claude-haiku-4-5")
    rows = m.generate(args.n, teacher, about=args.about, verbose=not args.json)
    path = write_rows(rows, args.out, m.schema)
    _emit(
        args,
        {"rows": len(rows), "path": os.path.abspath(path)},
        "wrote %d examples to %s; review them, then: decisionsmith train %s %s"
        % (len(rows), path, path, "--labels " + args.labels if args.labels else "--schema " + args.schema),
    )
    return OK if rows else NOT_READY


def _bench(args: argparse.Namespace) -> int:
    from .benchmark import bench

    report = bench(
        load_schema(args.schema), args.data, args.engines, threshold=args.threshold, limit=args.limit, out=args.out
    )
    _emit(args, report.to_dict(), str(report))
    return OK


def status_of(schema: Any, log_path: str, student: str | None, default: float) -> Any:
    from .status import compute_status
    from .training.adapt import adapt_key, threshold_for

    with open_log(log_path) as log:
        rows = log.rows(schema.name)
        student = student or next((r["student"] for r in reversed(rows) if r["student"]), None)
        calib = log.get(adapt_key(schema.name, student), {}) or {}
    modes = dict.fromkeys(schema.fields, "-")
    return compute_status(schema, rows, modes, student, lambda n: threshold_for(calib, n, default)), len(rows)


def _status_dict(schema: Any, log_path: str, student: str | None, default: float) -> dict[str, Any]:
    status, n = status_of(schema, log_path, student, default)
    return {"decisions": n, **status.to_dict()}


def _status(args: argparse.Namespace) -> int:
    from .schema import compile_schema

    status, n = status_of(compile_schema(load_schema(args.schema)), args.log, args.student, args.threshold)
    _emit(
        args,
        {"decisions": n, **status.to_dict()},
        "%d decisions · student %s\n%s" % (n, status.student or "-", status),
    )
    ready = all(f.advice.startswith(("ready", "in student", "cascade is right")) for f in status.fields.values())
    return OK if ready else NOT_READY


def _export(args: argparse.Namespace) -> int:
    from .schema import compile_schema
    from .training.export import export

    with open_log(args.log) as log:
        n = export(compile_schema(load_schema(args.schema)), log, args.out, args.format)
    _emit(
        args,
        {"rows": n, "path": os.path.abspath(args.out), "format": args.format},
        "wrote %d rows to %s" % (n, args.out),
    )
    return OK


def _doctor(args: argparse.Namespace) -> int:
    from importlib.metadata import PackageNotFoundError, version

    from .engines import from_string, response

    checks: list[dict[str, Any]] = [
        {"check": "python", "ok": sys.version_info >= (3, 10), "detail": platform.python_version()}
    ]
    packages = {
        "decisionsmith": "",
        "laya": "laya",
        "torch": "laya",
        "anthropic": "anthropic",
        "mcp": "mcp",
    }
    for pkg, extra in packages.items():
        try:
            checks.append({"check": pkg, "ok": True, "detail": version(pkg)})
        except PackageNotFoundError:
            fix = "pip install 'decisionsmith[%s]'" % extra if extra else ""
            checks.append({"check": pkg, "ok": not extra, "detail": "not installed", "fix": fix})
    try:
        from .training.train import device

        checks.append({"check": "device", "ok": True, "detail": str(device())})
    except ImportError:
        pass
    for key in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "SYSTEMONE_API_KEY"):
        checks.append({"check": key, "ok": True, "detail": "set" if os.environ.get(key) else "not set"})
    question = {"ok": {"type": "noul", "instructions": "Is this a connectivity test?"}}
    for spec in [e.strip() for e in (args.engines or "").split(",") if e.strip()]:
        t0 = time.perf_counter()
        try:
            engine = from_string(spec)
            response(engine, engine.ask("This is a connectivity test.", question), question)
            ms = (time.perf_counter() - t0) * 1000
            checks.append({"check": "engine %s" % spec, "ok": True, "detail": "%.0f ms" % ms})
        except Exception as e:
            fix = e.fix if isinstance(e, EngineError) else ""
            checks.append({"check": "engine %s" % spec, "ok": False, "detail": str(e).splitlines()[0], "fix": fix})
    lines = [
        "%s %-20s %s%s"
        % ("ok " if c["ok"] else "!! ", c["check"], c["detail"], ("  -> " + c["fix"]) if c.get("fix") else "")
        for c in checks
    ]
    _emit(args, {"checks": checks}, "\n".join(lines))
    failed = [c for c in checks if not c["ok"]]
    return ENGINE if any(c["check"].startswith("engine ") for c in failed) else (ERROR if failed else OK)


def _mcp(args: argparse.Namespace) -> int:
    from .mcp import serve

    serve()
    return OK


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="decisionsmith", description="Use and fine-tune System One models (Jev, Laya).")
    p.add_argument("--version", action="version", version="decisionsmith 0.1.0")
    sub = p.add_subparsers(dest="command", required=True)

    def cmd(name: str, help: str) -> argparse.ArgumentParser:
        c = sub.add_parser(name, help=help, description=help)
        c.add_argument("--json", action="store_true", help="machine-readable output")
        return c

    def numbers(c: argparse.ArgumentParser, *names: str) -> None:
        for name in names:
            c.add_argument("--" + name, dest=name.replace("-", "_"), type=float if name == "lr" else int)

    def train_mode(c: argparse.ArgumentParser) -> None:
        c.add_argument("--train", default="auto", choices=["auto", "head", "full"])

    def log_args(c: argparse.ArgumentParser) -> None:
        c.add_argument("--schema", required=True)
        c.add_argument("--log", default="decisions.db")

    f = cmd("finetune", "fine-tune Laya on labelled data; writes a checkpoint that loads in laya.load()")
    f.add_argument("data", help="CSV (text + one column per field), JSONL, or typed-decisions JSONL")
    f.add_argument("--schema", help="app.py:Ticket (not needed for typed-decisions data)")
    f.add_argument("--base", default="laya", help="laya | laya:multilingual style name, or a checkpoint dir")
    f.add_argument("--out", default="runs/v1")
    train_mode(f)
    f.add_argument("--group-by", dest="group_by")
    f.add_argument("--resume", action="store_true")
    numbers(f, "epochs", "batch", "accum", "lr", "seed", "max-steps")
    f.add_argument("--loss", choices=["ce", "proper", "rlcd"])
    f.add_argument("--device")
    f.set_defaults(run=_finetune)

    def model_args(c: argparse.ArgumentParser) -> None:
        c.add_argument("--labels", help="comma list, e.g. billing,technical,sales (one answer per text)")
        c.add_argument("--schema", help="app.py:Ticket (several answers per text)")
        c.add_argument("--question", help="with --labels: the question to ask (default: which label fits)")
        c.add_argument("--teacher", help="LLM that labels or writes data, e.g. claude-haiku-4-5, jev, ollama/qwen3")
        c.add_argument(
            "--teacher-url",
            dest="teacher_url",
            help="OpenAI-compatible server for --teacher (key from DECISIONSMITH_TEACHER_KEY)",
        )
        c.add_argument("--about", help="what the texts are, e.g. 'support emails for a SaaS app'")

    t = cmd("train", "train Laya: on labelled data, on texts a teacher labels, or on data a teacher writes")
    t.add_argument("data", nargs="?", help="labelled CSV/JSONL; or texts (.txt/.csv/.jsonl) with --teacher")
    model_args(t)
    t.add_argument("--generate", type=int, help="no data: have the teacher write this many examples first")
    t.add_argument("--base", default="laya")
    t.add_argument("--out")
    train_mode(t)
    numbers(t, "epochs", "lr", "seed", "max-steps")
    t.add_argument("--device")
    t.set_defaults(run=_train)

    g = cmd("generate", "have a teacher LLM write a balanced, re-checked golden dataset")
    model_args(g)
    g.add_argument("-n", type=int, default=300, help="how many examples to write (default 300)")
    g.add_argument("--out", default="golden.csv", help=".csv (easy to review) or .jsonl")
    g.set_defaults(run=_generate)

    b = cmd("bench", "compare engines on labelled data")
    b.add_argument("data")
    b.add_argument("--schema", required=True)
    b.add_argument("--engines", required=True, help="comma list, e.g. claude-sonnet-5,jev,laya,laya:./runs/v1")
    b.add_argument("--threshold", type=float, default=0.8)
    b.add_argument("--limit", type=int)
    b.add_argument("--out", help="write the report (.json or .html)")
    b.set_defaults(run=_bench)

    s = cmd("status", "where each field stands, from a harness log")
    log_args(s)
    s.add_argument("--student", help="engine name; default: the most recent student in the log")
    s.add_argument("--threshold", type=float, default=0.8)
    s.set_defaults(run=_status)

    e = cmd("export", "write labelled decisions from a harness log as training JSONL")
    log_args(e)
    e.add_argument("--out", default="train.jsonl")
    e.add_argument("--format", default="answers", choices=["answers", "typed-decisions"])
    e.set_defaults(run=_export)

    d = cmd("doctor", "check the install, device, keys and (optionally) engines")
    d.add_argument("--engines", help="comma list to test with one tiny request each (may cost a fraction of a cent)")
    d.set_defaults(run=_doctor)

    m = sub.add_parser("mcp", help="run the MCP server on stdio")
    m.set_defaults(run=_mcp, json=False)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.run(args))
    except EngineError as e:
        _fail(args, "engine", str(e), e.fix)
        return ENGINE
    except (ValueError, TypeError, KeyError, FileNotFoundError) as e:
        _fail(args, "invalid", str(e), "")
        return INVALID
    except KeyboardInterrupt:
        return ERROR
    except Exception as e:
        _fail(args, "error", "%s: %s" % (type(e).__name__, e), "")
        return ERROR


def _fail(args: argparse.Namespace, code: str, message: str, fix: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps({"error": {"code": code, "message": message, "fix": fix}}))
    else:
        print("error: %s" % message, file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
