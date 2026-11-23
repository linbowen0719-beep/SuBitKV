"""Generate exactly one video from existing models and offline assets."""

import argparse
import fcntl
import json
import re
from pathlib import Path

from subitkv.baselines import BASELINE_QUANT_TYPES
from . import adapters
from .common import MODELS, REPO, digest, identity, load_config, write_json
from .experiment import validate_calibration
from .workflow import Run, require_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="wan2_2_5b")
    parser.add_argument("--config", type=Path, default=REPO / "configs/subitkv.yaml")
    parser.add_argument("--run-id", default="moviegen-main", help="offline calibration experiment ID")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--method", choices=("subitkv", "bf16", *BASELINE_QUANT_TYPES), default="subitkv")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--image", type=Path, help="required conditioning image for HY-WorldPlay")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
            raise ValueError("run-id must be a plain name")
        prompt = args.prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        cfg = load_config(args.config, args.model, "full")
        source = Run(cfg, args.run_id + "-calibration", True)
        budget = 0.5 if args.budget is None else args.budget
        if args.method == "subitkv":
            if budget not in cfg["budgets"]:
                raise ValueError("budget must be declared in config.budgets")
            selected = budget
        else:
            if args.budget is not None:
                raise ValueError("--budget belongs to --method subitkv")
            selected = None if args.method == "bf16" else args.method
        if args.seed is not None and args.seed < 0:
            raise ValueError("seed must be nonnegative")
        if args.model == "hyworldplay_8b" and args.image is None:
            raise ValueError("HY-WorldPlay single inference requires --image")
        if args.image is not None and args.model != "hyworldplay_8b":
            raise ValueError("--image belongs to HY-WorldPlay")
        image = args.image.expanduser().resolve() if args.image else None
        request = dict(model=args.model, prompt=prompt, method=args.method,
                       budget=budget if args.method == "subitkv" else None,
                       seed=args.seed if args.seed is not None else cfg["model_config"]["seed"],
                       image=str(image) if image else None,
                       image_sha256=digest(image) if image and image.is_file() else None)
        token = identity(request)[:16]
        run = Run(cfg, args.run_id + "-single-" + token, True)
        output = args.output_dir.expanduser().resolve() if args.output_dir else run.root / "video"
        print(json.dumps(dict(mode="execute" if args.execute else "dry-run", videos=1,
                              request=request, models=str(source.models),
                              bundle=str(source.bundle) if args.method == "subitkv" else None,
                              output=str(output), performs_calibration=False), indent=2))
        if not args.execute:
            return
        if image and not image.is_file():
            raise FileNotFoundError(f"Conditioning image missing: {image}")
        validate_calibration(source, need_bundle=args.method == "subitkv")
        require_runtime()
        cfg["model_config"]["seed"] = request["seed"]
        cfg["workload"].pop("eval_set", None)
        if image:
            cfg["data"]["single_conditioning_image"] = str(image)
        run.models, run.bundle = source.models, source.bundle
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_json(output / "request.json", request)
            existing = output / "record.json"
            if existing.exists():
                record = json.loads(existing.read_text())
                if digest(Path(record["video"])) != record["video_sha256"]:
                    raise ValueError("Existing single inference video was modified")
            else:
                attempt = adapters.attempt(output / "attempts")
                attempt.mkdir(parents=True, exist_ok=True)
                record = adapters.generate(run, 0, prompt, "custom", attempt, budget=selected)
                write_json(existing, record)
            print("Video:", record["video"])
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"infer: {exc}\n")
