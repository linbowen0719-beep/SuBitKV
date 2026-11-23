"""Model-local RVQ, residual fitting and live attention calibration."""

from __future__ import annotations

import json
import shutil

from subitkv.cache_budget import rvq_schedule_budget

from .adapters import attempt, generate, label
from .common import (
    digest,
    identity,
    save_tensor,
    seal,
    validate_schedule,
    write_json,
)
from .scheduling import solve_schedule, solve_dp
from .workflow import prompts


def train_codebooks(run):
    m = run.config["model_config"]
    work = run.root / "calibration/codebooks"
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "kv_dump").exists():
        (work / "kv_dump").symlink_to(run.capture / "kv_dump", target_is_directory=True)
    if not (work / "prompt_split.json").exists():
        shutil.copy2(run.capture / "prompt_split.json", work / "prompt_split.json")
    for tensor in ("K", "V"):
        subdir = f"nonorm_{tensor.lower()}_l16c64_codebooks"

        def train_one(tensor=tensor, subdir=subdir):
            # Existing deterministic trainer; codebooks are shared by all budgets.
            from .codebooks import train

            train(work, run.config, tensor)
            files = sorted((work / subdir).glob("layer_*_head_*.pt"))
            if len(files) != m["layers"] * m["heads"]:
                raise ValueError(f"Incomplete {tensor} codebooks: {len(files)}")
            target = run.bundle / subdir
            shutil.copytree(work / subdir, target, dirs_exist_ok=True)
            return sorted(target.glob("*.pt"))

        run.task("codebooks_" + tensor, train_one)


def teacher_pass(run, phase):
    cfg = run.config
    for i, prompt in enumerate(prompts(cfg, "train")):

        def one(i=i, prompt=prompt):
            directory = attempt(run.root / "calibration" / phase / f"prompt_{i:04d}")
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / "events"
            settings = directory / "settings.json"
            write_json(
                settings,
                dict(
                    config=cfg,
                    phase=phase,
                    prompt_id=i,
                    output=str(output),
                    bundle=str(run.bundle),
                ),
            )
            generate(
                run,
                i,
                prompt,
                "train",
                directory / "teacher",
                extra_env={"SUBITKV_CALIBRATION_SETTINGS": str(settings)},
            )
            m = cfg["model_config"]
            expected = (
                [output / f"layer_{layer}.pt" for layer in range(m["layers"])]
                if phase == "residual"
                else [
                    output / f"layer_{layer}_span_{span}.json"
                    for layer in range(m["layers"])
                    for span in range(1, m["calibration_history_steps"] + 1)
                ]
            )
            missing = [p for p in expected if not p.is_file()]
            if missing:
                raise RuntimeError(
                    f"Teacher calibration incomplete; first missing: {missing[0]}"
                )
            return expected

        run.task(f"{phase}_{i}", one)


def event_files(run, phase):
    paths = []
    for i in range(len(prompts(run.config, "train"))):
        marker = run.root / "status" / f"{phase}_{i}.json"
        paths.extend(
            run.root / name for name in json.loads(marker.read_text())["files"]
        )
    return paths


def fit_residual(run):
    import torch

    from subitkv import routed_gaussian
    from subitkv.progressive_sign_adaptive_absmax_scalar import PARAMS_SCHEMA

    from . import residual_fitting as helpers

    c, m = run.config["calibration"], run.config["model_config"]
    device = run.config["runtime"]["device"]
    files = event_files(run, "residual")
    scalars = []
    for index, path in enumerate(files):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        scalars.append(
            helpers.normalized_k_scalars(
                saved["K_centered"].to(device),
                seed=c["residual"]["codec_seed"],
                samples_per_head=c["residual"]["scalars_per_head"],
                sample_seed=c["seed"] + index,
            )
        )
    values = torch.cat(scalars).float()
    generator = torch.Generator().manual_seed(c["seed"])
    values = values[
        torch.randperm(len(values), generator=generator)[: c["residual"]["fit_scalars"]]
    ]
    tree = helpers.fit_gaussian_tree(values, c["residual"]["tree_iters"])
    base = {"K": tree, "V": tree}
    tree_params = {
        name: routed_gaussian.extend_gaussian_tree_to_depth5(base, tensor_name=name)
        for name in ("K", "V")
    }
    gpu_tree = routed_gaussian.extend_gaussian_tree_to_depth5(
        base, tensor_name="K", device=device
    )
    moments = torch.zeros(
        len(prompts(run.config, "train")), 8, 2, 16, dtype=torch.float64
    )
    sums = {
        (layer, head): {b: [0.0, 0] for b in range(1, 6)}
        for layer in range(m["layers"])
        for head in range(m["heads"])
    }
    for index, path in enumerate(files):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        layer = int(path.stem.removeprefix("layer_"))
        prompt_index = index // m["layers"]
        values = saved["K_centered"].to(device)
        for head in range(m["heads"]):
            for depth, (sse, count) in helpers.gaussian_depth_sse(
                values[head : head + 1], gpu_tree, c["residual"]["codec_seed"]
            ).items():
                sums[layer, head][depth][0] += sse
                sums[layer, head][depth][1] += count
        moments[prompt_index] += helpers.v_moments(saved["V_residual"].to(device))
    errors = {
        key: {b: s / n for b, (s, n) in value.items()} for key, value in sums.items()
    }
    candidates = []
    for fraction in c["residual"]["routing_fractions"]:
        count = round(len(errors) * fraction)
        high = set(
            sorted(errors, key=lambda k: (-(errors[k][4] - errors[k][5]), k))[:count]
        )
        low = set(
            sorted(set(errors) - high, key=lambda k: (errors[k][3] - errors[k][4], k))[
                :count
            ]
        )
        offsets = {key: 1 if key in high else -1 if key in low else 0 for key in errors}
        loss = sum(errors[key][4 + offsets[key]] for key in errors)
        candidates.append((loss, fraction, offsets))
    _, fraction, offsets = min(candidates, key=lambda row: (row[0], row[1]))
    routing = dict(
        schema_version=routed_gaussian.ROUTING_SCHEMA,
        num_layers=m["layers"],
        num_heads=m["heads"],
        offsets={
            "K": [
                [offsets[layer, head] for head in range(m["heads"])]
                for layer in range(m["layers"])
            ],
            "V": [[0] * m["heads"] for _ in range(m["layers"])],
        },
        budget_scope="per_tensor",
        fit=dict(split="train", test_prompts_read=False, fraction=fraction),
    )
    routed_gaussian.validate_routing_table(routing)
    order, tables, cv = helpers.select_v_params(moments)
    run.bundle.mkdir(parents=True, exist_ok=True)
    write_json(run.bundle / "routing.json", routing)
    save_tensor(
        run.bundle / "tree_params.pt",
        dict(schema_version="routed_gaussian.tree_params.v1", **tree_params),
    )
    save_tensor(
        run.bundle / "v_params.pt",
        dict(
            schema_version=PARAMS_SCHEMA,
            methods={
                "midpoint_msb": dict(
                    bit_order=tuple(order),
                    reconstruction=tuple(
                        torch.tensor(table, dtype=torch.float16) for table in tables
                    ),
                    objective="train-only cross-validated residual MSE",
                )
            },
            fit=cv,
        ),
    )
    return [
        run.bundle / name for name in ("routing.json", "tree_params.pt", "v_params.pt")
    ]


def aggregate_weights(run):
    files = event_files(run, "weights")
    events = [json.loads(p.read_text()) for p in files]
    horizon = run.config["calibration"]["schedule"]["horizon"]
    weights = []
    for age in range(horizon):
        observed = [
            event["age_weights"][age]
            for event in events
            if len(event["age_weights"]) > age
        ]
        weights.append(
            max(sum(observed) / len(observed), 1e-9) if observed else weights[-1]
        )
    tensors = {}
    for name in ("K", "V"):
        losses = {
            str(b): sum(e["distortions"][name][str(b)]["loss"] for e in events)
            / len(events)
            for b in (8, 4, 3, 2, 1, 0)
        }
        monotone, floor = {"16": 0.0}, 0.0
        for bits in (8, 4, 3, 2, 1, 0):
            floor = max(floor, losses[str(bits)])
            monotone[str(bits)] = floor
        tensors[name] = dict(
            distortions=monotone, raw_distortions=losses, age_weights=weights
        )
    value = dict(
        schema_version=1,
        tensor_profiles=tensors,
        training=dict(
            split="train",
            dataset_set="calibration",
            test_prompts_read=False,
            objective="K: KL + output NMSE; V: output NMSE",
            calibration_events=len(events),
            terminal_age_extrapolation="repeat oldest observed age",
        ),
    )
    write_json(run.bundle / "weights.json", value)
    return [run.bundle / "weights.json"]


def build_profiles(run):
    weights = json.loads((run.bundle / "weights.json").read_text())
    cfg = run.config["calibration"]["schedule"]
    outputs = []
    inputs = dict(
        schema_version="subitkv.calibration_constants.v1",
        model=run.config["model"], model_config=run.config["model_config"],
        calibration=run.config["calibration"],
        source=dict(training=weights.get("training", {}), weights_sha256=digest(run.bundle / "weights.json")),
        tensor_profiles=weights["tensor_profiles"],
    )
    if run.config["model"] != "hyworldplay_8b":
        inputs["shared_profile"] = dict(
            age_weights=weights["tensor_profiles"]["K"]["age_weights"],
            distortions={str(b): sum(v["distortions"][str(b)] for v in weights["tensor_profiles"].values()) / 2
                         for b in (16, 8, 4, 3, 2, 1, 0)},
            first_anchor_weight=None,
        )
    write_json(run.bundle / "schedule_inputs.json", inputs)
    outputs.append(run.bundle / "schedule_inputs.json")
    for budget in run.config["budgets"]:
        spec = cfg["budgets"][f"{budget:g}"]
        memory = rvq_schedule_budget(run.config, spec["terminal"])
        capacity = max(memory.capacities)

        def schedule(raw):
            rows = spec.get("rows")
            if rows is None:
                rows = (solve_dp(raw, memory.capacities, spec["terminal"])
                        if cfg["mode"] == "base_aware_dp" else solve_schedule(
                            raw, capacity, spec["terminal"], cfg["horizon"],
                            capacities=memory.capacities))
            validate_schedule(
                rows,
                capacity,
                spec["terminal"],
                cfg["horizon"],
                capacities=memory.capacities,
            )
            return rows

        provenance = (
            "frozen_config_schedule"
            if "rows" in spec
            else "base_aware_train_attention_" + ("dp" if cfg["mode"] == "base_aware_dp" else "milp")
        )
        if run.config["model"] == "hyworldplay_8b":
            profiles = {}
            for tensor, raw in weights["tensor_profiles"].items():
                rows = schedule(raw)
                profiles[tensor] = dict(
                    capacity_bit_chunks=capacity,
                    memory_budget=memory.as_dict(),
                    terminal_bit_chunks=spec["terminal"],
                    joint_schedule=rows,
                    age_weights=raw["age_weights"],
                    distortions={
                        str(b): raw["distortions"][str(b)] for b in (8, 4, 3, 2, 1)
                    },
                )
            profile = dict(
                schema_version="hyworldplay_rvq_temporal.profile.v3",
                horizon=cfg["horizon"],
                allowed_bits=[16, 8, 4, 3, 2, 1, 0],
                tensor_profiles=profiles,
                training=dict(test_prompts_read=False, selection_source=provenance),
            )
        else:
            # Existing Wan/SF runtime shares the schedule between K and V.
            raw = dict(
                age_weights=weights["tensor_profiles"]["K"]["age_weights"],
                distortions={
                    str(b): sum(
                        v["distortions"][str(b)]
                        for v in weights["tensor_profiles"].values()
                    )
                    / 2
                    for b in (16, 8, 4, 3, 2, 1, 0)
                },
            )
            rows = schedule(raw)
            profile = dict(
                schema_version="rvq_temporal_adaptive_residual.profile.v1",
                **raw,
                first_anchor_weight=None,
                joint_schedule=rows,
                joint_schedule_capacity_bit_chunks=capacity,
                memory_budget=memory.as_dict(),
                joint_schedule_terminal_bit_chunks=spec["terminal"],
                fit=dict(test_prompts_read=False, selection_source=provenance),
            )
        destination = run.bundle / "profiles" / (label(budget) + ".json")
        write_json(destination, profile)
        outputs.append(destination)
    seal(
        run.bundle,
        run.config["model"],
        identity(run.config),
        calibration="model-local live attention",
        budgets=run.config["budgets"],
        calibration_hash=identity(run.config["calibration"]),
        model_config_hash=identity(run.config["model_config"]),
        model_revisions=(
            json.loads((run.models / "models.lock.json").read_text())
            if (run.models / "models.lock.json").exists()
            else []
        ),
    )
    return outputs + [run.bundle / "manifest.json"]


def calibrate(run, part):
    run.bundle.mkdir(parents=True, exist_ok=True)
    if part in ("all", "codebooks"):
        train_codebooks(run)
    if part in ("all", "residual"):
        teacher_pass(run, "residual")
        run.task("fit_residual", lambda: fit_residual(run))
    if part in ("all", "weights"):
        teacher_pass(run, "weights")
        run.task("aggregate_weights", lambda: aggregate_weights(run))
    if part in ("all", "schedule"):
        run.task("profiles", lambda: build_profiles(run))
