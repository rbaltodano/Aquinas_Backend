"""Controlled, resumable 8-bit DWQ diagnostics for the Aquinas checkpoint.

This is research tooling, not a production export recipe.  It implements the
cache, telemetry, checkpoint, and fresh-process resume gates required by the
authoritative protocol in Aquinas-QAT-DWQ-Writeup.md.  The conservative
defaults intentionally describe the first 128-token diagnostic only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import mlx_lm.quant.dwq as installed_dwq
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.quant.dwq import load_data
from mlx_lm.tuner.losses import kl_div_loss
from mlx_lm.tuner.trainer import grad_checkpoint, iterate_batches
from mlx_lm.utils import (
    get_total_parameters,
    load_model,
    load_tokenizer,
    quantize_model,
    save,
)


MODEL_PATH = Path("models/Aquinas-Final-HF")
DEFAULT_OUTPUT = Path("models/dwq_full_run_8bit")
EXPECTED_TRAINABLE_RATIO = (0.0310, 0.0314)


@dataclass(frozen=True)
class RunConfig:
    model_path: str
    output_path: str
    num_samples: int
    batch_size: int
    max_seq_length: int
    learning_rate: float
    seed: int
    bits: int
    group_size: int
    gradient_checkpoint: bool
    cache_limit_bytes: int
    checkpoint_every: int
    validation_every: int
    max_steps: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--cache-limit-bytes", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--prepare-resume-test-at",
        type=int,
        help="Checkpoint after this many updates, compute the expected next update, and exit.",
    )
    parser.add_argument(
        "--verify-resume-test",
        action="store_true",
        help="Reload --resume, verify its validation fingerprint and exact next update, then exit.",
    )
    parser.add_argument("--disk-stop-gib", type=float, default=60.0)
    parser.add_argument("--swap-growth-stop-gib", type=float, default=8.0)
    parser.add_argument("--monitor-interval", type=float, default=10.0)
    return parser.parse_args()


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def memory_snapshot(event: str, iteration: int) -> dict[str, Any]:
    snapshot = {
        "event": event,
        "iteration": iteration,
        "timestamp": time.time(),
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }
    print("MEMORY " + json.dumps(snapshot, sort_keys=True), flush=True)
    return snapshot


def record_environment(run_dir: Path) -> None:
    installed_dwq_path = Path(inspect.getfile(installed_dwq))
    payload = {
        "timestamp": time.time(),
        "python": sys.version,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "device_info": mx.device_info(),
        "files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(Path(__file__).with_name("dwq_monitor.py").resolve()): sha256(
                Path(__file__).with_name("dwq_monitor.py").resolve()
            ),
            str(installed_dwq_path.resolve()): sha256(installed_dwq_path.resolve()),
        },
        "git_status": subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
    }
    json_dump(run_dir / "environment.json", payload)


def save_array_tree(path: Path, tree: Any, prefix: str = "") -> None:
    arrays = {
        prefix + name: value
        for name, value in tree_flatten(tree)
        if isinstance(value, mx.array)
    }
    mx.eval(arrays)
    mx.save_safetensors(str(path), dict(sorted(arrays.items())))


def load_array_tree(path: Path, prefix: str = "") -> Any:
    arrays = mx.load(str(path))
    if prefix:
        arrays = {
            name[len(prefix) :]: value
            for name, value in arrays.items()
            if name.startswith(prefix)
        }
    return tree_unflatten(dict(sorted(arrays.items())))


def state_digest(path: Path, params: Any, optimizer_state: Any) -> str:
    combined = {}
    combined.update({"params." + k: v for k, v in tree_flatten(params)})
    combined.update({"optimizer." + k: v for k, v in tree_flatten(optimizer_state)})
    mx.eval(combined)
    mx.save_safetensors(str(path), dict(sorted(combined.items())))
    digest = sha256(path)
    path.unlink()
    return digest


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        model_path=str(MODEL_PATH),
        output_path=str(args.output),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        seed=args.seed,
        bits=args.bits,
        group_size=args.group_size,
        gradient_checkpoint=True,
        cache_limit_bytes=args.cache_limit_bytes,
        checkpoint_every=args.checkpoint_every,
        validation_every=args.validation_every,
        max_steps=args.max_steps,
    )


def require_config_match(expected: RunConfig, metadata: dict[str, Any]) -> None:
    recorded = metadata["config"]
    current = asdict(expected)
    mismatches = {
        key: {"checkpoint": recorded.get(key), "current": value}
        for key, value in current.items()
        if recorded.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint configuration drift: {mismatches}")


def unfreeze_quantization_parameters(model: nn.Module) -> None:
    def unfreeze(_: str, module: nn.Module) -> None:
        if (
            hasattr(module, "bits")
            and hasattr(module, "group_size")
            and module.mode == "affine"
            and module.bits <= 8
        ):
            module.unfreeze(keys=["scales", "biases"], recurse=False)

    model.train()
    model.apply_to_modules(unfreeze)


def parameter_counts(model: nn.Module) -> tuple[int, int, float]:
    trainable = sum(value.size for _, value in tree_flatten(model.trainable_parameters()))
    # Quantized weights are physically packed, so summing stored array sizes
    # undercounts the logical parameter denominator. Match mlx-lm's own
    # print_trainable_parameters() accounting instead.
    total = get_total_parameters(model)
    ratio = trainable / total
    if not EXPECTED_TRAINABLE_RATIO[0] <= ratio <= EXPECTED_TRAINABLE_RATIO[1]:
        raise RuntimeError(
            f"Unexpected trainable ratio {ratio:.6%} ({trainable}/{total}); "
            "the 8-bit unfreeze gate may be missing or the model changed."
        )
    print(f"Trainable parameters: {ratio:.3%} ({trainable / 1e6:.3f}M/{total / 1e6:.3f}M)", flush=True)
    return trainable, total, ratio


def make_batch_iterator(data: Any, config: RunConfig, skip: int = 0):
    iterator = iterate_batches(
        data,
        config.batch_size,
        config.max_seq_length,
        seed=config.seed,
    )
    for _ in range(skip):
        next(iterator)
    return iterator


def start_monitor(args: argparse.Namespace, run_dir: Path) -> subprocess.Popen:
    monitor_log = run_dir / "monitor.jsonl"
    stop_file = run_dir / "STOP"
    command = [
        sys.executable,
        str(Path(__file__).with_name("dwq_monitor.py")),
        "--pid",
        str(os.getpid()),
        "--log",
        str(monitor_log),
        "--stop-file",
        str(stop_file),
        "--interval",
        str(args.monitor_interval),
        "--disk-stop-gib",
        str(args.disk_stop_gib),
        "--swap-growth-stop-gib",
        str(args.swap_growth_stop_gib),
    ]
    process = subprocess.Popen(command)
    print(f"Started monitor pid={process.pid} log={monitor_log}", flush=True)
    return process


def require_monitor_healthy(run_dir: Path, process: subprocess.Popen) -> None:
    stop_file = run_dir / "STOP"
    if process.poll() is not None:
        raise RuntimeError(f"Monitor exited unexpectedly with status {process.returncode}")
    if stop_file.exists():
        raise RuntimeError(f"Monitor requested stop: {stop_file.read_text(encoding='utf-8').strip()}")


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()) and args.resume is None:
        raise RuntimeError(f"New run directory must be empty: {run_dir}")
    json_dump(run_dir / "config.json", asdict(config))
    record_environment(run_dir)

    monitor = start_monitor(args, run_dir)
    old_cache_limit: int | None = None
    old_wired_limit: int | None = None
    status = "failed"
    try:
        old_cache_limit = mx.set_cache_limit(config.cache_limit_bytes)
        device_info = mx.device_info()
        recommended_limit = device_info["max_recommended_working_set_size"]
        old_wired_limit = mx.set_wired_limit(recommended_limit)
        print(
            f"MLX limits: cache {old_cache_limit} -> {config.cache_limit_bytes}; "
            f"wired {old_wired_limit} -> {recommended_limit}",
            flush=True,
        )
        memory_snapshot("baseline", 0)
        require_monitor_healthy(run_dir, monitor)

        print("Loading model (strict=False to drop unused shared-KV weights)...", flush=True)
        teacher, model_config = load_model(MODEL_PATH, lazy=True, strict=False)
        tokenizer = load_tokenizer(MODEL_PATH)
        print("Loading data...", flush=True)
        train_data, valid_data = load_data(
            tokenizer,
            "data/training_data",
            config.num_samples,
            config.max_seq_length,
        )
        print(f"train={len(train_data)} valid={len(valid_data)}", flush=True)

        print("Quantizing student model...", flush=True)
        student = copy.deepcopy(teacher)
        _, quantization_config = quantize_model(
            student,
            model_config,
            group_size=config.group_size,
            bits=config.bits,
        )
        unfreeze_quantization_parameters(student)
        trainable, total, ratio = parameter_counts(student)
        if config.gradient_checkpoint:
            grad_checkpoint(student.layers[0])
        memory_snapshot("models_loaded", 0)
        require_monitor_healthy(run_dir, monitor)

        optimizer = optimizers.Adam(
            learning_rate=config.learning_rate,
            bias_correction=True,
        )
        scale = 0.5

        def target_fn(batch: mx.array) -> mx.array:
            targets = teacher(batch)
            mx.eval(targets)
            return targets

        def loss_fn(params: Any, batch: mx.array, targets: mx.array, lengths: mx.array):
            student.update(tree_map(lambda value: value.astype(mx.bfloat16), params))
            logits = student(batch)
            losses = kl_div_loss(scale * logits, scale * targets)
            mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]
            ntokens = mask.sum()
            return (mask * losses).sum() / ntokens, ntokens

        def step(batch: mx.array, lengths: mx.array, params: Any):
            inputs = batch[:, :-1]
            targets = target_fn(inputs)
            (loss, ntokens), gradients = mx.value_and_grad(loss_fn)(
                params, inputs, targets, lengths
            )
            gradients = nn.average_gradients(gradients)
            checks = [(name, mx.all(mx.isfinite(value))) for name, value in tree_flatten(gradients)]
            gradients_finite = mx.all(mx.stack([check for _, check in checks]))
            mx.eval(loss, ntokens, gradients_finite)
            if not mx.isfinite(loss).item() or not gradients_finite.item():
                bad = [name for name, check in checks if not check.item()]
                raise RuntimeError(
                    f"Rejected non-finite update: loss={loss.item()} bad_grad_count={len(bad)} "
                    f"bad_grads={bad[:12]}"
                )
            updated = optimizer.apply_gradients(gradients, params)
            mx.eval(updated, optimizer.state)
            return loss.item(), ntokens.item(), updated

        def validate(
            params: Any,
            limit_batches: int | None = None,
            event_iteration: int = 0,
        ) -> float:
            memory_snapshot("validation_start", event_iteration)
            weighted_loss = 0.0
            token_count = 0
            iterator = iterate_batches(
                valid_data,
                config.batch_size,
                config.max_seq_length,
                seed=config.seed,
            )
            for index, (batch, lengths) in enumerate(iterator):
                inputs = batch[:, :-1]
                targets = target_fn(inputs)
                loss, ntokens = loss_fn(params, inputs, targets, lengths)
                mx.eval(loss, ntokens)
                count = ntokens.item()
                weighted_loss += loss.item() * count
                token_count += count
                if limit_batches is not None and index + 1 >= limit_batches:
                    break
            if token_count == 0:
                raise RuntimeError("Validation produced zero tokens")
            result = weighted_loss / token_count
            memory_snapshot("validation_end", event_iteration)
            return result

        params = tree_map(lambda value: value.astype(mx.float32), student.trainable_parameters())
        iteration = 0
        skipped_updates = 0
        initial_validation: float
        latest_validation: float

        if args.resume:
            checkpoint = args.resume.resolve()
            metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
            require_config_match(config, metadata)
            full_student_path = checkpoint / "full-student.safetensors"
            if not full_student_path.exists():
                raise RuntimeError(
                    "Checkpoint predates full quantized-student serialization and cannot "
                    "be used for exact resume verification"
                )
            student.load_weights(str(full_student_path), strict=True)
            params = load_array_tree(checkpoint / "student.safetensors")
            optimizer.state = load_array_tree(checkpoint / "optimizer.safetensors")
            mx.eval(student.parameters(), params, optimizer.state)
            iteration = metadata["iteration"]
            skipped_updates = metadata["skipped_updates"]
            initial_validation = metadata["initial_validation"]
            latest_validation = metadata["latest_validation"]
            if skipped_updates != 0:
                raise RuntimeError("Candidate checkpoint contains skipped updates")
            fingerprint = validate(params, limit_batches=2, event_iteration=iteration)
            expected_fingerprint = metadata["validation_fingerprint"]
            if not np.isclose(fingerprint, expected_fingerprint, rtol=0.0, atol=1e-7):
                raise RuntimeError(
                    f"Reload validation mismatch: {fingerprint} != {expected_fingerprint}"
                )
            print(
                f"Reloaded {checkpoint}: iteration={iteration} "
                f"validation_fingerprint={fingerprint:.9f}",
                flush=True,
            )
        else:
            initial_validation = latest_validation = validate(params, event_iteration=iteration)
            print(f"Validation: iteration=0 loss={initial_validation:.9f}", flush=True)

        def checkpoint_state(reason: str) -> tuple[Path, dict[str, Any]]:
            nonlocal latest_validation
            checkpoint = run_dir / f"checkpoint-{iteration:06d}"
            checkpoint.mkdir()
            fingerprint = validate(params, limit_batches=2, event_iteration=iteration)
            repeated_fingerprint = validate(
                params,
                limit_batches=2,
                event_iteration=iteration,
            )
            if not np.isclose(
                fingerprint,
                repeated_fingerprint,
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError(
                    "Validation is not deterministic before checkpoint: "
                    f"{fingerprint} != {repeated_fingerprint}"
                )
            # loss_fn leaves the live student synchronized to the bf16 form of
            # the float32 accumulator. Persist the entire quantized student as
            # well as that accumulator: recreating frozen quantized tensors is
            # not an exact restart contract.
            student.save_weights(str(checkpoint / "full-student.safetensors"))
            save_array_tree(checkpoint / "student.safetensors", params)
            save_array_tree(checkpoint / "optimizer.safetensors", optimizer.state)
            metadata = {
                "schema_version": 1,
                "reason": reason,
                "iteration": iteration,
                "data_position": iteration,
                "seed": config.seed,
                "skipped_updates": skipped_updates,
                "initial_validation": initial_validation,
                "latest_validation": latest_validation,
                "validation_fingerprint": fingerprint,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "trainable_ratio": ratio,
                "config": asdict(config),
                "student_sha256": sha256(checkpoint / "student.safetensors"),
                "full_student_sha256": sha256(
                    checkpoint / "full-student.safetensors"
                ),
                "optimizer_sha256": sha256(checkpoint / "optimizer.safetensors"),
                "memory": memory_snapshot("checkpoint_saved", iteration),
            }
            json_dump(checkpoint / "metadata.json", metadata)
            mx.clear_cache()
            mx.reset_peak_memory()
            memory_snapshot("checkpoint_cache_cleared", iteration)
            require_monitor_healthy(run_dir, monitor)
            return checkpoint, metadata

        iterator = make_batch_iterator(train_data, config, skip=iteration)

        if args.verify_resume_test:
            if not args.resume:
                raise RuntimeError("--verify-resume-test requires --resume")
            metadata = json.loads((args.resume / "metadata.json").read_text(encoding="utf-8"))
            expected = metadata.get("expected_next")
            if not expected:
                raise RuntimeError("Checkpoint has no expected-next-step record")
            batch, lengths = next(iterator)
            next_loss, _, params = step(batch, lengths, params)
            iteration += 1
            digest = state_digest(run_dir / "actual-next-state.safetensors", params, optimizer.state)
            if not np.isclose(next_loss, expected["loss"], rtol=0.0, atol=1e-7):
                raise RuntimeError(f"Next-step loss mismatch: {next_loss} != {expected['loss']}")
            if digest != expected["state_sha256"]:
                raise RuntimeError(f"Next-step state mismatch: {digest} != {expected['state_sha256']}")
            result = {
                "status": "passed",
                "resumed_from": str(args.resume.resolve()),
                "iteration": iteration,
                "loss": next_loss,
                "state_sha256": digest,
                "timestamp": time.time(),
            }
            json_dump(run_dir / "resume-verification.json", result)
            print("Fresh-process resume verification PASSED: " + json.dumps(result), flush=True)
            status = "resume-verification-passed"
            return

        started = time.time()
        last_checkpoint_iteration = -1
        while iteration < len(train_data) // config.batch_size:
            require_monitor_healthy(run_dir, monitor)
            batch, lengths = next(iterator)
            loss, ntokens, params = step(batch, lengths, params)
            iteration += 1
            print(
                f"TRAIN iteration={iteration} loss={loss:.9f} tokens={ntokens} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

            if args.prepare_resume_test_at == iteration:
                checkpoint, metadata = checkpoint_state("resume-test")
                next_batch, next_lengths = next(iterator)
                expected_loss, _, expected_params = step(next_batch, next_lengths, params)
                expected_digest = state_digest(
                    run_dir / "expected-next-state.safetensors",
                    expected_params,
                    optimizer.state,
                )
                metadata["expected_next"] = {
                    "iteration": iteration + 1,
                    "loss": expected_loss,
                    "state_sha256": expected_digest,
                }
                json_dump(checkpoint / "metadata.json", metadata)
                print(f"Prepared resume test checkpoint: {checkpoint}", flush=True)
                status = "resume-test-prepared"
                return

            if iteration % config.validation_every == 0:
                latest_validation = validate(params, event_iteration=iteration)
                print(f"Validation: iteration={iteration} loss={latest_validation:.9f}", flush=True)
                if latest_validation > initial_validation:
                    raise RuntimeError(
                        f"Validation regression: {latest_validation} > {initial_validation}"
                    )
                checkpoint_state("validation-gate")
                last_checkpoint_iteration = iteration
            elif iteration % config.checkpoint_every == 0:
                checkpoint_state("safety")
                last_checkpoint_iteration = iteration

            if config.max_steps is not None and iteration >= config.max_steps:
                if last_checkpoint_iteration != iteration:
                    checkpoint_state("max-steps")
                status = "diagnostic-complete"
                return

        latest_validation = validate(params, event_iteration=iteration)
        if latest_validation > initial_validation:
            raise RuntimeError(f"Final validation regression: {latest_validation} > {initial_validation}")
        checkpoint_state("final")
        student.update(tree_map(lambda value: value.astype(mx.bfloat16), params))
        save(config.output_path, str(MODEL_PATH), student, tokenizer, quantization_config)
        status = "candidate-saved"
    finally:
        print(f"Run status: {status}", flush=True)
        if old_cache_limit is not None:
            restored_from = mx.set_cache_limit(old_cache_limit)
            print(f"Restored MLX cache limit {restored_from} -> {old_cache_limit}", flush=True)
        if old_wired_limit is not None:
            restored_from = mx.set_wired_limit(old_wired_limit)
            print(f"Restored MLX wired limit {restored_from} -> {old_wired_limit}", flush=True)
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=5)
        json_dump(run_dir / "final-status.json", {"status": status, "timestamp": time.time()})


if __name__ == "__main__":
    main()
