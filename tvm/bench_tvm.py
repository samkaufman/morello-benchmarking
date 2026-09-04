import argparse
import contextlib
import datetime
import functools
import json
import logging
import os
import pathlib
import sys
import tempfile
import time
from collections.abc import Callable, Iterator

import numpy

logger = logging.getLogger(__name__)

DEFAULT_TRIALS = 128
STATS_FILENAME = "build_stats.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    arg_parser = argparse.ArgumentParser()
    subparsers = arg_parser.add_subparsers(dest="workload", required=True)

    matmul_parsers = []
    for dtype_name, numpy_dtype in (("f32", "float32"), ("u32", "uint32")):
        matmul_parser = subparsers.add_parser(f"batch-parallel-{dtype_name}")
        matmul_parser.add_argument("batch_size", type=int)
        matmul_parser.add_argument("m", type=int)
        matmul_parser.add_argument("k", type=int)
        matmul_parser.add_argument("n", type=int)
        matmul_parser.set_defaults(
            func=functools.partial(
                run_batch_parallel_matmul,
                dtype_name=dtype_name,
                dtype=numpy_dtype,
            )
        )
        matmul_parsers.append(matmul_parser)

    softmax_parser = subparsers.add_parser("softmax-f32")
    softmax_parser.add_argument("batch_size", type=int)
    softmax_parser.add_argument("length", type=int)
    softmax_parser.add_argument("num_threads", type=int)
    softmax_parser.set_defaults(func=run_softmax_f32)

    for parser in (*matmul_parsers, softmax_parser):
        parser.add_argument(
            "--scheduling",
            choices=("relax", "metaschedule"),
            default="relax",
            help="relax: out-of-the-box Relax build; metaschedule: auto-tune",
        )
        parser.add_argument(
            "--trials",
            type=int,
            default=DEFAULT_TRIALS,
            help="MetaSchedule tuning trials",
        )

    args = arg_parser.parse_args()
    if not hasattr(args, "num_threads"):
        args.num_threads = args.batch_size
    os.environ["TVM_NUM_THREADS"] = str(args.num_threads)
    args.func(args)


def run_batch_parallel_matmul(
    args: argparse.Namespace, dtype_name: str, dtype: str
) -> None:
    from tvm import relax, topi

    b, m, k, n = args.batch_size, args.m, args.k, args.n

    _run_benchmark(
        args.scheduling,
        args.trials,
        num_threads=args.num_threads,
        input_shapes=[(b, m, k), (b, k, n)],
        out_shape=(b, m, n),
        relax_op=relax.op.matmul,
        topi_op=functools.partial(
            topi.nn.batch_matmul, transpose_a=False, transpose_b=False
        ),
        workload_name=f"batch-matmul-{dtype_name}-{b}x{m}x{k}x{n}",
        dtype=dtype,
    )


def run_softmax_f32(args: argparse.Namespace) -> None:
    from tvm import relax, topi

    batch, length = args.batch_size, args.length

    _run_benchmark(
        args.scheduling,
        args.trials,
        num_threads=args.num_threads,
        input_shapes=[(batch, length)],
        out_shape=(batch, length),
        relax_op=functools.partial(relax.op.nn.softmax, axis=-1),
        topi_op=functools.partial(topi.nn.softmax, axis=-1),
        workload_name=f"softmax-f32-{batch}x{length}",
        dtype="float32",
    )


def _run_benchmark(
    scheduling: str,
    trials: int,
    num_threads: int,
    input_shapes: list[tuple[int, ...]],
    out_shape: tuple[int, ...],
    relax_op: Callable,
    topi_op: Callable,
    workload_name: str,
    dtype: str,
) -> None:
    """Runs a workload under the given scheduling mode."""
    import tvm

    target = _host_target(num_cores=num_threads)

    rng = numpy.random.default_rng(0)
    device = tvm.cpu(0)
    inputs = [
        tvm.nd.array(_random_input(rng, shape, dtype), device)
        for shape in input_shapes
    ]

    if scheduling == "relax":
        vm = _build_relax_vm(input_shapes, relax_op, target, dtype)
        vm["main"](*inputs)  # Warm-up
        _time_and_report(vm.module, device, *inputs)
    else:
        prim_func = _te_prim_func(input_shapes, topi_op, target, dtype=dtype)
        lib = _build_tuned(workload_name, prim_func, target, trials)
        out_nd = tvm.nd.empty(out_shape, dtype, device)
        lib["main"](*inputs, out_nd)  # Warm-up
        _time_and_report(lib, device, *inputs, out_nd)


def _random_input(
    rng: numpy.random.Generator, shape: tuple[int, ...], dtype: str
) -> numpy.ndarray:
    if dtype == "float32":
        # Same input distribution as the other f32 baselines: [1, 2).
        return rng.random(shape, dtype=numpy.float32) + 1.0
    # Integer workloads: small positive values.
    return rng.integers(1, 100, size=shape, dtype=dtype)


def _host_target(num_cores: int | None = None) -> str:
    """Builds an LLVM target string aimed at the host CPU.

    Like the other baselines, which detect the host's microarchitecture at
    runtime, TVM targets whatever machine it's running on: containers are
    built and run on the same machine.
    """
    import tvm

    target = f"llvm -mcpu={tvm.target.codegen.llvm_get_system_cpu()}"
    if num_cores is not None:
        target += f" -num-cores {num_cores}"
    return target


def _te_prim_func(
    input_shapes: list[tuple[int, ...]],
    topi_op: Callable,
    target: str,
    dtype: str = "float32",
):
    """Builds a TIR PrimFunc computing `topi_op` over placeholder inputs."""
    import tvm
    from tvm import te

    placeholders = [
        te.placeholder(shape, dtype=dtype) for shape in input_shapes
    ]
    with tvm.target.Target(target):
        out = topi_op(*placeholders)
    return te.create_prim_func(placeholders + [out])


def _build_relax_vm(
    input_shapes: list[tuple[int, ...]],
    op_builder: Callable,
    target: str,
    dtype: str,
):
    """Compiles a single-op model with TVM's out-of-the-box Relax pipeline.

    No scheduling or tuning is applied beyond tvm.relax.build's standard
    pipelines. Returns a Relax VirtualMachine whose "main" takes the input
    tensors and returns the output tensor.
    """
    import tvm
    from tvm import relax

    bb = relax.BlockBuilder()
    params = [
        relax.Var(f"x{i}", relax.TensorStructInfo(shape, dtype))
        for i, shape in enumerate(input_shapes)
    ]
    with bb.function("main", params):
        out = bb.emit(op_builder(*params))
        bb.emit_func_output(out)
    mod = bb.get()

    build_start = time.perf_counter()
    with _stdout_to_stderr():
        executable = tvm.relax.build(mod, target=target)
    stats = _base_stats(target)
    stats.update(
        scheduling="relax", compile_seconds=time.perf_counter() - build_start
    )
    _write_output_file(STATS_FILENAME, json.dumps(stats, indent=2))

    # Log the compiled module (Relax "main" plus generated TIR kernels). The
    # lowering below re-runs most of the build, so skip it when an earlier
    # invocation of this job already dumped it.
    out_dir = _output_dir()
    if out_dir is not None and not (out_dir / "tir.txt").exists():
        try:
            with _stdout_to_stderr(), tvm.target.Target(target):
                lowered = relax.pipeline.default_build_pipeline()(mod)
            _write_output_file("tir.txt", str(lowered))
        except Exception as exc:  # Diagnostics only; never fail the benchmark.
            logger.warning("Failed to dump lowered module: %s", exc)

    return relax.VirtualMachine(executable, tvm.cpu(0))


def _base_stats(target: str) -> dict[str, object]:
    import tvm

    return {
        "target": target,
        "num_threads": os.environ.get("TVM_NUM_THREADS"),
        "tvm_version": tvm.__version__,
    }


def _copy_cached_stats(stats_path: pathlib.Path) -> None:
    """Copies cached tuning stats into the job's output directory.

    Skipped when the output directory already has stats: the invocation that
    actually tuned shares that directory, and its record is the interesting
    one.
    """
    out_dir = _output_dir()
    if out_dir is None or not stats_path.exists():
        return
    dest = out_dir / STATS_FILENAME
    if dest.exists():
        return
    stats = json.loads(stats_path.read_text())
    stats["cache_hit"] = True
    dest.write_text(json.dumps(stats, indent=2))


def _build_tuned(workload_name: str, prim_func, target: str, trials: int):
    """Returns a runtime module for `prim_func`, tuned with MetaSchedule.

    The tuned, compiled artifact is cached under `_cache_root()`; a cache hit
    skips tuning entirely. The cache key includes everything the artifact was
    specialized for — the workload, the host CPU (the cache may live on a
    mount shared between machines), thread count, and trial budget.
    """
    import tvm
    from tvm import meta_schedule as ms

    cpu = tvm.target.codegen.llvm_get_system_cpu()
    num_threads = os.environ["TVM_NUM_THREADS"]
    cache_key = f"{workload_name}-{cpu}-t{num_threads}-x{trials}"

    cache_dir = _cache_root() / cache_key
    lib_path = cache_dir / "lib.so"
    tir_path = cache_dir / "tir.txt"
    stats_path = cache_dir / STATS_FILENAME
    if lib_path.exists():
        logger.info("Using cached tuned module: %s", lib_path)
        if tir_path.exists():
            _write_output_file(
                "tir.txt", tir_path.read_text(), skip_existing=True
            )
        _copy_cached_stats(stats_path)
        return tvm.runtime.load_module(str(lib_path))

    work_dir = cache_dir / "tuning"
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Tuning %s with MetaSchedule (%d trials)", cache_key, trials)
    with _stdout_to_stderr():
        tune_start = time.perf_counter()
        database = ms.tune_tir(
            mod=prim_func,
            target=target,
            work_dir=str(work_dir),
            max_trials_global=trials,
        )
        tune_end = time.perf_counter()
        sch = ms.tir_integration.compile_tir(database, prim_func, target)
        if sch is None:
            raise RuntimeError("MetaSchedule did not produce a schedule")
        lib = tvm.build(sch.mod, target=target)
        compile_end = time.perf_counter()

    stats = _base_stats(target)
    stats.update(
        workload=cache_key,
        scheduling="metaschedule",
        cache_hit=False,
        trials_requested=trials,
        tune_seconds=tune_end - tune_start,
        compile_seconds=compile_end - tune_end,
        tuned_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    try:
        records = database.get_all_tuning_records()
        stats["trials_measured"] = len(records)
        trial_means = [
            sum(float(s) for s in record.run_secs) / len(record.run_secs)
            for record in records
            if record.run_secs
        ]
        if trial_means:
            stats["best_trial_seconds"] = min(trial_means)
    except Exception as exc:  # Diagnostics only; never fail the benchmark.
        logger.warning("Failed to summarize tuning records: %s", exc)
    stats_json = json.dumps(stats, indent=2)
    stats_path.write_text(stats_json)
    _write_output_file(STATS_FILENAME, stats_json)

    # Also keep the per-trial tuning log, which records each measured
    # candidate, for offline analysis of search progress vs. time budget.
    records_path = work_dir / "database_tuning_record.json"
    if records_path.exists():
        _write_output_file(
            "metaschedule_tuning_record.json", records_path.read_text()
        )

    tir_text = str(sch.mod)
    tir_path.write_text(tir_text)
    _write_output_file("tir.txt", tir_text)
    # Export to a temporary path and rename so an interrupted invocation
    # can't leave a truncated lib.so in the cache.
    tmp_path = lib_path.with_name("lib.so.tmp")
    lib.export_library(str(tmp_path))
    tmp_path.rename(lib_path)
    return tvm.runtime.load_module(str(lib_path))


def _time_and_report(module, device, *tensors) -> None:
    """Times the module's "main" and prints one line per outer step.

    Uses TVM's time_evaluator so the inner loop is not interpreted.
    """
    outer_steps = int(os.getenv("CHERRYBENCH_OUTER_STEPS", "10"))
    inner_steps = int(os.environ["CHERRYBENCH_LOOP_STEPS"])
    evaluator = module.time_evaluator(
        "main", device, number=inner_steps, repeat=outer_steps
    )
    for mean_secs in evaluator(*tensors).results:
        print(f"{mean_secs * inner_steps:.10f}s")


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Temporarily points file descriptor 1 at stderr.

    MetaSchedule installs console log handlers on sys.stdout, but stdout must
    carry only timing lines for cherrybench.
    """
    sys.stdout.flush()
    saved_fd = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def _cache_root() -> pathlib.Path:
    cherrybench_mount = pathlib.Path("/cherrybench")
    if cherrybench_mount.is_dir():
        return cherrybench_mount / "tvm-cache"
    return pathlib.Path(tempfile.gettempdir()) / "tvm-bench-cache"


def _output_dir() -> pathlib.Path | None:
    out_dir = os.getenv("CHERRYBENCH_OUTPUT_DIR")
    return pathlib.Path(out_dir) if out_dir else None


def _write_output_file(
    name: str, content: str, skip_existing: bool = False
) -> None:
    out_dir = _output_dir()
    if out_dir is None:
        return
    dest = out_dir / name
    if skip_existing and dest.exists():
        return
    dest.write_text(content)


if __name__ == "__main__":
    main()
