#!/usr/bin/env python3
"""Read back what a Vulkan compute shader's inner loop actually does.

The compute-peak shaders report a fixed op budget per work-item, so a shader
whose loop does not issue exactly the ops that budget assumes reports a wrong
number -- silently, on hardware nobody has.  Three of those shipped: a loop the
driver folded, a second accumulator chain derivable from the first, and an
uncounted XOR beside every dot product.  Each was visible in the compiled
SPIR-V.

This freezes the specialization constants to a real device's tile, isolates the
inner loop, and prints what is in it: the work ops, everything issued beside
them, whether the chain is a single dependent chain, and how many distinct
operand pairs the work ops use.  A run of identical work ops is what a compiler
strength-reduces; a chain that is not dependent is not measuring latency; an op
that is not a work op is throughput the budget does not credit.

  tool/shader_ops.py src/vulkan/shaders/coopmat_int8.comp --tile 8x8x32
  tool/shader_ops.py src/vulkan/shaders/compute_int8_dp_v4.comp

Needs glslc and spirv-opt/spirv-dis on PATH (the Vulkan SDK).
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Must match CompileShaders.cmake.
TARGET_ENV = "vulkan1.2"

# The instructions a compute-peak shader is there to issue.  Everything else in
# the loop is overhead the per-work-item op budget does not account for.
WORK_OPS = {
    "OpCooperativeMatrixMulAddKHR": "coopmat MulAdd",
    "OpSDotAccSat": "int8 dot-accumulate",
    "OpUDotAccSat": "uint8 dot-accumulate",
    "OpFma": "fma",
    "OpExtInst": "fma",          # GLSL.std.450 Fma
}

# Bookkeeping every loop has and no budget counts: the trip counter and its
# compare.  Reported separately from real uncounted work.
LOOP_OVERHEAD = {"OpIAdd", "OpSLessThan", "OpULessThan", "OpINotEqual", "OpPhi"}

# Block structure, not issued work.
STRUCTURAL = {"OpBranch", "OpBranchConditional", "OpLoopMerge", "OpSelectionMerge"}

# Traffic on a function-local variable spirv-opt could not promote to SSA.
# It happens for the cooperative-matrix types spirv-opt does not model --
# bfloat16 and fp8 -- and every driver promotes it itself: an RTX 5060 reads
# 42.35 TFLOPS bf16 coopmat against 42.54 for the same tile through CUDA WMMA,
# so nothing survives to the hardware.  Worth printing, not worth failing on.
LOCAL_TRAFFIC = {"OpStore", "OpLoad", "OpAccessChain", "OpVariable"}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{cmd[0]} failed:\n{r.stderr or r.stdout}")
    return r.stdout


def disassemble(shader, specs, include_dir):
    with tempfile.TemporaryDirectory() as tmp:
        spv = Path(tmp) / "s.spv"
        run(["glslc", f"--target-env={TARGET_ENV}", "-O", "-I", str(include_dir),
             str(shader), "-o", str(spv)])
        if specs:
            frozen = Path(tmp) / "f.spv"
            run(["spirv-opt",
                 "--set-spec-const-default-value=" + " ".join(specs),
                 "--freeze-spec-const", "--eliminate-dead-const", "-O",
                 str(spv), "-o", str(frozen)])
            spv = frozen
        return run(["spirv-dis", "--no-color", str(spv)])


INSTR = re.compile(r"^\s*(?:(%\w+)\s*=\s*)?(Op\w+)(.*)$")


def parse(dis):
    """[(block_label, result_id, opcode, operand_ids)] in program order."""
    out, block = [], None
    for line in dis.splitlines():
        m = INSTR.match(line)
        if not m:
            continue
        result, op, rest = m.groups()
        if op == "OpLabel":
            block = result
            continue
        ids = re.findall(r"%\w+", rest)
        # An instruction with a result names its result type first; that is a
        # type, not an operand, and counting it shifts every operand by one.
        if result and ids:
            ids = ids[1:]
        out.append((block, result, op, ids))
    return out


def loop_bodies(instrs):
    """Blocks reachable inside a structured loop, keyed by the merge header."""
    headers = [b for b, _, op, _ in instrs if op == "OpLoopMerge"]
    if not headers:
        return {}
    merges, continues = {}, {}
    for b, _, op, ids in instrs:
        if op == "OpLoopMerge":
            merges[b], continues[b] = ids[0], ids[1]
    bodies = {}
    for h in headers:
        # Everything between the header and the merge block, header excluded:
        # for these shaders that is exactly the straight-line run plus the
        # continue block.
        seen, collecting = [], False
        for b, r, op, ids in instrs:
            if b == h:
                collecting = True
                continue
            if collecting and b == merges[h]:
                break
            if collecting:
                seen.append((b, r, op, ids))
        bodies[h] = seen
    return bodies


def report(shader, body, quiet_ops):
    work = [(r, op, ids) for _, r, op, ids in body if op in WORK_OPS]
    other = [op for _, _, op, _ in body
             if op not in WORK_OPS and op not in STRUCTURAL]
    if not work:
        print("  no work ops in the loop -- the compiler folded it away")
        return False

    kind = WORK_OPS[work[0][1]]
    print(f"  loop body      : {len(work)} x {kind}")

    bookkeeping = [o for o in other if o in LOOP_OVERHEAD]
    local = [o for o in other if o in LOCAL_TRAFFIC]
    overhead = [o for o in other
                if o not in LOOP_OVERHEAD and o not in LOCAL_TRAFFIC]
    if overhead:
        print(f"  UNCOUNTED      : {len(overhead)} op(s) beside the work: "
              f"{', '.join(sorted(set(overhead)))}")
    else:
        print(f"  uncounted      : none "
              f"({len(bookkeeping)} loop-control op(s) only)")
    if local:
        print(f"  note           : {len(local)} local-variable op(s) spirv-opt "
              f"left unpromoted; drivers promote these themselves")

    # Distinct operand pairs.  Every work op here is (src0, src1, accumulator);
    # a run that repeats a pair is a run a compiler can strength-reduce.
    pairs = [tuple(ids[:2]) for _, _, ids in work]
    distinct = len(set(pairs))
    verdict = "OK" if distinct == len(pairs) else "REPEATS -- foldable"
    print(f"  operand pairs  : {distinct} distinct of {len(pairs)}  [{verdict}]")

    # Chain shape: how many work ops consume the immediately preceding result.
    results = [r for r, _, _ in work]
    chained = sum(1 for i in range(1, len(work))
                  if results[i - 1] in work[i][2])
    if chained == len(work) - 1:
        print(f"  chain          : one dependent chain, {len(work)} deep")
    else:
        independent = len(work) - chained
        print(f"  chain          : {independent} independent chain(s)")
    return not overhead and distinct == len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shader", type=Path)
    ap.add_argument("--tile", help="MxNxK to freeze the coopmat shape to, "
                                   "e.g. 8x8x32 (an Arc A380 sint8 tile)")
    ap.add_argument("--subgroup", type=int, default=32,
                    help="subgroup width the tile runs at (default 32)")
    args = ap.parse_args()

    for exe in ("glslc", "spirv-opt", "spirv-dis"):
        if not shutil.which(exe):
            sys.exit(f"{exe} not found; install the Vulkan SDK")

    specs = []
    label = args.shader.stem
    if args.tile:
        m, n, k = (int(x) for x in args.tile.lower().split("x"))
        specs = [f"0:{m}", f"1:{n}", f"2:{k}", f"3:{args.subgroup}"]
        label += f"  {m}x{n}x{k} @ subgroup {args.subgroup}"

    dis = disassemble(args.shader, specs, args.shader.parent)
    instrs = parse(dis)
    bodies = loop_bodies(instrs)

    print(label)
    if not bodies:
        sys.exit("  no loop survived compilation -- the whole run was folded")
    ok = all(report(args.shader, b, args) for b in bodies.values())

    # Trip count: a compile-time bound is one the driver may unroll wholesale.
    header = next(b for b, _, op, _ in instrs if op == "OpLoopMerge")
    bounds = [op for b, _, op, _ in instrs
              if b == header and op in ("OpSLessThan", "OpULessThan")]
    pushed = "OpAccessChain" in [op for b, _, op, _ in instrs if b == header]
    if args.tile:
        print(f"  trip count     : "
              f"{'runtime (pushed)' if bounds and pushed else 'compile-time'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
