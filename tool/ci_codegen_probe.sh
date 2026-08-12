#!/usr/bin/env bash
# =============================================================================
# TEMPORARY -- CI codegen probe.  DELETE THIS FILE and the two workflow steps
# that call it ("TEMP codegen probe" in .github/workflows/build.yml) once the
# questions below are answered.
#
# Why it exists: several benchmark rows differ between toolchains on identical
# silicon, and the only way to tell "the compiler made different but honest
# code" from "the compiler deleted work we still charge for" is to look at the
# instructions.  CI runners are not reproducible (the Windows x64 runner moved
# from a Granite Rapids Xeon to an EPYC 7763 between runs, taking AVX-512 /
# AMX / FP16 with it), so the probe has to run where the build happens.
#
# Open questions this is meant to settle:
#   1. GCC div/sqrt reads ~22% of the hardware divider rate where clang hits it
#      exactly (2.67 floats/cy on Zen 3).  Is the instruction count right?
#   2. clang-cl vs clang on the SAME EPYC 7763: L1 read -19%, DRAM triad -46%.
#      Did triad vectorise?  Does the read loop fuse loads into ldp/vmovup?
#   3. Did the AVX-512 FP16 collapse fix hold?  Needs 64 vfmadd*ph, was 40.
#   4. ARM read loop: separate ldr q (fast) or fused ldp q (slower on Apple)?
#
# Never fails the job: every command is best-effort and the script always
# exits 0.
# =============================================================================
set +e

echo "::group::TEMP codegen probe -- host"
"${CXX:-c++}" --version 2>&1 | head -2
uname -srm 2>/dev/null
if [ -r /proc/cpuinfo ]; then
  grep -m1 'model name' /proc/cpuinfo
  grep -m1 -E 'cpu MHz|BogoMIPS' /proc/cpuinfo
fi
echo "::endgroup::"

# Pick whatever disassembler this image has.
DIS=""
for c in llvm-objdump objdump otool; do
  command -v "$c" >/dev/null 2>&1 && DIS="$c" && break
done
echo "disassembler: ${DIS:-NONE FOUND}"
[ -z "$DIS" ] && exit 0

dis() {
  case "$DIS" in
    otool)        otool -tv "$1" ;;
    llvm-objdump) llvm-objdump -d --no-show-raw-insn "$1" ;;
    objdump)      objdump -d --no-show-raw-insn "$1" ;;
  esac
}

# count <obj> <label> <extended-regex>
count() {
  local n
  n=$(dis "$1" 2>/dev/null | grep -cE "$3")
  printf "  %-34s %s\n" "$2" "$n"
}

find_obj() {   # find_obj <cmake-target-dir-fragment>
  find build -path "*$1*" \( -name '*.o' -o -name '*.obj' \) 2>/dev/null | head -1
}

# ---- per-ISA kernel TUs -----------------------------------------------------
for tag in generic sse42 avx2 avx512 avx512fp16 avx512bf16 fp16 dotprod bf16 i8mm sve; do
  obj=$(find_obj "peak_cpu_tu_${tag}.dir")
  [ -z "$obj" ] && continue
  echo "::group::TU ${tag}  ($obj)"
  # Q1/Q3: divider + the fast-math estimate substitutions that must NEVER appear
  count "$obj" "x86 divps/divpd"            '\bv?divp[sd]\b'
  count "$obj" "x86 sqrtps/sqrtpd"          '\bv?sqrtp[sd]\b'
  count "$obj" "x86 rcpps/rsqrtps (MUST=0)" '\bv?r(cp|sqrt)p s?'
  count "$obj" "arm fdiv"                   '\bfdiv'
  count "$obj" "arm fsqrt"                  '\bfsqrt'
  count "$obj" "arm frecpe/frsqrte (MUST=0)" '\bf(recpe|rsqrte)'
  # Q3: fp16 chain must emit NACC(16) x UNROLL_K(4) = 64 FMAs, was 40 when it
  # collapsed under -ffast-math.
  count "$obj" "vfmadd*ph  (want 64)"       'vfmadd[0-9]*ph'
  count "$obj" "vdpbf16ps  (want 64)"       'vdpbf16ps'
  # Q2/Q4: shape of the streaming-read loop.
  count "$obj" "x86 256b loads (vmovups y)" 'vmovup[sd][[:space:]]+.*%ymm'
  count "$obj" "arm ldr q (separate)"       '\bldr[[:space:]]+q'
  count "$obj" "arm ldp q (fused)"          '\bldp[[:space:]]+q'
  echo "--- first divide hot loop ---"
  dis "$obj" 2>/dev/null | grep -B4 -A12 -m1 -E '\bv?divp[sd]\b|\bfdiv' | head -20
  echo "::endgroup::"
done

# ---- bandwidth.cpp: did the STREAM triad loop vectorise? --------------------
# triad is the one plain-C loop (A[i] = B[i] + s*C[i]) rather than a per-ISA
# kernel, so it is the most likely place for a toolchain to differ.
obj=$(find_obj "bandwidth.cpp")
if [ -n "$obj" ]; then
  echo "::group::bandwidth.cpp  ($obj)"
  count "$obj" "x86 packed mul/add (vector)" '\bv?(mul|add)p[sd]\b'
  count "$obj" "x86 scalar mul/add (SCALAR)" '\bv?(mul|add)s[sd]\b'
  count "$obj" "x86 FMA (vfmadd*ps)"         'vfmadd[0-9]*p[sd]'
  count "$obj" "arm fmla/fmul vector"        '\bfm(la|ul)[[:space:]]+v'
  count "$obj" "arm fmla/fmul SCALAR"        '\bfm(la|ul)[[:space:]]+[sd]'
  echo "::endgroup::"
fi

exit 0
