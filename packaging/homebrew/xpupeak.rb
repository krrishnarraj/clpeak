# Why a git URL instead of a release tarball: XpuPeak derives its version from
# `git describe` (src/common/cmake/version.cmake).  A GitHub source tarball has
# no .git, so the build would report its version as "unknown".  Passing
# `tag:` makes Homebrew fetch that tag, so git-describe reports the real version.
class Xpupeak < Formula
  desc "Peak compute, bandwidth and latency benchmark for CPUs, GPUs and NPUs"
  homepage "https://github.com/krrishnarraj/xpupeak"
  # TODO: add `revision:` with 4.0.0's commit sha once the tag is pushed —
  # homebrew-core requires it alongside `tag:`.  The URLs name the repo as it
  # is renamed just before 4.0.0.
  url "https://github.com/krrishnarraj/xpupeak.git",
      tag: "4.0.0"
  license "Apache-2.0"
  head "https://github.com/krrishnarraj/xpupeak.git", branch: "master"

  livecheck do
    url :stable
    strategy :github_latest
  end

  depends_on "cmake"          => :build
  depends_on "ninja"          => :build
  depends_on "shaderc"        => :build # provides `glslc` to compile the Vulkan shaders
  depends_on "vulkan-headers" => :build # headers only; the loader below is the runtime lib

  depends_on "vulkan-loader" # libvulkan, dlopen'd at runtime

  on_macos do
    depends_on "molten-vk" # Vulkan-over-Metal ICD (OpenCL/Metal come from the macOS SDK)
  end

  on_linux do
    # CPU-backend codegen needs clang: GCC<=14 serialises the kernels' FMA
    # accumulator chains, ~halving fp32/fp64 (see CMakeLists.txt top comment).
    # macOS already builds with AppleClang; on Linuxbrew the default is GCC, and
    # Homebrew exports CC/CXX which suppresses the build's own clang
    # auto-detection, so pull in LLVM and force clang in `install`.
    depends_on "llvm" => :build
  end

  def install
    # CUDA/ROCm/oneAPI need vendor toolkits + drivers Homebrew can't provide, so
    # disable them explicitly (they would auto-disable, but this keeps the build
    # deterministic if a system SDK happens to be present on a Linuxbrew box).
    # The desktop app needs a Flutter SDK, which no formula provides, so it is
    # off for the same reason: only the command-line tool is installed.
    # Metal/OpenCL (macOS) and OpenCL (Linux) stay on and auto-detect.
    args = %w[
      -DXPUPEAK_ENABLE_CUDA=OFF
      -DXPUPEAK_ENABLE_ROCM=OFF
      -DXPUPEAK_ENABLE_ONEAPI=OFF
      -DXPUPEAK_ENABLE_GUI=OFF
    ]

    # Build with clang for the CPU-backend codegen.  macOS already uses
    # AppleClang; on Linux point CMake at the brewed LLVM clang explicitly.
    if OS.linux?
      llvm = formula_opt_bin("llvm")
      args << "-DCMAKE_C_COMPILER=#{llvm}/clang"
      args << "-DCMAKE_CXX_COMPILER=#{llvm}/clang++"
    end

    system "cmake", "-S", ".", "-B", "build", "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release", *args, *std_cmake_args
    system "cmake", "--build", "build"
    # Installs bin/xpupeak and, beside it, bin/clpeak: the 3.x command, a
    # link to the same program.
    system "cmake", "--install", "build"
  end

  test do
    # CI runners have no GPU; both commands must still run and exit cleanly
    # with no devices.  --version exercises the git-describe version wiring,
    # and its line starts with the name the command was started as.
    system bin/"xpupeak", "--help"
    assert_match "xpupeak version: #{version}", shell_output("#{bin}/xpupeak --version")
    assert_match "clpeak version: #{version}", shell_output("#{bin}/clpeak --version")
  end
end
