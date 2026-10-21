# One empty library per NPU module and ABI (../module.gradle.kts).
# bundletool refuses a bundle whose modules carry native libraries for
# different ABI sets (AbiParityValidator), and the base carries every ABI
# Flutter builds -- arm64-v8a, armeabi-v7a, x86_64 -- while the NPU
# libraries exist for arm64 only.  With this library every module carries
# the base's ABIs, so the bundle still serves the 32-bit phones and x86_64
# Chromebooks the base does; Play delivers the modules to arm64 chipsets
# only, where it sits beside the real libraries unused.
cmake_minimum_required(VERSION 3.22)
project(clpeak_npu_stub C)

add_library(${CLPEAK_NPU_STUB} SHARED stub.c)
