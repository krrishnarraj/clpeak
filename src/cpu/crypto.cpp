#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

// Crypto/hash throughput tests (Category::Crypto): AES-128 encrypt, SHA-256 /
// SHA-512 compression, CRC32-C.
// These measure the dedicated crypto pipes in GB/s -- the kernels' opsPerIter
// is the byte count per outer iteration, so the shared emitCompute() math
// lands directly in GB/s.  Kernel bodies live in kernels/crypto_compute.h;
// per-ISA variants (e.g. AES-NI vs VAES-512) come from kernelMenu() like the
// compute tests.  The unit is "gbps" but the category is passed explicitly:
// categoryFromUnit() would otherwise file these under Bandwidth.

using clpeak_cpu::kernelMenu;

int CpuPeak::runCryptoAes(benchmark_config_t &cfg)
{
  emitVariants(*this, {"aes_encrypt", "AES-128 encrypt", "gbps", Category::Crypto},
               "aes", kernelMenu().aes, "no AES instructions on this CPU", cfg);
  return 0;
}

int CpuPeak::runCryptoSha256(benchmark_config_t &cfg)
{
  emitVariants(*this, {"sha256_hash", "SHA-256 hash", "gbps", Category::Crypto},
               "sha256", kernelMenu().sha256, "no SHA-256 instructions on this CPU", cfg);
  return 0;
}

int CpuPeak::runCryptoSha512(benchmark_config_t &cfg)
{
  // The x86 SHA512 EVEX extension (Arrow/Lunar Lake) is detected but has no
  // kernel yet, so x86 always reports the Unsupported row here.
  emitVariants(*this, {"sha512_hash", "SHA-512 hash", "gbps", Category::Crypto},
               "sha512", kernelMenu().sha512,
               "no SHA-512 instruction path on this CPU", cfg);
  return 0;
}

int CpuPeak::runCryptoCrc32c(benchmark_config_t &cfg)
{
  emitVariants(*this, {"crc32c", "CRC32-C checksum", "gbps", Category::Crypto},
               "crc32c", kernelMenu().crc32c,
               "no hardware CRC32 instruction on this CPU", cfg);
  return 0;
}

#endif // ENABLE_CPU
