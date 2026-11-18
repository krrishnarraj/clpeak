#ifndef XPUPEAK_ONNX_THREADS_H
#define XPUPEAK_ONNX_THREADS_H

#ifdef ENABLE_ONNX

#include <vector>

struct OrtRuntime;
struct onnx_ep_info_t;

// Where the thread that calls Run() runs, beside ONNX Runtime's own CPU
// threads.
//
// The CPU provider runs its kernels on a session's intra-op pool.  By
// default the pool has one member per physical core, and the calling thread
// is one of them: a parallel kernel splits its work among the pool's
// threads and the caller.  On Linux the runtime pins each of the threads it
// creates to one core's logical CPUs, and leaves the first core for the
// caller, whose affinity it never sets -- its own documentation says the
// calling thread is the application's to place.  A caller left to the
// scheduler often shares a core with a pinned thread while the free core
// idles.  Then every split kernel waits for the two halves that share a
// core.
//
// On a Threadripper PRO 3955WX (ONNX Runtime 1.31.0) the pool pinned 15
// threads to cores 1-15.  One run left the caller on CPU 25, core 9's
// second logical CPU, for every fp32 rung: 692, 626 and 659 GFLOPS at
// 1024-, 2048- and 4096-wide layers.  With taskset, the caller read 676,
// 689 and 713 on core 9, and 1377, 1519 and 1535 on core 0.  Four runs left
// to the scheduler read fp32 at 963-1645 GFLOPS and int8 QDQ at 977-1668
// GOPS, peaking at square rungs in some runs and tall ones in others.  Five
// runs with this guard, alternated with them, read 1640-1656 and 1846-1850,
// each row at the same rung every time.
//
// OnnxCallerPin finds where the runtime put its threads rather than
// assuming it.  It builds one trivial session on the provider and reads the
// affinity of each thread the build created.  It then pins the calling
// thread to the CPUs those threads left free, for as long as it lives, and
// restores the thread's own mask when it goes.  It does nothing in these
// cases:
//
// - the provider is not a CPU one (a GPU's or an NPU's work is not on the
//   pool);
// - the runtime pinned none of the threads it created (macOS, Android, a
//   Windows client, a pool of one);
// - the pinned threads leave no CPU this thread may use (a taskset that
//   puts the caller on one of their cores).
//
// The guard covers Linux only.  ONNX Runtime also pins its threads on
// Windows Server, where the caller is still left to the scheduler.  What
// the guard found is logged at Info, so every run file says where the
// caller ran.
class OnnxCallerPin
{
public:
  OnnxCallerPin(const OrtRuntime &rt, const onnx_ep_info_t &ep);
  ~OnnxCallerPin();
  OnnxCallerPin(const OnnxCallerPin &) = delete;
  OnnxCallerPin &operator=(const OnnxCallerPin &) = delete;

private:
  // The CPUs this thread could use before it was pinned; empty while it is
  // not pinned.
  std::vector<int> restore;
};

#endif // ENABLE_ONNX

#endif // XPUPEAK_ONNX_THREADS_H
