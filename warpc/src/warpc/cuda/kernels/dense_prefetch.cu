// L2 prefetch: pulls the head of the next GEMV's weights into the 6 MB L2 with `prefetch.global.L2` while the preceding task runs and leaves DRAM idle.
// It reads nothing a kernel could observe and writes nothing, so a plan with or without it computes the same bits. Launched as a side-stream branch of
// the captured graph beside the producer; the consumer waits for it at the join.
//
//   ptr / bytes    the range, rounded down to whole 128-byte lines; consecutive lines by consecutive threads
//   spread_ns      0: issue as fast as the threads can; > 0: every thread issues its k-th line at start + k * spread_ns / (lines per thread), so the
//                  prefetch trickles through the producer's runtime instead of queueing a burst of DRAM requests in front of the producer's own loads
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void __launch_bounds__(256) k_prefetch(const char* p, unsigned lines, unsigned spread_ns) {
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
  if (first >= lines) return;
  const unsigned long long t0 = spread_ns ? gtimer() : 0;
  const unsigned per = (lines - first + stride - 1) / stride;  // lines this thread issues
  const unsigned long long step = spread_ns ? spread_ns / per : 0;
  unsigned k = 0;
  for (unsigned i = first; i < lines; i += stride, k++) {
    if (step) {
      const unsigned long long due = t0 + k * step;
      while (gtimer() < due) __nanosleep(20);
    }
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p + (size_t)i * 128));
  }
}

}  // namespace

extern "C" {

int warpc_prefetch(const void* ptr, size_t bytes, int blocks, int spread_ns, cudaStream_t st) {
  if (blocks < 1 || spread_ns < 0) return cudaErrorInvalidValue;
  k_prefetch<<<blocks, 256, 0, st>>>(reinterpret_cast<const char*>(ptr), (unsigned)(bytes / 128), (unsigned)spread_ns);
  return cudaGetLastError();
}

}  // extern "C"
