#ifndef HYPRE_CUDA_ASSEMBLER_H
#define HYPRE_CUDA_ASSEMBLER_H

#include "HypreLinearSystem.h"

#ifdef KOKKOS_ENABLE_CUDA

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define THREADBLOCK_SIZE_SCAN 256
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#ifndef HYPRE_CUDA_ASSEMBLER_DEBUG
#define HYPRE_CUDA_ASSEMBLER_DEBUG
#endif // HYPRE_CUDA_ASSEMBLER_DEBUG
#undef HYPRE_CUDA_ASSEMBLER_DEBUG

#ifndef HYPRE_CUDA_ASSEMBLER_TIMER
#define HYPRE_CUDA_ASSEMBLER_TIMER
#endif // HYPRE_CUDA_ASSEMBLER_TIMER
#undef HYPRE_CUDA_ASSEMBLER_TIMER

#ifndef HYPRE_CUDA_ASSEMBLER_CUSTOM_SCAN
#define HYPRE_CUDA_ASSEMBLER_CUSTOM_SCAN
#endif // HYPRE_CUDA_ASSEMBLER_CUSTOM_SCAN
#undef HYPRE_CUDA_ASSEMBLER_CUSTOM_SCAN

#ifndef HYPRE_CUDA_ASSEMBLER_USE_CUB
#define HYPRE_CUDA_ASSEMBLER_USE_CUB
#endif // HYPRE_CUDA_ASSEMBLER_USE_CUB
//#undef HYPRE_CUDA_ASSEMBLER_USE_CUB

#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
#include <cub-1.8.0/cub/cub.cuh>
#endif

namespace sierra {
namespace nalu {

#define ASSEMBLER_CUDA_SAFE_CALL(call) do {                                                                    \
   cudaError_t err = call;                                                                                     \
   if (cudaSuccess != err) {                                                                                   \
      printf("CUDA ERROR (code = %d, %s) at %s:%d\n", err, cudaGetErrorString(err), __FILE__, __LINE__);       \
      exit(1);                                                                                                 \
   } } while(0)

int nextPowerOfTwo(int v);

struct saxpy_functor
{
  const HypreIntType nc_;
  saxpy_functor(HypreIntType nc) : nc_(nc) {}
  __host__ __device__
  HypreIntType operator()(const HypreIntType& x, const HypreIntType& y) const
  { 
    return nc_ * x + y;
  }
};

struct min_abs_diff
{
  const HypreIntType base_;
  min_abs_diff(HypreIntType base) : base_(base) {}
  __host__ __device__
  bool operator()(const HypreIntType& x, const HypreIntType& y) const
  { 
    return abs(x-base_) < abs(y-base_);
  }
};

struct lessThanOrdering64
{
  lessThanOrdering64() {}
  __host__ __device__
  bool operator()(const thrust::tuple<HypreIntType,double>& x, const thrust::tuple<HypreIntType,double>& y) const
  { 
    HypreIntType x1 = thrust::get<0>(x);
    double x2 = thrust::get<1>(x);
    HypreIntType y1 = thrust::get<0>(y);
    double y2 = thrust::get<1>(y);
   if (x1<y1) return true;
    else if (x1>y1) return false;
    else {
      if (abs(x2)<abs(y2)) return true;
      else return false;
    }
  }
};

  //typedef struct saxpy_functor;

  //typedef struct min_abs_diff;

  //typedef struct lessThanOrdering64;

inline __device__ HypreIntType scan1Inclusive(HypreIntType idata, volatile HypreIntType *shmem, cg::thread_block cta);

inline __device__ HypreIntType scan1Exclusive(HypreIntType idata, volatile HypreIntType *shmem, cg::thread_block cta);

__global__ void scanExclusiveShared(HypreIntType * d_Buf, HypreIntType * d_Dst, HypreIntType * d_Src, uint N);

__global__ void uniformUpdate(HypreIntType *d_Data, HypreIntType *d_Buffer, uint N);

__global__ void addKernel(const HypreIntType N, const HypreIntType * x, HypreIntType * y);

void exclusive_scan(HypreIntType *d_Dst, HypreIntType *d_Src, HypreIntType* d_work, uint N);

void inclusive_scan(HypreIntType *d_Dst, HypreIntType *d_Src, HypreIntType * d_work, uint N);

__global__ void sequenceKernel(const HypreIntType N, HypreIntType * x);

__global__ void absKernel(const HypreIntType N, const double * x, double *y);

template<typename TYPE>
__global__ void gatherKernel(const HypreIntType N, const HypreIntType * inds, const TYPE * x, TYPE * y);

__global__ void multiCooGatherKernel(const HypreIntType N, const HypreIntType * inds, 
				     const HypreIntType * rows_in, HypreIntType * rows_out,
				     const HypreIntType * cols_in, HypreIntType * cols_out,
				     const double * data_in, double * data_out);


/**********************************************************************************************************/
/*                            Matrix Specific CUDA Functions                                              */
/**********************************************************************************************************/

#define THREADBLOCK_SIZE 128

__global__ void matrixElemLocationsFinalKernel(const int N1, const HypreIntType N2, const HypreIntType * x,
						 const HypreIntType * y, HypreIntType * z);

__global__ void matrixElemLocationsInitialKernel(const HypreIntType * x, const HypreIntType N,
						 const HypreIntType M, HypreIntType * y, HypreIntType * z);

__global__ void fillRowStartExpandedKernel(const int num_rows, const HypreIntType * row_start,
					   HypreIntType * row_start_expanded);

__global__ void fillCSRMatrix(int num_rows, int threads_per_row, const HypreIntType * bins_in,
			      const HypreIntType * row_bins_in, const HypreIntType *cols_in, const double * data_in,
			      unsigned long long int * row_count_out, HypreIntType * cols_out, double * data_out);

__global__ void generateDenseIndexKeyKernel(const int num_cols, const HypreIntType * row_start,
					    const HypreIntType * rows, const HypreIntType * cols,
					    HypreIntType * dense_index_key);

__global__ void gatherColsDataKernel(const HypreIntType N, const HypreIntType * inds, 
				     const HypreIntType * cols_in, HypreIntType * cols_out,
				     const double * data_in, double * data_out);

void sortCooAscendingThrust(const HypreIntType N,
			    const HypreIntType num_rows,
			    const HypreIntType global_num_cols,
			    const HypreIntType * _d_kokkos_row_indices,
			    const HypreIntType * _d_kokkos_row_start,
			    void * _d_workspace,
			    HypreIntType * _d_cols,
			    double * _d_data);
  
void sortCooAscendingCub(const HypreIntType N,
			 const HypreIntType num_rows,
			 const HypreIntType global_num_cols,
			 const HypreIntType * _d_kokkos_row_indices,
			 const HypreIntType * _d_kokkos_row_start,
			 void * _d_workspace,
			 HypreIntType * _d_cols,
			 double * _d_data);

void sortCooThrust(const HypreIntType N,
		   const HypreIntType num_rows,
		   const HypreIntType global_num_cols,
		   const HypreIntType * _d_kokkos_row_indices,
		   const HypreIntType * _d_kokkos_row_start,
		   void * _d_workspace,
		   HypreIntType * _d_cols,
		   double * _d_data);

void sortCooCub(const HypreIntType N,
		const HypreIntType num_rows,
		const HypreIntType global_num_cols,
		const HypreIntType * _d_kokkos_row_indices,
		const HypreIntType * _d_kokkos_row_start,
		void * _d_workspace,
		HypreIntType * _d_cols,
		double * _d_data);

/**********************************************************************************************************/
/*                            RHS Specific CUDA Functions                                                 */
/**********************************************************************************************************/

__global__ void multiRhsGatherKernel(const HypreIntType N, const HypreIntType * inds, 
				     const HypreIntType * rows_in, HypreIntType * rows_out,
				     const double * data_in, double * data_out);

__global__ void fillRowsKernel(const int num_rows, const int threads_per_row, const HypreIntType * row_start,
			       const HypreIntType * row_indices, HypreIntType * rows);

__global__ void fillRhsVector(int num_rows, int threads_per_row, const HypreIntType * bins_in,
			      const double * data_in, double * data_out);

void sortRhsThrust(const HypreIntType N, HypreIntType * _d_rows, double * _d_data);

void sortRhsCub(const HypreIntType N, void * _d_workspace, HypreIntType * _d_rows, double * _d_data);

}  // nalu
}  // sierra

#endif // KOKKOS_ENABLE_CUDA

#endif /* HYPRE_CUDA_ASSEMBLER_H */
