#include "HypreLinearSystem.h"
#include "HypreCudaLinearSystemAssembler.h"

#ifdef KOKKOS_ENABLE_CUDA

namespace sierra {
namespace nalu {

#define MATRIX_ASSEMBLER_CUDA_SAFE_CALL(call) do {                                                             \
   cudaError_t err = call;                                                                                     \
   if (cudaSuccess != err) {                                                                                   \
      printf("CUDA ERROR (code = %d, %s) at %s:%d\n", err, cudaGetErrorString(err), __FILE__, __LINE__);       \
      exit(1);                                                                                                 \
   } } while(0)

int nextPowerOfTwo(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

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

#define THREADBLOCK_SIZE_SCAN 256
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

inline __device__ HypreIntType scan1Inclusive(HypreIntType idata, volatile HypreIntType *shmem, cg::thread_block cta) {
    uint pos = 2 * threadIdx.x - (threadIdx.x & (THREADBLOCK_SIZE_SCAN - 1));
    shmem[pos] = 0;
    pos += THREADBLOCK_SIZE_SCAN;
    shmem[pos] = idata;

    for (uint offset = 1; offset < THREADBLOCK_SIZE_SCAN; offset <<= 1) {
        cg::sync(cta);
        HypreIntType t = shmem[pos] + shmem[pos - offset];
        cg::sync(cta);
        shmem[pos] = t;
    }
    return shmem[pos];
}

inline __device__ HypreIntType scan1Exclusive(HypreIntType idata, volatile HypreIntType *shmem, cg::thread_block cta) {
    return scan1Inclusive(idata, shmem, cta) - idata;
}
////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(HypreIntType * d_Buf, HypreIntType * d_Dst, HypreIntType * d_Src, uint N) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ HypreIntType shmem[2 * THREADBLOCK_SIZE_SCAN];

    uint pos = blockIdx.x * THREADBLOCK_SIZE_SCAN + threadIdx.x;

    //Load data
    HypreIntType idata = 0;
    if (pos<N) idata = d_Src[pos];

    //Calculate exclusive scan
    HypreIntType odata = scan1Exclusive(idata, shmem, cta);

    //Write back
    if (pos<N) d_Dst[pos] = odata;
    if (threadIdx.x==THREADBLOCK_SIZE_SCAN-1 && pos<N) d_Buf[blockIdx.x] = odata+idata;
}


//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__global__ void uniformUpdate(HypreIntType *d_Data, HypreIntType *d_Buffer, uint N) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    uint pos = blockIdx.x * THREADBLOCK_SIZE_SCAN + threadIdx.x;
    __shared__ HypreIntType buf;
    if (threadIdx.x == 0) buf = d_Buffer[blockIdx.x];
    cg::sync(cta);

    if (pos<N) {
      HypreIntType data = d_Data[pos];
      d_Data[pos] = data+buf;
    }
}

void exclusive_scan(HypreIntType *d_Dst, HypreIntType *d_Src, HypreIntType* d_work, uint N)
{
  int nBlocks = (N + THREADBLOCK_SIZE_SCAN - 1) / THREADBLOCK_SIZE_SCAN;
  HypreIntType * d_Src_small=d_work, * d_Dst_small=d_work+N/4;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(d_Src_small, 0, nBlocks * sizeof(HypreIntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(d_Dst_small, 0, nBlocks * sizeof(HypreIntType)));
  
  /* scan the input vector */
  scanExclusiveShared<<<nBlocks, THREADBLOCK_SIZE_SCAN>>>(d_Src_small, d_Dst, d_Src, N);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /* recurse and call again */
  if (nBlocks>1) exclusive_scan(d_Dst_small, d_Src_small, d_work+N/2, nBlocks);

  if (nBlocks>1)
    uniformUpdate<<<nBlocks, THREADBLOCK_SIZE_SCAN>>>(d_Dst, d_Dst_small, N);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  
  return;
}

__global__ void addKernel(const HypreIntType N, const HypreIntType * x, HypreIntType * y) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) y[tid] += x[tid];
}

void inclusive_scan(HypreIntType *d_Dst, HypreIntType *d_Src, HypreIntType * d_work, uint N)
{
  exclusive_scan(d_Dst, d_Src, d_work, N);
  int num_threads=128;
  int num_blocks = (N + num_threads - 1)/num_threads;  
  addKernel<<<num_blocks,num_threads>>>(N,d_Src,d_Dst);
}


#define THREADBLOCK_SIZE 128

__global__ void binPointersFinalKernel(const int N1, const HypreIntType N2, const HypreIntType * x, const HypreIntType * y, HypreIntType * z) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  HypreIntType xx = 0;
  HypreIntType yy = 0;
  if (tid<=N2) {
    xx = x[tid];
    yy = y[tid];
    if (xx>0 && yy>0 && yy<=N1) z[yy] = (HypreIntType)xx;
  }
}

__global__ void binPointersKernel(const HypreIntType * x, const HypreIntType N, const HypreIntType M, HypreIntType * y, HypreIntType * z) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  
  /* load data into shmem */
  __shared__ HypreIntType shmem[THREADBLOCK_SIZE+1];
  int t = threadIdx.x;
  while (t<THREADBLOCK_SIZE+1 && blockIdx.x*blockDim.x+t<N) {
    shmem[t] = x[blockIdx.x*blockDim.x+t];
    t+=blockDim.x;
  }
  __syncthreads();

  /* Compute the differences and store. If on the very last entry, store N */
  if (tid<N-1) {
    y[tid+1] = (tid+1)*(shmem[threadIdx.x+1]>shmem[threadIdx.x]?1:0);
    z[tid+1] = shmem[threadIdx.x+1]>shmem[threadIdx.x]?1:0;
  } else if (tid==N-1) {
    y[tid+1]=N;
    z[tid+1]=1;
  }
}

__global__ void fillCSRMatrix(int num_rows, int threads_per_row, const HypreIntType * bins_in,
			      const HypreIntType * row_bins_in, const HypreIntType *cols_in, const double * data_in,
			      unsigned long long int * row_count_out, HypreIntType * cols_out, double * data_out) {
  
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;

  if (row<num_rows) {
   /* read the row pointers by the first thread in the warp */
    HypreIntType begin, end, rbegin;
    if (tid==0) {
      begin = bins_in[row];
      end = bins_in[row+1];
      rbegin = row_bins_in[begin];
    }
    /* broadcast across the warp */
    rbegin = __shfl_sync(0xffffffff, rbegin, 0, threads_per_row);
    begin = __shfl_sync(0xffffffff, begin, 0, threads_per_row);
    end = __shfl_sync(0xffffffff, end, 0, threads_per_row);
    
    /* This compute the loop size to the next multiple of threads_per_row */
    int roundUpSize = (((int)(end-begin) + threads_per_row - 1)/threads_per_row)*threads_per_row;
    double value=0.;
    double sum=0.;

    for (int t=tid; t<roundUpSize; t+=threads_per_row) {
      if (t>=end-begin) value=0.;
      else value = data_in[begin+t];
      for (int offset = threads_per_row/2; offset > 0; offset/=2)
      	value += __shfl_xor_sync(0xffffffff, value, offset, threads_per_row);      
      sum += value;
    }
    if (tid==0) {
      cols_out[row] = cols_in[begin];
      data_out[row] = sum;
      atomicAdd(row_count_out + rbegin,1);
    }
  }
}


__global__ void fillRhsVector(int num_rows, int threads_per_row, const HypreIntType * bins_in,
			      const double * data_in, double * data_out) {
  
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;
  
  if (row<num_rows) {
    /* read the row pointers by the first thread in the warp */
    HypreIntType begin, end;
    if (tid==0) {
      begin = bins_in[row];
      end = bins_in[row+1];
    }
    /* broadcast across the warp */
    begin = __shfl_sync(0xffffffff, begin, 0, threads_per_row);
    end = __shfl_sync(0xffffffff, end, 0, threads_per_row);
    
    /* This compute the loop size to the next multiple of threads_per_row */
    int roundUpSize = (((int)(end-begin) + threads_per_row - 1)/threads_per_row)*threads_per_row;
    double value=0.;
    double sum=0.;
    
    for (int t=tid; t<roundUpSize; t+=threads_per_row) {
      if (t>=end-begin) value=0.;
      else value = data_in[begin+t];
      for (int offset = threads_per_row/2; offset > 0; offset/=2)
      	value += __shfl_xor_sync(0xffffffff, value, offset, threads_per_row);      
      sum += value;
    }
    if (tid==0)
      data_out[row] = sum;
  }
}


__global__ void findDiagonalElementKernel(const int num_rows, const int threads_per_row, const int * rows, 
					  const HypreIntType * cols, int * colIndexForDiagonal) {
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;

  if (row<num_rows) {
    /* read the row pointers by the first thread in the warp */
    int rbegin, rend;
    if (tid==0) {
      rbegin = rows[row];
      rend = rows[row+1];
    }

    /* broadcast across the warp */
    rbegin = __shfl_sync(0xffffffff, rbegin, 0, threads_per_row);
    rend = __shfl_sync(0xffffffff, rend, 0, threads_per_row);

    /* This compute the loop size to the next multiple of threads_per_row */
    int roundUpSize = ((rend-rbegin + threads_per_row - 1)/threads_per_row)*threads_per_row;
    
    int colIndexForDiag=0;
    for (int t=tid; t<roundUpSize; t+=threads_per_row) {
      /* make a value for large threads that is guaranteed to be bigger than all others */
      int column = 2*num_rows;

      /* read the actual column for valid threads/columns */
      if (t<rend-rbegin) column = cols[rbegin+t];
      /* makt it absolute value so we can search for 0 */
      int val = abs(column-row);

      /* Try to find the location of the diagonal */
      colIndexForDiag = t;
      for (int offset = threads_per_row/2; offset > 0; offset/=2) {
      	int tmp1 = __shfl_down_sync(0xffffffff, val, offset, threads_per_row);
      	int tmp2 = __shfl_down_sync(0xffffffff, colIndexForDiag, offset, threads_per_row);
      	if (tmp1 < val) {
      	  val = tmp1;
      	  colIndexForDiag = tmp2;
      	}
      }
      /* broadcast in order to exit successfully for all threads in the warp */
      val = __shfl_sync(0xffffffff, val, 0, threads_per_row);
      if (val==0) break;
    }

    if (tid==0) {
      colIndexForDiagonal[row] = colIndexForDiag;
    }
  }
}

__global__ void shuffleDiagonalDLUKernel(const int num_rows, const int threads_per_row, const int * rows, 
					 const int * colIndexForDiagonal, HypreIntType * cols, double * values) {
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;

  if (row<num_rows) {
    /* read the row pointers by the first thread in the warp */
    int rbegin, rend, colIndex;
    if (tid==0) {
      rbegin = rows[row];
      rend = rows[row+1];
      colIndex = colIndexForDiagonal[row];
    }

    /* broadcast across the warp */
    rbegin = __shfl_sync(0xffffffff, rbegin, 0, threads_per_row);
    rend = __shfl_sync(0xffffffff, rend, 0, threads_per_row);
    colIndex = __shfl_sync(0xffffffff, colIndex, 0, threads_per_row);

    if (colIndex>0) {

      HypreIntType diag_column;
      double diag_value;
      if (tid==0) {
	diag_column = cols[rbegin+colIndex];
	diag_value = values[rbegin+colIndex];
      }

      /* This compute the loop size to the next multiple of threads_per_row */
      int roundUpSize = ((colIndex + threads_per_row - 1)/threads_per_row)*threads_per_row;

      int column;
      double value;
      for (int t=tid; t<roundUpSize; t+=threads_per_row) {
	int t1 = colIndex-1-t;
	if (t1>=0) {
	  value = values[rbegin+t1];
	  column = cols[rbegin+t1];
	  values[rbegin+t1+1] = value;
	  cols[rbegin+t1+1] = column;
	}    
      }
      /* write the column to the front of the row */
      if (tid==0) {
	values[rbegin] = diag_value;
	cols[rbegin] = diag_column;
      }
    }
  }
}


__global__ void shuffleDiagonalLDUKernel(const int num_rows, const int threads_per_row, const int * rows, 
					 const int * colIndexForDiagonal, HypreIntType * cols, double * values) {
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;

  if (row<num_rows) {
    /* read the row pointers by the first thread in the warp */
    int rbegin, rend, colIndex;
    if (tid==0) {
      rbegin = rows[row];
      rend = rows[row+1];
      colIndex = colIndexForDiagonal[row];
    }

    /* broadcast across the warp */
    rbegin = __shfl_sync(0xffffffff, rbegin, 0, threads_per_row);
    rend = __shfl_sync(0xffffffff, rend, 0, threads_per_row);
    colIndex = __shfl_sync(0xffffffff, colIndex, 0, threads_per_row);

    if (colIndex>0) {

      HypreIntType diag_column;
      double diag_value;
      if (tid==0) {
	diag_column = cols[rbegin];
	diag_value = values[rbegin];
      }

      /* This compute the loop size to the next multiple of threads_per_row */
      int roundUpSize = ((colIndex + threads_per_row - 1)/threads_per_row)*threads_per_row;

      int column;
      double value;
      for (int t=tid; t<roundUpSize; t+=threads_per_row) {
	if (t<colIndex) {
	  value = values[rbegin+t+1];
	  column = cols[rbegin+t+1];
	  values[rbegin+t] = value;
	  cols[rbegin+t] = column;
	}    
      }
      /* write the column to the front of the row */
      if (tid==0) {
	values[rbegin+colIndex] = diag_value;
	cols[rbegin+colIndex] = diag_column;
      }
    }
  }
}


#include <cub-1.8.0/cub/cub.cuh>

__global__ void sequenceKernel(const HypreIntType N, HypreIntType * x) {
  HypreIntType tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) x[tid] = tid;
}

__global__ void absKernel(const HypreIntType N, const double * x, double *y) {
  HypreIntType tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) y[tid] = abs(x[tid]);
}


__global__ void generateDenseIndexKeyKernel(const int num_cols, const HypreIntType * row_start,
					    const HypreIntType * rows, const HypreIntType * cols,
					    HypreIntType * dense_index_key) {
  
  int row_index = blockIdx.x;

  __shared__ HypreIntType shmem[2];
  __shared__ HypreIntType row;
  if (row_index<gridDim.x) {
    if (threadIdx.x<2)  shmem[threadIdx.x] = row_start[row_index+threadIdx.x];
    if (threadIdx.x==0) row = rows[row_index];
    __syncthreads();
    
    for (int t=threadIdx.x; t<shmem[1]-shmem[0]; t+=blockDim.x) {
      dense_index_key[shmem[0]+t] = row*num_cols + cols[shmem[0]+t];
    }
  }
}


template<typename TYPE>
__global__ void gatherKernel(const HypreIntType N, const HypreIntType * inds, const TYPE * x, TYPE * y) {
  HypreIntType tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) y[tid] = x[inds[tid]];
}

__global__ void multiCooGatherKernel(const HypreIntType N, const HypreIntType * inds, 
				     const HypreIntType * rows_in, HypreIntType * rows_out,
				     const HypreIntType * cols_in, HypreIntType * cols_out,
				     const double * data_in, double * data_out) {
  HypreIntType tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) {
    HypreIntType index = inds[tid];
    rows_out[tid] = rows_in[index];
    cols_out[tid] = cols_in[index];
    data_out[tid] = data_in[index];
  }
}

__global__ void gatherColsDataKernel(const HypreIntType N, const HypreIntType * inds, 
				     const HypreIntType * cols_in, HypreIntType * cols_out,
				     const double * data_in, double * data_out) {
  HypreIntType tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) {
    HypreIntType index = inds[tid];
    cols_out[tid] = cols_in[index];
    data_out[tid] = data_in[index];
  }
}

__global__ void multiRhsGatherKernel(const HypreIntType N, const HypreIntType * inds, 
				     const HypreIntType * rows_in, HypreIntType * rows_out,
				     const double * data_in, double * data_out) {
  HypreIntType tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) {
    HypreIntType index = inds[tid];
    rows_out[tid] = rows_in[index];
    data_out[tid] = data_in[index];
  }
}

template<typename IntType>
void sortCooAscendingThrust(const IntType N,
			    const IntType num_rows,
			    const IntType global_num_cols,
			    const IntType * _d_kokkos_row_indices,
			    const IntType * _d_kokkos_row_start,
			    void * _d_workspace,
			    IntType * _d_cols,
			    double * _d_data);
  
template<>
void sortCooAscendingThrust(const HypreIntType N,
			    const HypreIntType num_rows,
			    const HypreIntType global_num_cols,
			    const HypreIntType * _d_kokkos_row_indices,
			    const HypreIntType * _d_kokkos_row_start,
			    void * _d_workspace,
			    HypreIntType * _d_cols,
			    double * _d_data) {
  
  /* generate the dense index and sequence */
  HypreIntType * _d_dense_index_key = (HypreIntType *)(_d_workspace);
  int num_threads=128;
  int num_blocks = num_rows;
  generateDenseIndexKeyKernel<<<num_blocks,num_threads>>>(global_num_cols, _d_kokkos_row_start,
							  _d_kokkos_row_indices, _d_cols, _d_dense_index_key);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  thrust::device_ptr<HypreIntType> _d_key_ptr = thrust::device_pointer_cast(_d_dense_index_key);
  thrust::device_ptr<HypreIntType> _d_key_ptr_end = thrust::device_pointer_cast(_d_dense_index_key + N);
  thrust::device_ptr<double> _d_data_ptr = thrust::device_pointer_cast(_d_data);
  thrust::device_ptr<double> _d_data_ptr_end = thrust::device_pointer_cast(_d_data + N);
  thrust::device_ptr<HypreIntType> _d_cols_ptr = thrust::device_pointer_cast(_d_cols);

  typedef thrust::device_vector<HypreIntType>::iterator IntIterator;
  typedef thrust::device_vector<double>::iterator DoubleIterator;
  typedef thrust::tuple<IntIterator, DoubleIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator iter_begin(thrust::make_tuple(_d_key_ptr, _d_data_ptr));
  ZipIterator iter_end(thrust::make_tuple(_d_key_ptr_end, _d_data_ptr_end));
  thrust::stable_sort_by_key(thrust::device, iter_begin, iter_end, _d_cols_ptr, lessThanOrdering64());
}



template<typename IntType>
void sortCooAscendingCub(const IntType N,
			 const IntType num_rows,
			 const IntType global_num_cols,
			 const IntType * _d_kokkos_row_indices,
			 const IntType * _d_kokkos_row_start,
			 void * _d_workspace,
			 IntType * _d_cols,
			 double * _d_data);
  
template<>
void sortCooAscendingCub(const HypreIntType N,
			 const HypreIntType num_rows,
			 const HypreIntType global_num_cols,
			 const HypreIntType * _d_kokkos_row_indices,
			 const HypreIntType * _d_kokkos_row_start,
			 void * _d_workspace,
			 HypreIntType * _d_cols,
			 double * _d_data) {

  /* generate the absolute value key and the sequence */
  HypreIntType * _d_indices_out = (HypreIntType *)(_d_workspace);
  HypreIntType * _d_indices_in = (HypreIntType *)(_d_indices_out+N);
  double * _d_keys_abs_out = (double *)(_d_indices_in+N);
  double * _d_keys_abs_in = (double *)(_d_keys_abs_out+N);

  int num_threads = 128;
  int num_blocks = (N + num_threads - 1)/num_threads;  
  
  absKernel<<<num_blocks,num_threads>>>(N, _d_data, _d_keys_abs_in);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  sequenceKernel<<<num_blocks,num_threads>>>(N, _d_indices_in);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* Determine temporary device storage requirements ... already allocated in MemoryController */
  void * _d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_keys_abs_in, 
								  _d_keys_abs_out, _d_indices_in, _d_indices_out, N));

  /* point the temp storage to the already allocated memory */
  _d_temp_storage = (void *) (_d_keys_abs_in + N);

  /* Run sorting operation */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_keys_abs_in, 
								  _d_keys_abs_out, _d_indices_in, _d_indices_out, N));

  /* generate the dense index and sequence */
  HypreIntType * _d_dense_index_key = (HypreIntType *)(_d_indices_out + N);
  num_threads=128;
  num_blocks = num_rows;
  generateDenseIndexKeyKernel<<<num_blocks,num_threads>>>(global_num_cols, _d_kokkos_row_start,
							  _d_kokkos_row_indices, _d_cols, _d_dense_index_key);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /* now, reorder the rows, columns and data based on the indices */
  HypreIntType * _d_dense_index_key_temp = (HypreIntType *)(_d_dense_index_key + N);
  HypreIntType * _d_cols_temp = (HypreIntType *)(_d_dense_index_key_temp + N);
  double * _d_data_temp = (double *)(_d_cols_temp + N);
  multiCooGatherKernel<<<num_blocks,num_threads>>>(N, _d_indices_out, _d_dense_index_key, _d_dense_index_key_temp,
						   _d_cols, _d_cols_temp, _d_data, _d_data_temp);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_dense_index_key, _d_dense_index_key_temp, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_cols, _d_cols_temp, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, _d_data_temp, N*sizeof(double), cudaMemcpyDeviceToDevice));

  /***********************************************************************/
  /* At this point the data is sorted in absolute value double precision */
  /* Next, stable sort by the dense index : row*n_cols + col             */

  /* generate the dense index and sequence */
  HypreIntType * _d_dense_index_key_out = (HypreIntType *)(_d_workspace);
  HypreIntType * _d_dense_index_key_in = (HypreIntType *)(_d_dense_index_key_out+N);
  _d_indices_out = (HypreIntType *)(_d_dense_index_key_in+N);
  _d_indices_in = (HypreIntType *)(_d_indices_out+N);

  sequenceKernel<<<num_blocks,num_threads>>>(N, _d_indices_in);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* Determine temporary device storage requirements ... already allocated in MemoryController */
  _d_temp_storage = NULL;
  temp_storage_bytes = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_dense_index_key_in, 
								  _d_dense_index_key_out, _d_indices_in, _d_indices_out, N));

  /* point the temp storage to the already allocated memory */
  _d_temp_storage = (void *) (_d_indices_in + N);

  /* Run sorting operation on the dense index key */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_dense_index_key_in, 
								  _d_dense_index_key_out, _d_indices_in, _d_indices_out, N));
  
  /* now, reorder the rows, columns and data based on the new indices */
  _d_cols_temp = (HypreIntType *)(_d_indices_out + N);
  _d_data_temp = (double *)(_d_cols_temp + N);
  gatherColsDataKernel<<<num_blocks,num_threads>>>(N, _d_indices_out, _d_cols, _d_cols_temp, _d_data, _d_data_temp);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_cols, _d_cols_temp, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, _d_data_temp, N*sizeof(double), cudaMemcpyDeviceToDevice));
}

template<typename IntType>
void sortCooThrust(const IntType N,
		   const IntType num_rows,
		   const IntType global_num_cols,
		   const IntType * _d_kokkos_row_indices,
		   const IntType * _d_kokkos_row_start,
		   void * _d_workspace,
		   IntType * _d_cols,
		   double * _d_data);
  
template<>
void sortCooThrust(const HypreIntType N,
		   const HypreIntType num_rows,
		   const HypreIntType global_num_cols,
		   const HypreIntType * _d_kokkos_row_indices,
		   const HypreIntType * _d_kokkos_row_start,
		   void * _d_workspace,
		   HypreIntType * _d_cols,
		   double * _d_data) {
  
  /* generate the dense index and sequence */
  HypreIntType * _d_dense_index_key = (HypreIntType *)(_d_workspace);
  int num_threads=128;
  int num_blocks = num_rows;
  generateDenseIndexKeyKernel<<<num_blocks,num_threads>>>(global_num_cols, _d_kokkos_row_start,
							  _d_kokkos_row_indices, _d_cols, _d_dense_index_key);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  thrust::device_ptr<HypreIntType> _d_key_ptr = thrust::device_pointer_cast(_d_dense_index_key);
  thrust::device_ptr<HypreIntType> _d_key_ptr_end = thrust::device_pointer_cast(_d_dense_index_key + N);
  thrust::device_ptr<double> _d_data_ptr = thrust::device_pointer_cast(_d_data);
  thrust::device_ptr<HypreIntType> _d_cols_ptr = thrust::device_pointer_cast(_d_cols);

  typedef thrust::device_vector<HypreIntType>::iterator IntIterator;
  typedef thrust::device_vector<double>::iterator DoubleIterator;
  typedef thrust::tuple<IntIterator, DoubleIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator iter(thrust::make_tuple(_d_cols_ptr, _d_data_ptr));
  thrust::stable_sort_by_key(thrust::device, _d_key_ptr, _d_key_ptr_end, iter);
}


template<typename IntType>
void sortCooCub(const IntType N,
		const IntType num_rows,
		const IntType num_cols,
		const IntType * _d_kokkos_row_indices,
		const IntType * _d_kokkos_row_start,
		void * _d_workspace,
		IntType * _d_cols,
		double * _d_data);

template<>
void sortCooCub(const HypreIntType N,
		const HypreIntType num_rows,
		const HypreIntType global_num_cols,
		const HypreIntType * _d_kokkos_row_indices,
		const HypreIntType * _d_kokkos_row_start,
		void * _d_workspace,
		HypreIntType * _d_cols,
		double * _d_data) {
  
  HypreIntType * _d_dense_index_key_out = (HypreIntType *)(_d_workspace);
  HypreIntType * _d_indices_out = (HypreIntType *) (_d_dense_index_key_out+N);
  HypreIntType * _d_dense_index_key_in = (HypreIntType *)(_d_indices_out+N);
  HypreIntType * _d_indices_in = (HypreIntType *) (_d_dense_index_key_in+N);

  /* generate the dense index and sequence */
  int num_threads=128;
  int num_blocks = num_rows;
  generateDenseIndexKeyKernel<<<num_blocks,num_threads>>>(global_num_cols, _d_kokkos_row_start, 
							  _d_kokkos_row_indices, _d_cols, _d_dense_index_key_in);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /* generate the sequence for sorting */
  num_threads = 128;
  num_blocks = (N + num_threads - 1)/num_threads;  
  sequenceKernel<<<num_blocks,num_threads>>>(N, _d_indices_in);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* Determine temporary device storage requirements. Space has already been allocated in MemoryController */
  void * _d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_dense_index_key_in,
								  _d_dense_index_key_out, _d_indices_in, _d_indices_out, N));
  
  /* point the temp storage to the already allocated memory */
  _d_temp_storage = (void *) (_d_indices_in + N);

  /* Run sorting operation */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_dense_index_key_in,
								  _d_dense_index_key_out, _d_indices_in, _d_indices_out, N));

  /* now, reorder the rows, columns and data based on the new indices */
  HypreIntType * _d_cols_temp = (HypreIntType *)(_d_indices_out + N);
  double * _d_data_temp = (double *)(_d_cols_temp + N);
  gatherColsDataKernel<<<num_blocks,num_threads>>>(N, _d_indices_out, _d_cols, _d_cols_temp, _d_data, _d_data_temp);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_cols, _d_cols_temp, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, _d_data_temp, N*sizeof(double), cudaMemcpyDeviceToDevice));
}



__global__ void fillRowsKernel(const int num_rows, const int threads_per_row, const HypreIntType * row_start,
			       const HypreIntType * row_indices, HypreIntType * rows) {
  
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row_index = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;
  
  if (row_index<num_rows) {
    /* read the row pointers by the first thread in the warp */
    HypreIntType begin, end, row;
    if (tid==0) {
      begin = row_start[row_index];
      end = row_start[row_index+1];
      row = row_indices[row_index];
    }
    /* broadcast across the warp */
    begin = __shfl_sync(0xffffffff, begin, 0, threads_per_row);
    end = __shfl_sync(0xffffffff, end, 0, threads_per_row);
    row = __shfl_sync(0xffffffff, row, 0, threads_per_row);
    
    for (int t=tid; t<end-begin; t+=threads_per_row) {
      rows[begin+t] = row;
    }
  }
}

__global__ void fillRowStartExpandedKernel(const int num_rows, const HypreIntType * row_start,
					   HypreIntType * row_start_expanded) {
  
  int row_index = blockIdx.x;

  __shared__ HypreIntType shmem[2];
  if (row_index<num_rows) {
    if (threadIdx.x<2)
      shmem[threadIdx.x] = row_start[row_index+threadIdx.x];
    __syncthreads();
    
    for (int t=threadIdx.x; t<shmem[1]-shmem[0]; t+=blockDim.x) {
      row_start_expanded[shmem[0]+t] = row_index;
    }
  }
}


template<typename IntType>
void sortRhsThrust(const IntType N, IntType * _d_rows, double * _d_data);

template<>
void sortRhsThrust(const HypreIntType N, HypreIntType * _d_rows, double * _d_data) {  
  thrust::device_ptr<HypreIntType> _d_rows_ptr = thrust::device_pointer_cast(_d_rows);
  thrust::device_ptr<HypreIntType> _d_rows_ptr_end = thrust::device_pointer_cast(_d_rows + N);
  thrust::device_ptr<double> _d_data_ptr = thrust::device_pointer_cast(_d_data);
  thrust::device_ptr<double> _d_data_ptr_end = thrust::device_pointer_cast(_d_data + N);

  typedef thrust::device_vector<HypreIntType>::iterator IntIterator;
  typedef thrust::device_vector<double>::iterator DoubleIterator;
  typedef thrust::tuple<IntIterator, DoubleIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator iter_begin(thrust::make_tuple(_d_rows_ptr, _d_data_ptr));
  ZipIterator iter_end(thrust::make_tuple(_d_rows_ptr_end, _d_data_ptr_end));
  thrust::stable_sort(thrust::device, iter_begin, iter_end, lessThanOrdering64());
}

template<typename IntType>
void sortRhsCub(const IntType N,
			 void * _d_workspace,
			 IntType * _d_rows,
			 double * _d_data);

template<>
void sortRhsCub(const HypreIntType N,
			 void * _d_workspace,
			 HypreIntType * _d_rows,
			 double * _d_data) {

  /* generate the absolute value key and the sequence */
  HypreIntType * _d_indices_out = (HypreIntType *)(_d_workspace);
  HypreIntType * _d_indices_in = (HypreIntType *)(_d_indices_out + N);
  double * _d_keys_abs_out = (double *)(_d_indices_in + N);
  double * _d_keys_abs_in = (double *)(_d_keys_abs_out + N);

  int num_threads = 128;
  int num_blocks = (N + num_threads - 1)/num_threads;  

  absKernel<<<num_blocks,num_threads>>>(N, _d_data, _d_keys_abs_in);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  sequenceKernel<<<num_blocks,num_threads>>>(N, _d_indices_in);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* Determine temporary device storage requirements ... already allocated in MemoryController */
  void * _d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_keys_abs_in, 
								  _d_keys_abs_out, _d_indices_in, _d_indices_out, N));

  /* point the temp storage to the already allocated memory */
  _d_temp_storage = (void *) (_d_keys_abs_in + N);

  /* Run sorting operation */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_keys_abs_in, 
								  _d_keys_abs_out, _d_indices_in, _d_indices_out, N));

  /* now, reorder the rows, columns and data based on the indices */
  HypreIntType * _d_rows_temp = (HypreIntType *)(_d_indices_out + N);
  double * _d_data_temp = (double *)(_d_rows_temp + N);
  multiRhsGatherKernel<<<num_blocks,num_threads>>>(N, _d_indices_out, _d_rows, _d_rows_temp, _d_data, _d_data_temp);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rows, _d_rows_temp, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, _d_data_temp, N*sizeof(double), cudaMemcpyDeviceToDevice));

  /***********************************************************************/
  /* At this point the data is sorted in absolute value double precision */
  /* Next, stable sort by the rows                                       */

  /* generate the dense index and sequence */
  HypreIntType * _d_rows_in = (HypreIntType *)(_d_rows);
  HypreIntType * _d_rows_out = (HypreIntType *)(_d_workspace);
  _d_indices_out = (HypreIntType *)(_d_rows_out+N);
  _d_indices_in = (HypreIntType *)(_d_indices_out+N);

  /* generate indices for the permutation */
  sequenceKernel<<<num_blocks,num_threads>>>(N, _d_indices_in);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* Determine temporary device storage requirements ... already allocated in MemoryController */
  _d_temp_storage = NULL;
  temp_storage_bytes = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_rows_in, 
								  _d_rows_out, _d_indices_in, _d_indices_out, N));

  /* point the temp storage to the already allocated memory */
  _d_temp_storage = (void *) (_d_indices_in + N);

  /* Run sorting operation on the dense index key */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_rows_in, 
								  _d_rows_out, _d_indices_in, _d_indices_out, N));
  
  /* copy the indices to the front of the memory pool */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rows_in, _d_rows_out, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));

  /* now, reorder the rows, columns and data based on the new indices */
  _d_data_temp = (double *)(_d_indices_out + N);
  gatherKernel<double><<<num_blocks,num_threads>>>(N, _d_indices_out, _d_data, _d_data_temp);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, _d_data_temp, N*sizeof(double), cudaMemcpyDeviceToDevice));
}


template<typename IntType>
MemoryController<IntType>::MemoryController(std::string name, IntType N, int rank)
  : _name(name), _N(N), _rank(rank)
{
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : N=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_N);
#endif

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_USE_CUB
  void     *_d_tmp1 = NULL;
  double * _d_dtmp2 = NULL;
  IntType * _d_tmp3 = NULL;
  size_t temp_bytes1=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_tmp1, temp_bytes1, _d_dtmp2, _d_dtmp2, _d_tmp3, _d_tmp3, N));

  _d_tmp1 = NULL;
  IntType * _d_tmp4 = NULL;
  size_t temp_bytes2=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_tmp1, temp_bytes2, _d_tmp4, _d_tmp4, _d_tmp4, _d_tmp4, N));
  
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : bytes1=%lld, bytes2=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),(IntType)temp_bytes1,(IntType)temp_bytes2);
#endif

  IntType _radix_sort_bytes = (IntType)(temp_bytes1 > temp_bytes2 ? temp_bytes1 : temp_bytes2);
  IntType bytes1 = 4*_N * sizeof(IntType) + _radix_sort_bytes;
  IntType bytes2 = 4*(_N+1)*sizeof(IntType);
  IntType bytes = bytes1 > bytes2 ? bytes1 : bytes2;
#else
  IntType bytes = 4*(_N+1)*sizeof(IntType);
#endif
  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_workspace, bytes));
  _memoryUsed = bytes;
  
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  double free=0., total=0.;
  deviceMemoryInGBs(free,total);
  printf("Done Rank %d %s %s %d : name=%s : N=%lld, Used Memory (this class) GBs=%1.6lf,  Free Device Memory=%1.6lf,  TotalDeviceMemory=%1.6lf\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_N,memoryInGBs(),free,total);
#endif
}

template<typename IntType>
MemoryController<IntType>::~MemoryController() {
  if (_d_workspace) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_workspace)); _d_workspace=NULL; }
}

template<typename IntType>
double MemoryController<IntType>::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}

template<typename IntType>
void MemoryController<IntType>::deviceMemoryInGBs(double & free, double & total) const {
  size_t f=0, t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemGetInfo(&f, &t));
  free = 1.0*f/(1024.*1024.*1024.);
  total = 1.0*t/(1024.*1024.*1024.);
}

//////////////////////////////////////////////////////////////////
// Explicit template instantiation
template class MemoryController<HypreIntType>;



template<typename IntType>
MatrixAssembler<IntType>::MatrixAssembler(std::string name, bool sort, IntType iLower,
					  IntType iUpper, IntType jLower, IntType jUpper,
					  IntType global_num_rows, IntType global_num_cols, IntType nDataPtsToAssemble, int rank,
					  IntType num_rows, IntType * kokkos_row_indices, IntType * kokkos_row_start)
  : _name(name), _sort(sort), _iLower(iLower), _iUpper(iUpper),
    _jLower(jLower), _jUpper(jUpper), _global_num_rows(global_num_rows), _global_num_cols(global_num_cols),
    _nDataPtsToAssemble(nDataPtsToAssemble), _rank(rank), _num_rows(num_rows)
{
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld, iLower=%lld, iUpper=%lld, jLower=%lld, jUpper=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,_iLower,_iUpper,_jLower,_jUpper);
#endif

  _num_rows_this_rank = _iUpper+1-_iLower;
  _num_cols_this_rank = _jUpper+1-_jLower;

  /* allocate some space */
  _d_kokkos_row_indices = kokkos_row_indices;
  _d_kokkos_row_start = kokkos_row_start;

  /* Allocate these data structures now */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_row_indices, _num_rows*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_row_indices, _num_rows*sizeof(IntType)));
  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_row_counts, _num_rows*sizeof(unsigned long long int)));    
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_row_counts, _num_rows*sizeof(IntType)));    

  _memoryUsed += _num_rows*(sizeof(IntType) + sizeof(unsigned long long int));
    

  /* create events */
  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);
  _assembleTime=0.f;
  _xferTime=0.f;
  _xferHostTime=0.f;
  _nAssemble=0;
 
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  double free=0., total=0.;
  deviceMemoryInGBs(free,total);
  printf("Done Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf,  Free Device Memory=%1.6lf,  TotalDeviceMemory=%1.6lf \n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs(),free,total);
#endif
}


template<typename IntType>
MatrixAssembler<IntType>::~MatrixAssembler() {

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_TIMER
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
  if (_nAssemble>0)
    printf("Mean Symbolic/Numeric Assembly Time (%d samples)=%1.5f msec, Data Xfer Time From Kokkos=%1.5f msec, Data Xfer Time To Host=%1.5f msec\n",
	   _nAssemble,_assembleTime/_nAssemble,_xferTime/_nAssemble,_xferHostTime/_nAssemble);
#endif
  
  /* free the data */
  if (_d_col_index_for_diagonal) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_col_index_for_diagonal)); _d_col_index_for_diagonal=NULL; }

  /* csr matrix */
  if (_d_row_indices) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_row_indices)); _d_row_indices=NULL; }
  if (_d_row_counts) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_row_counts)); _d_row_counts=NULL; }
  if (_d_col_indices) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_col_indices)); _d_col_indices=NULL; }
  if (_d_values) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_values)); _d_values=NULL; }

  if (_h_row_indices) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_row_indices)); _h_row_indices=NULL; }
  if (_h_row_counts) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_row_counts)); _h_row_counts=NULL; }
  if (_h_col_indices) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_col_indices)); _h_col_indices=NULL; }
  if (_h_values) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_values)); _h_values=NULL; }

  /* create events */
  cudaEventDestroy(_start);
  cudaEventDestroy(_stop);
}

template<typename IntType>
double MatrixAssembler<IntType>::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}

template<typename IntType>
void MatrixAssembler<IntType>::deviceMemoryInGBs(double & free, double & total) const {
  size_t f=0, t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemGetInfo(&f, &t));
  free = 1.0*f/(1024.*1024.*1024.);
  total = 1.0*t/(1024.*1024.*1024.);
}

template<typename IntType>
void MatrixAssembler<IntType>::copySrcDataFromKokkos(IntType * cols, double * data) {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  _d_cols = cols;
  _d_data = data;
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferTime+=t;
}

template<typename IntType>
void MatrixAssembler<IntType>::setTemporaryDataArrayPtrs(IntType * d_workspace) {
  _d_workspace = d_workspace;
}

template<typename IntType>
void MatrixAssembler<IntType>::copyCSRMatrixToHost() {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_row_indices, _d_row_indices, _num_rows*sizeof(IntType), cudaMemcpyDeviceToHost));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_row_counts, _d_row_counts, _num_rows*sizeof(IntType), cudaMemcpyDeviceToHost));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_col_indices, _d_col_indices, _num_nonzeros*sizeof(IntType), cudaMemcpyDeviceToHost));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_values, _d_values, _num_nonzeros*sizeof(double), cudaMemcpyDeviceToHost));

  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}


template<typename IntType>
void MatrixAssembler<IntType>::copyOwnedCSRMatrixToHost() {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  /* Not sure what to do here yet */

  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}


template<typename IntType>
void MatrixAssembler<IntType>::copySharedCSRMatrixToHost() {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  /* Not sure what to do here yet */

  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}


template<typename IntType>
void MatrixAssembler<IntType>::assemble() {

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif
    
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  /* dense key */
  IntType * _d_key = _d_workspace;
  IntType * _d_matelem_bin_ptrs = _d_workspace + (_nDataPtsToAssemble+1);
  IntType * _d_locations = _d_workspace + 2*(_nDataPtsToAssemble+1);
  IntType * _d_matelem_bin_ptrs_final = _d_workspace + 3*(_nDataPtsToAssemble+1);
  
  if (_sort) {
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_USE_CUB
    sortCooAscendingCub<IntType>(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_kokkos_row_indices,
				 _d_kokkos_row_start, (void *)(_d_workspace), _d_cols, _d_data);
#else
    sortCooAscendingThrust<IntType>(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_kokkos_row_indices,
				    _d_kokkos_row_start, (void *)(_d_workspace), _d_cols, _d_data);
#endif
  } else {
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_USE_CUB
    sortCooCub<IntType>(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_kokkos_row_indices,
			_d_kokkos_row_start, (void *)(_d_workspace), _d_cols, _d_data);
#else
    sortCooThrust<IntType>(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_kokkos_row_indices,
			   _d_kokkos_row_start, (void *)(_d_workspace), _d_cols, _d_data);
#endif
  }
  CUDA_SAFE_CALL(cudaMemset(_d_workspace+_nDataPtsToAssemble, 0, (3*(_nDataPtsToAssemble+1)+1)*sizeof(IntType)));

  /************************************************************************************************/
  /* First : compute the bin pointers, the data structure used to reduce matrix elements          */
  /************************************************************************************************/
  
  /* Step 3 : Create the bin_ptrs vector by looking at differences between the key_sorted vector */
  int num_threads=128;
  int num_blocks = (_nDataPtsToAssemble + num_threads - 1)/num_threads;  
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : _nDataPtsToAssemble=%lld, num_blocks=%d\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,num_blocks);
#endif

  binPointersKernel<<<num_blocks,num_threads>>>(_d_key, _nDataPtsToAssemble, _global_num_cols, 
						_d_matelem_bin_ptrs, _d_locations);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* get this value now. d_temp is going to be written over later in the algorithm */
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  IntType key;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&key, _d_key, sizeof(IntType), cudaMemcpyDeviceToHost));
  printf("Rank %d %s %s %d : name=%s : key=%lld, global_num_cols=%lld, sizeof(IntType)=%u\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),key,_global_num_cols,sizeof(IntType));
#endif
  
  /* Step 4 : inclusive scan on the locations gives the relative positions of where to write the bin pointers
     Use custom kernel instead of thrust since it requires no additional memory. Thrust allocates on device */

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_CUSTOM_SCAN

  IntType * d_work = _d_workspace + 3*(_nDataPtsToAssemble+1);
  inclusive_scan(_d_workspace, _d_locations, d_work, _nDataPtsToAssemble+1);    
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_locations, _d_workspace, (_nDataPtsToAssemble+1)*sizeof(IntType), cudaMemcpyDeviceToDevice));

#else

  thrust::inclusive_scan(thrust::device, thrust::device_pointer_cast(_d_locations),
   			 thrust::device_pointer_cast(_d_locations+_nDataPtsToAssemble+1),
   			 thrust::device_pointer_cast(_d_locations));

#endif

  /* Step 5: get the value at the end of the scans. This is the number of num_nonzeros, num_rows */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&_num_nonzeros, _d_locations+_nDataPtsToAssemble, sizeof(IntType), cudaMemcpyDeviceToHost));    

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : _num_nonzeros=%lld, _num_rows=%lld\n",
    _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_num_nonzeros,_num_rows);
#endif

  /* Step 6 : Compute the final row pointers array */
  num_blocks = (_nDataPtsToAssemble + 1 + num_threads - 1)/num_threads;  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_matelem_bin_ptrs_final, 0, (_nDataPtsToAssemble+1)*sizeof(IntType)));
  binPointersFinalKernel<<<num_blocks,num_threads>>>(_num_nonzeros, _nDataPtsToAssemble,
						     _d_matelem_bin_ptrs, _d_locations, _d_matelem_bin_ptrs_final);  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /* just set pointers */
  _d_matelem_bin_ptrs = _d_matelem_bin_ptrs_final;

  /* Step 8 : allocate space */
  if (!_csrMatMemoryAdded) {
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_col_indices, _num_nonzeros*sizeof(IntType)));
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_values, _num_nonzeros*sizeof(double)));
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_col_indices, _num_nonzeros*sizeof(IntType)));
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_values, _num_nonzeros*sizeof(double)));
    _memoryUsed += (_num_nonzeros)*(sizeof(IntType) + sizeof(double));
    _csrMatMemoryAdded = true;
  }

  /* always reset this */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_row_counts, 0, _num_rows*sizeof(unsigned long long int)));

  /* copy from kokkos to internal data structure ... which will be manipulated later */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_row_indices, _d_kokkos_row_indices, _num_rows*sizeof(IntType), 
					     cudaMemcpyDeviceToDevice));

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  double free=0., total=0.;
  deviceMemoryInGBs(free,total);
  printf("Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf,  Free Device Memory=%1.6lf,  TotalDeviceMemory=%1.6lf \n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs(),free,total);
#endif

  /************************************************************************************************/
  /* Compute the CSR Matrix                                                                       */
  /************************************************************************************************/

  num_threads=128;
  num_blocks = _num_rows;
  HypreIntType * _d_kokkos_row_start_expanded = _d_workspace;
  fillRowStartExpandedKernel<<<num_blocks,num_threads>>>(_num_rows, _d_kokkos_row_start, _d_kokkos_row_start_expanded);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  
    
  /* Step 8 : reduce the array and create the "True" CSR matrix */
  num_threads=128;
  int threads_per_row = nextPowerOfTwo((_nDataPtsToAssemble + _num_nonzeros - 1)/_num_nonzeros);
  int num_rows_per_block = num_threads/threads_per_row;
  num_blocks = (_num_nonzeros + num_rows_per_block - 1)/num_rows_per_block;

  /* fill the matrix */
  fillCSRMatrix<<<num_blocks,num_threads>>>(_num_nonzeros, threads_per_row,
					    _d_matelem_bin_ptrs, _d_kokkos_row_start_expanded,
					    _d_cols, _d_data, _d_row_counts,
					    _d_col_indices, _d_values);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /**********************************************/
  /* split into owned and shared not-owned rows */

  /* Use the custom exclusive_scan version since thrust allocates memory on device */
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_CUSTOM_SCAN
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_workspace, 0, (_num_rows+1)*sizeof(IntType)));
  d_work = _d_workspace + _nDataPtsToAssemble+1;
  exclusive_scan(_d_workspace, (IntType *)_d_row_counts, d_work, _num_rows+1);    
#else
  thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(_d_row_counts),
   			 thrust::device_pointer_cast(_d_row_counts+_num_rows+1),
   			 thrust::device_pointer_cast(_d_workspace));
#endif

  /* Figure out the location of the lower boundary */
  thrust::device_ptr<IntType> iLowerLoc = thrust::min_element(thrust::device,
							      thrust::device_pointer_cast(_d_row_indices),
							      thrust::device_pointer_cast(_d_row_indices+_num_rows),
							      min_abs_diff(_iLower));
  IntType iLowLoc = thrust::raw_pointer_cast(iLowerLoc)-_d_row_indices;    
  
  /* Figure out the location of the upper boundary */
  thrust::device_ptr<IntType> iUpperLoc = thrust::min_element(thrust::device,
							      thrust::device_pointer_cast(_d_row_indices),
							      thrust::device_pointer_cast(_d_row_indices+_num_rows),
							      min_abs_diff(_iUpper));
  IntType iUppLoc = thrust::raw_pointer_cast(iUpperLoc)-_d_row_indices;    
    
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  IntType smallest_row = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&smallest_row, _d_row_indices, sizeof(IntType), cudaMemcpyDeviceToHost));
  IntType row_index_smallest = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&row_index_smallest, _d_row_indices+iLowLoc,
					     sizeof(IntType), cudaMemcpyDeviceToHost));
  printf("Rank %d %s %s %d : name=%s : smallest row=%lld, row index at iLowLoc=%lld, iLower=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),smallest_row,row_index_smallest,_iLower);
  
  IntType largest_row = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&largest_row, _d_row_indices+_num_rows-1, sizeof(IntType), cudaMemcpyDeviceToHost));
  IntType row_index_largest = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&row_index_largest, _d_row_indices+iUppLoc, sizeof(IntType), cudaMemcpyDeviceToHost));
  printf("Rank %d %s %s %d : name=%s : biggest row=%lld, row index at iUppLoc=%lld, iUpper=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),largest_row,row_index_largest,_iUpper);
#endif      

  if (_iLower==0) {

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
    printf("Rank %d %s %s %d : name=%s : CASE 1\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* need to throw an exception here as this is a catastrophic failure */
    assert(_num_rows_this_rank==iUppLoc+1);
      
    /* shared rhs vector exists under the following condition */
    if (iUppLoc+1<_num_rows) _has_shared = true;

    /* assert that this is always true */
    _num_rows_owned = _num_rows_this_rank;
    /* owned num_nonzeros is the value of the exclusive scan at iUppLoc */
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&_num_nonzeros_owned, _d_workspace+iUppLoc+1,
					       sizeof(IntType), cudaMemcpyDeviceToHost));

    /* set the device/host owned pointers */
    _d_row_indices_owned = _d_row_indices;
    _d_row_counts_owned = _d_row_counts;
    _d_col_indices_owned = _d_col_indices;
    _d_values_owned = _d_values;

    _h_row_indices_owned = _h_row_indices;
    _h_row_counts_owned = _h_row_counts;
    _h_col_indices_owned = _h_col_indices;
    _h_values_owned = _h_values;

    if (_has_shared) {
      /* shared num_rowss is the diff between the total and the number on this rank */
      _num_rows_shared = _num_rows - _num_rows_this_rank;
      /* shared num_nonzeros is the diff between the total and the number on this rank */
      _num_nonzeros_shared = _num_nonzeros - _num_nonzeros_owned;
      
      /* set the device/host shared pointers */
      _d_row_indices_shared = _d_row_indices + _num_rows_this_rank;
      _d_row_counts_shared = _d_row_counts + _num_rows_this_rank;
      _d_col_indices_shared = _d_col_indices+_num_nonzeros_owned;
      _d_values_shared = _d_values+_num_nonzeros_owned;
    
      _h_row_indices_shared = _h_row_indices + _num_rows_this_rank;
      _h_row_counts_shared = _h_row_counts + _num_rows_this_rank;
      _h_col_indices_shared = _h_col_indices+_num_nonzeros_owned;
      _h_values_shared = _h_values+_num_nonzeros_owned;
      
    } else {

      _num_rows_shared = 0;
      _num_nonzeros_shared = 0;

      _d_row_indices_shared = NULL;
      _d_row_counts_shared = NULL;
      _d_col_indices_shared = NULL;
      _d_values_shared = NULL;
      
      _h_row_indices_shared = NULL;
      _h_row_counts_shared = NULL;
      _h_col_indices_shared = NULL;
      _h_values_shared = NULL;
    }
    
  } else if (_iUpper+1==_global_num_rows) {

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
    printf("Rank %d %s %s %d : name=%s : CASE 2\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* need to throw an exception here as this is a catastrophic failure */
    assert(_num_rows_this_rank==_num_rows-iLowLoc);
    
    /* shared matrix exists under the following condition */
    if (iLowLoc>0) _has_shared = true;

    if (_has_shared) {
      /* shared num_rows is the diff between the total and the number on this rank */    
      _num_rows_shared = _num_rows - _num_rows_this_rank;
      /* shared num_nonzeros is the value of the exclusive scan at iLowLoc */
      MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&_num_nonzeros_shared, _d_workspace+iLowLoc,
						 sizeof(IntType), cudaMemcpyDeviceToHost));

      /* set the device/host shared pointers */
      _d_row_indices_shared = _d_row_indices;
      _d_row_counts_shared = _d_row_counts;
      _d_col_indices_shared = _d_col_indices;
      _d_values_shared = _d_values;
      
      _h_row_indices_shared = _h_row_indices;
      _h_row_counts_shared = _h_row_counts;
      _h_col_indices_shared = _h_col_indices;
      _h_values_shared = _h_values;

    } else {

      _num_rows_shared = 0;
      _num_nonzeros_shared = 0;

      _d_row_indices_shared = NULL;
      _d_row_counts_shared = NULL;
      _d_col_indices_shared = NULL;
      _d_values_shared = NULL;
      
      _h_row_indices_shared = NULL;
      _h_row_counts_shared = NULL;
      _h_col_indices_shared = NULL;
      _h_values_shared = NULL;
    }

    /* assert that this is always true */
    _num_rows_owned = _num_rows_this_rank;
    /* owned num_nonzeros is the diff between the total and the number on this rank */
    _num_nonzeros_owned = _num_nonzeros - _num_nonzeros_shared;

    /* set the device/host owned pointers */
    _d_row_indices_owned = _d_row_indices + _num_rows_shared;
    _d_row_counts_owned = _d_row_counts + _num_rows_shared;
    _d_col_indices_owned = _d_col_indices + _num_nonzeros_shared;
    _d_values_owned = _d_values + _num_nonzeros_shared;

    _h_row_indices_owned = _h_row_indices + _num_rows_shared;
    _h_row_counts_owned = _h_row_counts + _num_rows_shared;
    _h_col_indices_owned = _h_col_indices + _num_nonzeros_shared;
    _h_values_owned = _h_values + _num_nonzeros_shared;
    
  } else {

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
    printf("Rank %d %s %s %d : name=%s : CASE 3\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* need to throw an exception here as this is a catastrophic failure */
    assert(_num_rows_this_rank==iUppLoc+1-iLowLoc);

    /* shared matrix exists under the following condition */
    if (iLowLoc>0 || iUppLoc+1<_num_rows) _has_shared = true;

    IntType nnz_lower, nnz_upper;
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&nnz_lower, _d_workspace+iLowLoc, sizeof(IntType), cudaMemcpyDeviceToHost));
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&nnz_upper, _d_workspace+iUppLoc+1, sizeof(IntType), cudaMemcpyDeviceToHost));

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
    printf("Rank %d %s %s %d : name=%s : CASE 3 nnz_lower=%lld, nnz_upper=%lld\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),nnz_lower,nnz_upper);
#endif      

    /* assert that this is always true */
    _num_rows_owned = _num_rows_this_rank;
    _num_rows_shared = _num_rows - _num_rows_owned;

    _num_nonzeros_owned = nnz_upper - nnz_lower;
    _num_nonzeros_shared = _num_nonzeros - _num_nonzeros_owned;

#define SWAP(x1,x2,x3,n1,n2,n3,s) ( {					                          \
	IntType bytes1 = n1*s;						                          \
	IntType bytes2 = n2*s;						                          \
	MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x3, x1, bytes1, cudaMemcpyDeviceToDevice));    \
	MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x3+n3, x2, bytes2, cudaMemcpyDeviceToDevice)); \
	MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x1, x3+n3, bytes2, cudaMemcpyDeviceToDevice)); \
	MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x1+n2, x3, bytes1, cudaMemcpyDeviceToDevice)); \
      }									                          \
      )									                          \

    /* Only swap if there are shared entries above */
    if (iLowLoc>0) {
      IntType n1 = iLowLoc, n2=iUppLoc+1-iLowLoc, n3= _nDataPtsToAssemble;
      IntType s = sizeof(IntType);
      SWAP(_d_row_indices, _d_row_indices+iLowLoc, _d_workspace, n1, n2, n3, s);
      
      s = sizeof(unsigned long long int);
      SWAP(_d_row_counts, _d_row_counts+iLowLoc, _d_workspace, n1, n2, n3, s);
      
      n1 = nnz_lower, n2=nnz_upper-nnz_lower, n3= _nDataPtsToAssemble;
      s = sizeof(IntType);
      SWAP(_d_col_indices, _d_col_indices+nnz_lower, _d_workspace, n1, n2, n3, s);
      
      s = sizeof(double);
      SWAP(_d_values, _d_values+nnz_lower, _d_workspace, n1, n2, n3, s);
    }

    /* set the device/host owned pointers */
    _d_row_indices_owned = _d_row_indices;
    _d_row_counts_owned = _d_row_counts;
    _d_col_indices_owned = _d_col_indices;
    _d_values_owned = _d_values;

    _h_row_indices_owned = _h_row_indices;
    _h_row_counts_owned = _h_row_counts;
    _h_col_indices_owned = _h_col_indices;
    _h_values_owned = _h_values;

    /* set the device/host shared pointers */
    _d_row_indices_shared = _d_row_indices + _num_rows_this_rank;
    _d_row_counts_shared = _d_row_counts + _num_rows_this_rank;
    _d_col_indices_shared = _d_col_indices+_num_nonzeros_owned;
    _d_values_shared = _d_values+_num_nonzeros_owned;
    
    _h_row_indices_shared = _h_row_indices + _num_rows_this_rank;
    _h_row_counts_shared = _h_row_counts + _num_rows_this_rank;
    _h_col_indices_shared = _h_col_indices+_num_nonzeros_owned;
    _h_values_shared = _h_values+_num_nonzeros_owned;

  }
  

  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _assembleTime+=t;
  _nAssemble++;
}



template<typename IntType>
void MatrixAssembler<IntType>::reorderDLU() {
#if 0
  int num_threads=128;
  int threads_per_row = (_num_nonzeros + _num_rows - 1)/_num_rows;
  threads_per_row = std::min(nextPowerOfTwo(threads_per_row),32);    
  int num_rows_per_block = num_threads/threads_per_row;
  int num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;
  
  /* compute the location of the diagonal in each row */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_col_index_for_diagonal, _num_rows*sizeof(int)));
  findDiagonalElementKernel<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_row_offsets, 
							_d_col_indices, _d_col_index_for_diagonal);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  _col_index_determined = true;
  
  /* shuffle the rows to put the diagonal first */
  shuffleDiagonalDLUKernel<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_row_offsets, 
						       _d_col_index_for_diagonal, _d_col_indices, _d_values);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
#endif
}

template<typename IntType>
void MatrixAssembler<IntType>::reorderLDU() {
#if 0
  if (!_col_index_determined) {
    printf("column index for diagonal not computed. Must call reorderDLU first\n");
    return;
  }
  int num_threads=128;
  int threads_per_row = (_num_nonzeros + _num_rows - 1)/_num_rows;
  threads_per_row = std::min(nextPowerOfTwo(threads_per_row),32);    
  int num_rows_per_block = num_threads/threads_per_row;
  int num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;
  shuffleDiagonalLDUKernel<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_row_offsets, 
						       _d_col_index_for_diagonal, _d_col_indices, _d_values);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
#endif
}

//////////////////////////////////////////////////////////////////
// Explicit template instantiation
template class MatrixAssembler<HypreIntType>;


template<typename IntType>
RhsAssembler<IntType>::RhsAssembler(std::string name, bool sort, IntType iLower, IntType iUpper,
				    IntType global_num_rows, IntType nDataPtsToAssemble, int rank,
				    IntType num_rows, IntType * kokkos_row_indices, IntType * kokkos_row_start)
  : _name(name), _sort(sort), _iLower(iLower), _iUpper(iUpper),
    _global_num_rows(global_num_rows), _nDataPtsToAssemble(nDataPtsToAssemble), _rank(rank), _num_rows(num_rows)
{
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld, iLower=%lld, iUpper=%lld\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,_iLower,_iUpper);
#endif
  
  _num_rows_this_rank = _iUpper+1-_iLower;

  /* Step 7 : allocate space */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_rhs, _num_rows*sizeof(double)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_rhs_indices, _num_rows*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_rhs, _num_rows*sizeof(double)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_rhs_indices, _num_rows*sizeof(IntType)));
  _memoryUsed = _num_rows*(sizeof(IntType) + sizeof(double));
    
  /* allocate some space */
  _d_kokkos_row_indices = kokkos_row_indices;
  _d_kokkos_row_start = kokkos_row_start;
  
  /* create events */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_start));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_stop));
  _assembleTime=0.f;
  _xferTime=0.f;

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  double free=0., total=0.;
  deviceMemoryInGBs(free,total);
  printf("Done Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf,  Free Device Memory=%1.6lf,  TotalDeviceMemory=%1.6lf \n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs(),free,total);
#endif
}

template<typename IntType>
RhsAssembler<IntType>::~RhsAssembler() {

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_TIMER
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
  if (_nAssemble>0)
    printf("Mean RHS Assembly Time (%d samples)=%1.5f msec, Data Xfer Time From Kokkos=%1.5f msec, Data Xfer Time To Host=%1.5f msec\n",
	   _nAssemble,_assembleTime/_nAssemble,_xferTime/_nAssemble,_xferHostTime/_nAssemble);
#endif

  /* free the data */
  if (_d_rhs) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_rhs)); _d_rhs=NULL; }
  if (_h_rhs) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_rhs)); _d_rhs=NULL; }
  if (_d_rhs_indices) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_rhs_indices)); _d_rhs_indices=NULL; }
  if (_h_rhs_indices) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_rhs_indices)); _d_rhs_indices=NULL; }
  
  /* create events */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_start));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_stop));
}

template<typename IntType>
double RhsAssembler<IntType>::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}

template<typename IntType>
void RhsAssembler<IntType>::deviceMemoryInGBs(double & free, double & total) const {
  size_t f=0, t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemGetInfo(&f, &t));
  free = 1.0*f/(1024.*1024.*1024.);
  total = 1.0*t/(1024.*1024.*1024.);
}

template<typename IntType>
void RhsAssembler<IntType>::copySrcDataFromKokkos(double * data) {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  _d_data = data;

  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferTime+=t;
}

template<typename IntType>
void RhsAssembler<IntType>::setTemporaryDataArrayPtrs(IntType * d_workspace) {
  _d_workspace = d_workspace;
}

template<typename IntType>
void RhsAssembler<IntType>::copyRhsVectorToHost() {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_rhs_indices, _d_rhs_indices, _num_rows*sizeof(IntType), cudaMemcpyDeviceToHost));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_rhs, _d_rhs, _num_rows*sizeof(double), cudaMemcpyDeviceToHost));
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}

template<typename IntType>
void RhsAssembler<IntType>::copyOwnedRhsVectorToHost() {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  
  /* Not sure what to do here yet */
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}

template<typename IntType>
void RhsAssembler<IntType>::copySharedRhsVectorToHost() {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  
  /* Not sure what to do here yet */
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}

template<typename IntType>
void RhsAssembler<IntType>::assemble() {

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif

  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  /* Step 1 : sort if chosen */
  if (_sort) {
    int num_threads=128;
    int threads_per_row = nextPowerOfTwo((_nDataPtsToAssemble + _num_rows - 1)/_num_rows);
    //int threads_per_row = 1;
    int num_rows_per_block = num_threads/threads_per_row;
    int num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;
    HypreIntType * _d_rows = (HypreIntType *)(_d_workspace);
    fillRowsKernel<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_kokkos_row_start, _d_kokkos_row_indices, _d_rows);
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_USE_CUB
    void * _d_work = (void *)(_d_rows+_nDataPtsToAssemble);
    sortRhsCub<IntType>(_nDataPtsToAssemble, (void *) (_d_work), _d_rows, _d_data);
#else
    sortRhsThrust<IntType>(_nDataPtsToAssemble, _d_rows, _d_data);
#endif
  }

  /* copy from kokkos to internal data structure ... which will be manipulated later */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rhs_indices, _d_kokkos_row_indices, _num_rows*sizeof(IntType), cudaMemcpyDeviceToDevice));

  /* Step 2 : reduce the array and create the RHS Vector */
  int num_threads=128;
  int threads_per_row = nextPowerOfTwo((_nDataPtsToAssemble + _num_rows - 1)/_num_rows);
  //int threads_per_row = 1;
  int num_rows_per_block = num_threads/threads_per_row;
  int num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;

  fillRhsVector<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_kokkos_row_start, _d_data, _d_rhs);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /*******************************************************/
  /* Step 3 : split into owned and shared not-owned rows */
  /*******************************************************/

  /* Figure out the location of the lower boundary */
  thrust::device_ptr<IntType> iLowerLoc = thrust::min_element(thrust::device,
							      thrust::device_pointer_cast(_d_rhs_indices),
							      thrust::device_pointer_cast(_d_rhs_indices+_num_rows),
							      min_abs_diff(_iLower));
  IntType iLowLoc = thrust::raw_pointer_cast(iLowerLoc)-_d_rhs_indices;    
  
  /* Figure out the location of the upper boundary */
  thrust::device_ptr<IntType> iUpperLoc = thrust::min_element(thrust::device,
							      thrust::device_pointer_cast(_d_rhs_indices),
							      thrust::device_pointer_cast(_d_rhs_indices+_num_rows),
							      min_abs_diff(_iUpper));
  IntType iUppLoc = thrust::raw_pointer_cast(iUpperLoc)-_d_rhs_indices;    
    
#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  IntType smallest_row = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&smallest_row, _d_rhs_indices, sizeof(IntType), cudaMemcpyDeviceToHost));
  IntType row_index_smallest = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&row_index_smallest, _d_rhs_indices+iLowLoc,
					     sizeof(IntType), cudaMemcpyDeviceToHost));
  printf("Rank %d %s %s %d : name=%s : smallest row=%lld, row index at iLowLoc=%lld, iLower=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),smallest_row,row_index_smallest,_iLower);
  
  IntType largest_row = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&largest_row, _d_rhs_indices+_num_rows-1, sizeof(IntType), cudaMemcpyDeviceToHost));
  IntType row_index_largest = 0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&row_index_largest, _d_rhs_indices+iUppLoc, sizeof(IntType), cudaMemcpyDeviceToHost));
  printf("Rank %d %s %s %d : name=%s : biggest row=%lld, row index at iUppLoc=%lld, iUpper=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),largest_row,row_index_largest,_iUpper);
#endif      

  if (_iLower==0) {

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : CASE 1\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* need to throw an exception here as this is a catastrophic failure */
    assert(_num_rows_this_rank==iUppLoc+1);

    /* shared rhs vector exists under the following condition */
    if (iUppLoc+1<_num_rows) _has_shared = true;

    /* assert that this is always true */
    _num_rows_owned = _num_rows_this_rank;
    
    /* set the device/host owned pointers */
    _d_rhs_indices_owned = _d_rhs_indices;
    _d_rhs_owned = _d_rhs;
    _h_rhs_indices_owned = _h_rhs_indices;
    _h_rhs_owned = _h_rhs;
    
    /* shared rows is the diff between the total and the number on this rank */
    if (_has_shared) {
      _num_rows_shared = _num_rows - _num_rows_this_rank;
      
      /* set the device/host shared pointers */
      _d_rhs_indices_shared = _d_rhs_indices + _num_rows_owned;
      _d_rhs_shared = _d_rhs+_num_rows_owned;
      _h_rhs_indices_shared = _h_rhs_indices + _num_rows_owned;
      _h_rhs_shared = _h_rhs+_num_rows_owned;
    } else {
      _num_rows_shared = 0;
      _d_rhs_indices_shared = NULL;
      _d_rhs_shared = NULL;
      _h_rhs_indices_shared = NULL;
      _h_rhs_shared = NULL;
    }
    
  } else if (_iUpper+1==_global_num_rows) {

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : CASE 2\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* need to throw an exception here as this is a catastrophic failure */
    assert(_num_rows_this_rank==_num_rows-iLowLoc);
    
    /* shared rhs vector exists under the following condition */
    if (iLowLoc>0) _has_shared = true;
    
    /* shared rows is the diff between the total and the number on this rank */
    if (_has_shared) {
      _num_rows_shared = _num_rows - _num_rows_this_rank;

      /* set the device/host shared pointers */
      _d_rhs_indices_shared = _d_rhs_indices;
      _d_rhs_shared = _d_rhs;
      _h_rhs_indices_shared = _h_rhs_indices;
      _h_rhs_shared = _h_rhs;
    } else {
      _num_rows_shared = 0;
      _d_rhs_indices_shared = NULL;
      _d_rhs_shared = NULL;
      _h_rhs_indices_shared = NULL;
      _h_rhs_shared = NULL;
    }

    /* assert that this is always true */
    _num_rows_owned = _num_rows_this_rank;
    
    /* set the device/host owned pointers */
    _d_rhs_indices_owned = _d_rhs_indices + _num_rows_shared;
    _d_rhs_owned = _d_rhs + _num_rows_shared;
    _h_rhs_indices_owned = _h_rhs_indices + _num_rows_shared;
    _h_rhs_owned = _h_rhs + _num_rows_shared;
    
  } else {

#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : CASE 3\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* need to throw an exception here as this is a catastrophic failure */
    assert(_num_rows_this_rank==iUppLoc+1-iLowLoc);

    /* shared matrix exists under the following condition */
    if (iLowLoc>0 || iUppLoc+1<_num_rows) _has_shared = true;

    /* assert that this is always true */
    _num_rows_owned = _num_rows_this_rank;
    _num_rows_shared = _num_rows - _num_rows_owned;

#define SWAP_RHS(x1,x2,x3,n1,n2,n3,s) ( {					                  \
	IntType bytes1 = n1*s;						                          \
	IntType bytes2 = n2*s;						                          \
	MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x3, x1, bytes1, cudaMemcpyDeviceToDevice));    \
	MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x3+n3, x2, bytes2, cudaMemcpyDeviceToDevice)); \
	MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x1, x3+n3, bytes2, cudaMemcpyDeviceToDevice)); \
	MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x1+n2, x3, bytes1, cudaMemcpyDeviceToDevice)); \
      }									                          \
      )									                          \

    /* Only swap if there are shared entries above */
    if (iLowLoc>0) {
      IntType n1 = iLowLoc, n2=iUppLoc+1-iLowLoc, n3= _nDataPtsToAssemble;
      IntType s = sizeof(IntType);
      SWAP_RHS(_d_rhs_indices, _d_rhs_indices+iLowLoc, _d_workspace, n1, n2, n3, s);

      s = sizeof(double);
      SWAP_RHS(_d_rhs, _d_rhs+iLowLoc, _d_workspace, n1, n2, n3, s);
    }

    /* set the device/host owned pointers */
    _d_rhs_indices_owned = _d_rhs_indices;
    _d_rhs_owned = _d_rhs;

    _h_rhs_indices_owned = _h_rhs_indices;
    _h_rhs_owned = _h_rhs;

    _d_rhs_indices_shared = _d_rhs_indices_owned + _num_rows_this_rank;
    _d_rhs_shared = _d_rhs_owned + _num_rows_this_rank;

    _h_rhs_indices_shared = _h_rhs_indices_owned + _num_rows_this_rank;
    _h_rhs_shared = _h_rhs_owned + _num_rows_this_rank;

  }

  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _assembleTime+=t;
  _nAssemble++;
}

//////////////////////////////////////////////////////////////////
// Explicit template instantiation
template class RhsAssembler<HypreIntType>;


}  // nalu
}  // sierra

#endif
