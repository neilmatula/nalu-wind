#include "HypreCudaAssembler.h"

#ifdef KOKKOS_ENABLE_CUDA

namespace sierra {
namespace nalu {

  //typedef unsigned long long int ULLIntType;

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

#if 0
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
#endif

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
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(d_Src_small, 0, nBlocks * sizeof(HypreIntType)));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(d_Dst_small, 0, nBlocks * sizeof(HypreIntType)));
  
  /* scan the input vector */
  scanExclusiveShared<<<nBlocks, THREADBLOCK_SIZE_SCAN>>>(d_Src_small, d_Dst, d_Src, N);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /* recurse and call again */
  if (nBlocks>1) exclusive_scan(d_Dst_small, d_Src_small, d_work+N/2, nBlocks);

  if (nBlocks>1)
    uniformUpdate<<<nBlocks, THREADBLOCK_SIZE_SCAN>>>(d_Dst, d_Dst_small, N);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  
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

__global__ void sequenceKernel(const HypreIntType N, HypreIntType * x) {
  HypreIntType tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) x[tid] = tid;
}

__global__ void absKernel(const HypreIntType N, const double * x, double *y) {
  HypreIntType tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) y[tid] = abs(x[tid]);
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


/**********************************************************************************************************/
/*                            Matrix Specific CUDA Functions                                              */
/**********************************************************************************************************/

#define THREADBLOCK_SIZE 128

__global__ void matrixElemLocationsFinalKernel(const int N1, const HypreIntType N2, const HypreIntType * x, const HypreIntType * y, HypreIntType * z) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  HypreIntType xx = 0;
  HypreIntType yy = 0;
  if (tid<=N2) {
    xx = x[tid];
    yy = y[tid];
    if (xx>0 && yy>0 && yy<=N1) z[yy] = (HypreIntType)xx;
  }
}

__global__ void matrixElemLocationsInitialKernel(const HypreIntType * x, const HypreIntType N, const HypreIntType M, HypreIntType * y, HypreIntType * z) {
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
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

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
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

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

#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
  
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
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  sequenceKernel<<<num_blocks,num_threads>>>(N, _d_indices_in);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* Determine temporary device storage requirements ... already allocated in MemoryController */
  void * _d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_keys_abs_in, 
								  _d_keys_abs_out, _d_indices_in, _d_indices_out, N));

  /* point the temp storage to the already allocated memory */
  _d_temp_storage = (void *) (_d_keys_abs_in + N);

  /* Run sorting operation */
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_keys_abs_in, 
								  _d_keys_abs_out, _d_indices_in, _d_indices_out, N));

  /* generate the dense index and sequence */
  HypreIntType * _d_dense_index_key = (HypreIntType *)(_d_indices_out + N);
  num_threads=128;
  num_blocks = num_rows;
  generateDenseIndexKeyKernel<<<num_blocks,num_threads>>>(global_num_cols, _d_kokkos_row_start,
							  _d_kokkos_row_indices, _d_cols, _d_dense_index_key);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /* now, reorder the rows, columns and data based on the indices */
  HypreIntType * _d_dense_index_key_temp = (HypreIntType *)(_d_dense_index_key + N);
  HypreIntType * _d_cols_temp = (HypreIntType *)(_d_dense_index_key_temp + N);
  double * _d_data_temp = (double *)(_d_cols_temp + N);
  multiCooGatherKernel<<<num_blocks,num_threads>>>(N, _d_indices_out, _d_dense_index_key, _d_dense_index_key_temp,
						   _d_cols, _d_cols_temp, _d_data, _d_data_temp);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_dense_index_key, _d_dense_index_key_temp, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_cols, _d_cols_temp, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, _d_data_temp, N*sizeof(double), cudaMemcpyDeviceToDevice));

  /***********************************************************************/
  /* At this point the data is sorted in absolute value double precision */
  /* Next, stable sort by the dense index : row*n_cols + col             */

  /* generate the dense index and sequence */
  HypreIntType * _d_dense_index_key_out = (HypreIntType *)(_d_workspace);
  HypreIntType * _d_dense_index_key_in = (HypreIntType *)(_d_dense_index_key_out+N);
  _d_indices_out = (HypreIntType *)(_d_dense_index_key_in+N);
  _d_indices_in = (HypreIntType *)(_d_indices_out+N);

  sequenceKernel<<<num_blocks,num_threads>>>(N, _d_indices_in);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* Determine temporary device storage requirements ... already allocated in MemoryController */
  _d_temp_storage = NULL;
  temp_storage_bytes = 0;
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_dense_index_key_in, 
								  _d_dense_index_key_out, _d_indices_in, _d_indices_out, N));

  /* point the temp storage to the already allocated memory */
  _d_temp_storage = (void *) (_d_indices_in + N);

  /* Run sorting operation on the dense index key */
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_dense_index_key_in, 
								  _d_dense_index_key_out, _d_indices_in, _d_indices_out, N));
  
  /* now, reorder the rows, columns and data based on the new indices */
  _d_cols_temp = (HypreIntType *)(_d_indices_out + N);
  _d_data_temp = (double *)(_d_cols_temp + N);
  gatherColsDataKernel<<<num_blocks,num_threads>>>(N, _d_indices_out, _d_cols, _d_cols_temp, _d_data, _d_data_temp);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_cols, _d_cols_temp, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, _d_data_temp, N*sizeof(double), cudaMemcpyDeviceToDevice));
}


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
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /* generate the sequence for sorting */
  num_threads = 128;
  num_blocks = (N + num_threads - 1)/num_threads;  
  sequenceKernel<<<num_blocks,num_threads>>>(N, _d_indices_in);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* Determine temporary device storage requirements. Space has already been allocated in MemoryController */
  void * _d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_dense_index_key_in,
								  _d_dense_index_key_out, _d_indices_in, _d_indices_out, N));
  
  /* point the temp storage to the already allocated memory */
  _d_temp_storage = (void *) (_d_indices_in + N);

  /* Run sorting operation */
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_dense_index_key_in,
								  _d_dense_index_key_out, _d_indices_in, _d_indices_out, N));

  /* now, reorder the rows, columns and data based on the new indices */
  HypreIntType * _d_cols_temp = (HypreIntType *)(_d_indices_out + N);
  double * _d_data_temp = (double *)(_d_cols_temp + N);
  gatherColsDataKernel<<<num_blocks,num_threads>>>(N, _d_indices_out, _d_cols, _d_cols_temp, _d_data, _d_data_temp);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_cols, _d_cols_temp, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, _d_data_temp, N*sizeof(double), cudaMemcpyDeviceToDevice));
}

#endif // HYPRE_CUDA_ASSEMBLER_USE_CUB

/**********************************************************************************************************/
/*                            RHS Specific CUDA Functions                                                 */
/**********************************************************************************************************/

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

#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB

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
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  sequenceKernel<<<num_blocks,num_threads>>>(N, _d_indices_in);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* Determine temporary device storage requirements ... already allocated in MemoryController */
  void * _d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_keys_abs_in, 
							   _d_keys_abs_out, _d_indices_in, _d_indices_out, N));

  /* point the temp storage to the already allocated memory */
  _d_temp_storage = (void *) (_d_keys_abs_in + N);

  /* Run sorting operation */
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_keys_abs_in, 
								  _d_keys_abs_out, _d_indices_in, _d_indices_out, N));

  /* now, reorder the rows, columns and data based on the indices */
  HypreIntType * _d_rows_temp = (HypreIntType *)(_d_indices_out + N);
  double * _d_data_temp = (double *)(_d_rows_temp + N);
  multiRhsGatherKernel<<<num_blocks,num_threads>>>(N, _d_indices_out, _d_rows, _d_rows_temp, _d_data, _d_data_temp);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rows, _d_rows_temp, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, _d_data_temp, N*sizeof(double), cudaMemcpyDeviceToDevice));

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
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* Determine temporary device storage requirements ... already allocated in MemoryController */
  _d_temp_storage = NULL;
  temp_storage_bytes = 0;
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_rows_in, 
								  _d_rows_out, _d_indices_in, _d_indices_out, N));

  /* point the temp storage to the already allocated memory */
  _d_temp_storage = (void *) (_d_indices_in + N);

  /* Run sorting operation on the dense index key */
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_temp_storage, temp_storage_bytes, _d_rows_in, 
								  _d_rows_out, _d_indices_in, _d_indices_out, N));
  
  /* copy the indices to the front of the memory pool */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rows_in, _d_rows_out, N*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));

  /* now, reorder the rows, columns and data based on the new indices */
  _d_data_temp = (double *)(_d_indices_out + N);
  gatherKernel<double><<<num_blocks,num_threads>>>(N, _d_indices_out, _d_data, _d_data_temp);
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, _d_data_temp, N*sizeof(double), cudaMemcpyDeviceToDevice));
}

}  // nalu
}  // sierra

#endif // HYPRE_CUDA_ASSEMBLER_USE_CUB

#endif // KOKKOS_ENABLE_CUDA
