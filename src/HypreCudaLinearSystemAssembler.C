#include "HypreLinearSystem.h"
#include "HypreCudaLinearSystemAssembler.h"

#ifdef KOKKOS_ENABLE_CUDA

namespace sierra {
namespace nalu {

/* --------------------------------------------------------------------------------------------------------- */
/*                                     CUDA MemoryPool Assembler Class                                       */
/* --------------------------------------------------------------------------------------------------------- */


MemoryPool::MemoryPool(std::string name, HypreIntType N, int rank)
  : _name(name), _N(N), _rank(rank)
{
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : N=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_N);
#endif

#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
  void     *_d_tmp1 = NULL;
  double * _d_dtmp2 = NULL;
  HypreIntType * _d_tmp3 = NULL;
  size_t temp_bytes1=0;
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_tmp1, temp_bytes1, _d_dtmp2, _d_dtmp2, _d_tmp3, _d_tmp3, N));

  _d_tmp1 = NULL;
  HypreIntType * _d_tmp4 = NULL;
  size_t temp_bytes2=0;
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_tmp1, temp_bytes2, _d_tmp4, _d_tmp4, _d_tmp4, _d_tmp4, N));
  
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : bytes1=%lld, bytes2=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),(HypreIntType)temp_bytes1,(HypreIntType)temp_bytes2);
#endif

  HypreIntType _radix_sort_bytes = (HypreIntType)(temp_bytes1 > temp_bytes2 ? temp_bytes1 : temp_bytes2);
  HypreIntType bytes1 = 4*_N * sizeof(HypreIntType) + _radix_sort_bytes;
  HypreIntType bytes2 = 4*(_N+1)*sizeof(HypreIntType);
  HypreIntType bytes = bytes1 > bytes2 ? bytes1 : bytes2;
#else
  HypreIntType bytes = 4*(_N+1)*sizeof(HypreIntType);
#endif
  
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_workspace, bytes));
  _memoryUsed = bytes;
  
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  double free=0., total=0.;
  deviceMemoryInGBs(free,total);
  printf("Done Rank %d %s %s %d : name=%s : N=%lld, Used Memory (this class) GBs=%1.6lf,  Free Device Memory=%1.6lf,  TotalDeviceMemory=%1.6lf\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_N,memoryInGBs(),free,total);
#endif
}

MemoryPool::~MemoryPool() {
  if (_d_workspace) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_workspace)); _d_workspace=NULL; }
}

double MemoryPool::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}

void MemoryPool::deviceMemoryInGBs(double & free, double & total) const {
  size_t f=0, t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemGetInfo(&f, &t));
  free = 1.0*f/(1024.*1024.*1024.);
  total = 1.0*t/(1024.*1024.*1024.);
}



/* --------------------------------------------------------------------------------------------------------- */
/*                                     CUDA Matrix Assembler Class                                           */
/* --------------------------------------------------------------------------------------------------------- */


MatrixAssembler::MatrixAssembler(std::string name, bool sort, HypreIntType iLower,
				 HypreIntType iUpper, HypreIntType jLower, HypreIntType jUpper,
				 HypreIntType global_num_rows, HypreIntType global_num_cols,
				 HypreIntType nDataPtsToAssemble, int rank,
				 HypreIntType num_rows, HypreIntType * kokkos_row_indices, HypreIntType * kokkos_row_start)
  : _name(name), _sort(sort), _iLower(iLower), _iUpper(iUpper),
    _jLower(jLower), _jUpper(jUpper), _global_num_rows(global_num_rows), _global_num_cols(global_num_cols),
    _nDataPtsToAssemble(nDataPtsToAssemble), _rank(rank), _num_rows(num_rows)
{
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld, iLower=%lld, iUpper=%lld, jLower=%lld, jUpper=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,_iLower,_iUpper,_jLower,_jUpper);
#endif

  _num_rows_this_rank = _iUpper+1-_iLower;
  _num_cols_this_rank = _jUpper+1-_jLower;

  /* allocate some space */
  _d_kokkos_row_indices = kokkos_row_indices;
  _d_kokkos_row_start = kokkos_row_start;

  /* Allocate these data structures now */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_row_indices, _num_rows*sizeof(HypreIntType)));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_row_indices, _num_rows*sizeof(HypreIntType)));
  
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_row_counts, _num_rows*sizeof(unsigned long long int)));    
  ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_row_counts, _num_rows*sizeof(HypreIntType)));    

  _memoryUsed += _num_rows*(sizeof(HypreIntType) + sizeof(unsigned long long int));    

  /* create events */
  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);
  _assembleTime=0.f;
  _xferTime=0.f;
  _xferHostTime=0.f;
  _nAssemble=0;
 
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  double free=0., total=0.;
  deviceMemoryInGBs(free,total);
  printf("Done Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf,  Free Device Memory=%1.6lf,  TotalDeviceMemory=%1.6lf \n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs(),free,total);
#endif
}


MatrixAssembler::~MatrixAssembler() {

#ifdef HYPRE_CUDA_ASSEMBLER_TIMER
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
  if (_nAssemble>0)
    printf("Mean Symbolic/Numeric Assembly Time (%d samples)=%1.5f msec, Data Xfer Time From Kokkos=%1.5f msec, Data Xfer Time To Host=%1.5f msec\n",
	   _nAssemble,_assembleTime/_nAssemble,_xferTime/_nAssemble,_xferHostTime/_nAssemble);
#endif
  
  /* csr matrix */
  if (_d_row_indices) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_row_indices)); _d_row_indices=NULL; }
  if (_d_row_counts) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_row_counts)); _d_row_counts=NULL; }
  if (_d_col_indices) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_col_indices)); _d_col_indices=NULL; }
  if (_d_values) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_values)); _d_values=NULL; }

  if (_h_row_indices) { ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_row_indices)); _h_row_indices=NULL; }
  if (_h_row_counts) { ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_row_counts)); _h_row_counts=NULL; }
  if (_h_col_indices) { ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_col_indices)); _h_col_indices=NULL; }
  if (_h_values) { ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_values)); _h_values=NULL; }

  /* create events */
  cudaEventDestroy(_start);
  cudaEventDestroy(_stop);
}


double MatrixAssembler::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}


void MatrixAssembler::deviceMemoryInGBs(double & free, double & total) const {
  size_t f=0, t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemGetInfo(&f, &t));
  free = 1.0*f/(1024.*1024.*1024.);
  total = 1.0*t/(1024.*1024.*1024.);
}


void MatrixAssembler::setTemporaryDataArrayPtrs(HypreIntType * d_workspace) {
  _d_workspace = d_workspace;
}


void MatrixAssembler::copyCSRMatrixToHost() {
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_row_indices, _d_row_indices, _num_rows*sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_row_counts, _d_row_counts, _num_rows*sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_col_indices, _d_col_indices, _num_nonzeros*sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_values, _d_values, _num_nonzeros*sizeof(double), cudaMemcpyDeviceToHost));

  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}


void MatrixAssembler::copyOwnedCSRMatrixToHost() {
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  /* Not sure what to do here yet */

  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}


void MatrixAssembler::copySharedCSRMatrixToHost() {
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  /* Not sure what to do here yet */

  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}


void MatrixAssembler::assemble(Kokkos::View<HypreIntType *>& cols, Kokkos::View<double *>& data) {

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif


  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  _d_cols = cols.data();
  _d_data = data.data();
  
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferTime+=t;

    
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  /* dense key */
  HypreIntType * _d_key = _d_workspace;
  HypreIntType * _d_matelem_bin_ptrs = _d_workspace + (_nDataPtsToAssemble+1);
  HypreIntType * _d_locations = _d_workspace + 2*(_nDataPtsToAssemble+1);
  HypreIntType * _d_matelem_bin_ptrs_final = _d_workspace + 3*(_nDataPtsToAssemble+1);
  
  if (_sort) {
#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
    sortCooAscendingCub(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_kokkos_row_indices,
				 _d_kokkos_row_start, (void *)(_d_workspace), _d_cols, _d_data);
#else
    sortCooAscendingThrust(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_kokkos_row_indices,
				    _d_kokkos_row_start, (void *)(_d_workspace), _d_cols, _d_data);
#endif
  } else {
#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
    sortCooCub(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_kokkos_row_indices,
			_d_kokkos_row_start, (void *)(_d_workspace), _d_cols, _d_data);
#else
    sortCooThrust(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_kokkos_row_indices,
			   _d_kokkos_row_start, (void *)(_d_workspace), _d_cols, _d_data);
#endif
  }
  CUDA_SAFE_CALL(cudaMemset(_d_workspace+_nDataPtsToAssemble, 0, (3*(_nDataPtsToAssemble+1)+1)*sizeof(HypreIntType)));

  /************************************************************************************************/
  /* First : compute the bin pointers, the data structure used to reduce matrix elements          */
  /************************************************************************************************/
  
  /* Step 3 : Create the bin_ptrs vector by looking at differences between the key_sorted vector */
  int num_threads=128;
  int num_blocks = (_nDataPtsToAssemble + num_threads - 1)/num_threads;  
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : _nDataPtsToAssemble=%lld, num_blocks=%d\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,num_blocks);
#endif

  matrixElemLocationsInitialKernel<<<num_blocks,num_threads>>>(_d_key, _nDataPtsToAssemble, _global_num_cols, 
						_d_matelem_bin_ptrs, _d_locations);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /* get this value now. d_temp is going to be written over later in the algorithm */
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  HypreIntType key;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&key, _d_key, sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  printf("Rank %d %s %s %d : name=%s : key=%lld, global_num_cols=%lld, sizeof(HypreIntType)=%u\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),key,_global_num_cols,sizeof(HypreIntType));
#endif
  
  /* Step 4 : inclusive scan on the locations gives the relative positions of where to write the bin pointers
     Use custom kernel instead of thrust since it requires no additional memory. Thrust allocates on device */

#ifdef HYPRE_CUDA_ASSEMBLER_CUSTOM_SCAN

  HypreIntType * d_work = _d_workspace + 3*(_nDataPtsToAssemble+1);
  inclusive_scan(_d_workspace, _d_locations, d_work, _nDataPtsToAssemble+1);    
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_locations, _d_workspace, (_nDataPtsToAssemble+1)*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));

#else

  thrust::inclusive_scan(thrust::device, thrust::device_pointer_cast(_d_locations),
   			 thrust::device_pointer_cast(_d_locations+_nDataPtsToAssemble+1),
   			 thrust::device_pointer_cast(_d_locations));

#endif

  /* Step 5: get the value at the end of the scans. This is the number of num_nonzeros, num_rows */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&_num_nonzeros, _d_locations+_nDataPtsToAssemble, sizeof(HypreIntType), cudaMemcpyDeviceToHost));    

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : _num_nonzeros=%lld, _num_rows=%lld\n",
    _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_num_nonzeros,_num_rows);
#endif

  /* Step 6 : Compute the final row pointers array */
  num_blocks = (_nDataPtsToAssemble + 1 + num_threads - 1)/num_threads;  
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_matelem_bin_ptrs_final, 0, (_nDataPtsToAssemble+1)*sizeof(HypreIntType)));
  matrixElemLocationsFinalKernel<<<num_blocks,num_threads>>>(_num_nonzeros, _nDataPtsToAssemble,
						     _d_matelem_bin_ptrs, _d_locations, _d_matelem_bin_ptrs_final);  
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /* just set pointers */
  _d_matelem_bin_ptrs = _d_matelem_bin_ptrs_final;

  /* Step 8 : allocate space */
  if (!_csrMatMemoryAdded) {
    ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_col_indices, _num_nonzeros*sizeof(HypreIntType)));
    ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_values, _num_nonzeros*sizeof(double)));
    ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_col_indices, _num_nonzeros*sizeof(HypreIntType)));
    ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_values, _num_nonzeros*sizeof(double)));
    _memoryUsed += (_num_nonzeros)*(sizeof(HypreIntType) + sizeof(double));
    _csrMatMemoryAdded = true;
  }

  /* always reset this */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_row_counts, 0, _num_rows*sizeof(unsigned long long int)));

  /* copy from kokkos to internal data structure ... which will be manipulated later */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_row_indices, _d_kokkos_row_indices, _num_rows*sizeof(HypreIntType), 
					     cudaMemcpyDeviceToDevice));

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
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
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  
    
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
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /**********************************************/
  /* split into owned and shared not-owned rows */

  /* Use the custom exclusive_scan version since thrust allocates memory on device */
#ifdef HYPRE_CUDA_ASSEMBLER_CUSTOM_SCAN
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_workspace, 0, (_num_rows+1)*sizeof(HypreIntType)));
  d_work = _d_workspace + _nDataPtsToAssemble+1;
  exclusive_scan(_d_workspace, (HypreIntType *)_d_row_counts, d_work, _num_rows+1);    
#else
  thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(_d_row_counts),
   			 thrust::device_pointer_cast(_d_row_counts+_num_rows+1),
   			 thrust::device_pointer_cast(_d_workspace));
#endif

  /* Figure out the location of the lower boundary */
  thrust::device_ptr<HypreIntType> iLowerLoc = thrust::min_element(thrust::device,
							      thrust::device_pointer_cast(_d_row_indices),
							      thrust::device_pointer_cast(_d_row_indices+_num_rows),
							      min_abs_diff(_iLower));
  HypreIntType iLowLoc = thrust::raw_pointer_cast(iLowerLoc)-_d_row_indices;    
  
  /* Figure out the location of the upper boundary */
  thrust::device_ptr<HypreIntType> iUpperLoc = thrust::min_element(thrust::device,
							      thrust::device_pointer_cast(_d_row_indices),
							      thrust::device_pointer_cast(_d_row_indices+_num_rows),
							      min_abs_diff(_iUpper));
  HypreIntType iUppLoc = thrust::raw_pointer_cast(iUpperLoc)-_d_row_indices;    
    
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  HypreIntType smallest_row = 0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&smallest_row, _d_row_indices, sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  HypreIntType row_index_smallest = 0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&row_index_smallest, _d_row_indices+iLowLoc,
					     sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  printf("Rank %d %s %s %d : name=%s : smallest row=%lld, row index at iLowLoc=%lld, iLower=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),smallest_row,row_index_smallest,_iLower);
  
  HypreIntType largest_row = 0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&largest_row, _d_row_indices+_num_rows-1, sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  HypreIntType row_index_largest = 0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&row_index_largest, _d_row_indices+iUppLoc, sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  printf("Rank %d %s %s %d : name=%s : biggest row=%lld, row index at iUppLoc=%lld, iUpper=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),largest_row,row_index_largest,_iUpper);
#endif      

  /* shared num_nonzeros is the value of the exclusive scan at iLowLoc */
  HypreIntType nnz_lower=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&nnz_lower, _d_workspace+iLowLoc,
					     sizeof(HypreIntType), cudaMemcpyDeviceToHost));

  /* owned num_nonzeros is the value of the exclusive scan at iUppLoc */
  HypreIntType nnz_upper=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&nnz_upper, _d_workspace+iUppLoc+1,
					     sizeof(HypreIntType), cudaMemcpyDeviceToHost));

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : nnz_lower=%lld, nnz_upper=%lld\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),nnz_lower,nnz_upper);
#endif      

  /* Compute owned/shared meta data */
  _num_rows_owned = _num_rows_this_rank;
  _num_rows_shared = _num_rows - _num_rows_owned;

  _num_nonzeros_owned = nnz_upper - nnz_lower;
  _num_nonzeros_shared = _num_nonzeros - _num_nonzeros_owned;

  /*******************************************************/
  /* Consistency check : make sure things are consistent */
  /*******************************************************/
  if (_num_rows_owned!=_num_rows && _num_nonzeros_owned!=_num_nonzeros)
    _has_shared = true;
  else if ((_num_rows_owned!=_num_rows && _num_nonzeros_owned==_num_nonzeros) ||
	   (_num_rows_owned==_num_rows && _num_nonzeros_owned!=_num_nonzeros)) {
    /* This is inconsistent. Need to throw an exception */
    throw std::runtime_error("Inconsistency detected. This should not happen.");
  } else {
    _has_shared=false;
    return;
  }

  if (_iLower==0) {

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
    printf("Rank %d %s %s %d : name=%s : CASE 1\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

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
      
    
  } else if (_iUpper+1==_global_num_rows) {

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
    printf("Rank %d %s %s %d : name=%s : CASE 2\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* set the device/host shared pointers */
    _d_row_indices_shared = _d_row_indices;
    _d_row_counts_shared = _d_row_counts;
    _d_col_indices_shared = _d_col_indices;
    _d_values_shared = _d_values;
    
    _h_row_indices_shared = _h_row_indices;
    _h_row_counts_shared = _h_row_counts;
    _h_col_indices_shared = _h_col_indices;
    _h_values_shared = _h_values;

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

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
    printf("Rank %d %s %s %d : name=%s : CASE 3\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

#define SWAP(x1,x2,x3,n1,n2,n3,s) ( {					                   \
	HypreIntType bytes1 = n1*s;						           \
	HypreIntType bytes2 = n2*s;						           \
	ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x3, x1, bytes1, cudaMemcpyDeviceToDevice));    \
	ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x3+n3, x2, bytes2, cudaMemcpyDeviceToDevice)); \
	ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x1, x3+n3, bytes2, cudaMemcpyDeviceToDevice)); \
	ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x1+n2, x3, bytes1, cudaMemcpyDeviceToDevice)); \
      }									                   \
      )									                   \

    /* Only swap if there are shared entries above */
    if (iLowLoc>0) {
      HypreIntType n1 = iLowLoc, n2=iUppLoc+1-iLowLoc, n3= _nDataPtsToAssemble;
      HypreIntType s = sizeof(HypreIntType);
      SWAP(_d_row_indices, _d_row_indices+iLowLoc, _d_workspace, n1, n2, n3, s);
      
      s = sizeof(unsigned long long int);
      SWAP(_d_row_counts, _d_row_counts+iLowLoc, _d_workspace, n1, n2, n3, s);
      
      n1 = nnz_lower, n2=nnz_upper-nnz_lower, n3= _nDataPtsToAssemble;
      s = sizeof(HypreIntType);
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
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _assembleTime+=t;
  _nAssemble++;
}


/* --------------------------------------------------------------------------------------------------------- */
/*                                     CUDA RHS Assembler Class                                              */
/* --------------------------------------------------------------------------------------------------------- */


RhsAssembler::RhsAssembler(std::string name, bool sort, HypreIntType iLower, HypreIntType iUpper,
			   HypreIntType global_num_rows, HypreIntType nDataPtsToAssemble, int rank,
			   HypreIntType num_rows, HypreIntType * kokkos_row_indices, HypreIntType * kokkos_row_start)
  : _name(name), _sort(sort), _iLower(iLower), _iUpper(iUpper), _global_num_rows(global_num_rows),
    _nDataPtsToAssemble(nDataPtsToAssemble), _rank(rank), _num_rows(num_rows)
{
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld, iLower=%lld, iUpper=%lld\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,_iLower,_iUpper);
#endif
  
  _num_rows_this_rank = _iUpper+1-_iLower;

  /* Step 7 : allocate space */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_rhs, _num_rows*sizeof(double)));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_rhs_indices, _num_rows*sizeof(HypreIntType)));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_rhs, _num_rows*sizeof(double)));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_rhs_indices, _num_rows*sizeof(HypreIntType)));
  _memoryUsed = _num_rows*(sizeof(HypreIntType) + sizeof(double));
    
  /* allocate some space */
  _d_kokkos_row_indices = kokkos_row_indices;
  _d_kokkos_row_start = kokkos_row_start;
  
  /* create events */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_start));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_stop));
  _assembleTime=0.f;
  _xferTime=0.f;

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  double free=0., total=0.;
  deviceMemoryInGBs(free,total);
  printf("Done Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf,  Free Device Memory=%1.6lf,  TotalDeviceMemory=%1.6lf \n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs(),free,total);
#endif
}

RhsAssembler::~RhsAssembler() {

#ifdef HYPRE_CUDA_ASSEMBLER_TIMER
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
  if (_nAssemble>0)
    printf("Mean RHS Assembly Time (%d samples)=%1.5f msec, Data Xfer Time From Kokkos=%1.5f msec, Data Xfer Time To Host=%1.5f msec\n",
	   _nAssemble,_assembleTime/_nAssemble,_xferTime/_nAssemble,_xferHostTime/_nAssemble);
#endif

  /* free the data */
  if (_d_rhs) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_rhs)); _d_rhs=NULL; }
  if (_h_rhs) { ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_rhs)); _d_rhs=NULL; }
  if (_d_rhs_indices) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_rhs_indices)); _d_rhs_indices=NULL; }
  if (_h_rhs_indices) { ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_rhs_indices)); _d_rhs_indices=NULL; }
  
  /* create events */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_start));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_stop));
}

double RhsAssembler::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}

void RhsAssembler::deviceMemoryInGBs(double & free, double & total) const {
  size_t f=0, t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemGetInfo(&f, &t));
  free = 1.0*f/(1024.*1024.*1024.);
  total = 1.0*t/(1024.*1024.*1024.);
}

void RhsAssembler::setTemporaryDataArrayPtrs(HypreIntType * d_workspace) {
  _d_workspace = d_workspace;
}

void RhsAssembler::copyRhsVectorToHost() {
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_rhs_indices, _d_rhs_indices, _num_rows*sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_rhs, _d_rhs, _num_rows*sizeof(double), cudaMemcpyDeviceToHost));
  
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}

void RhsAssembler::copyOwnedRhsVectorToHost() {
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  
  /* Not sure what to do here yet */
  
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}

void RhsAssembler::copySharedRhsVectorToHost() {
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  
  /* Not sure what to do here yet */
  
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}

void RhsAssembler::assemble(Kokkos::View<double **>& data, const int index) {

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif

  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  _d_data = &data(0,index);

  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferTime+=t;

  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  /* Step 1 : sort if chosen */
  if (_sort) {
    int num_threads=128;
    int threads_per_row = nextPowerOfTwo((_nDataPtsToAssemble + _num_rows - 1)/_num_rows);
    //int threads_per_row = 1;
    int num_rows_per_block = num_threads/threads_per_row;
    int num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;
    HypreIntType * _d_rows = (HypreIntType *)(_d_workspace);
    fillRowsKernel<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_kokkos_row_start, _d_kokkos_row_indices, _d_rows);
#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
    void * _d_work = (void *)(_d_rows+_nDataPtsToAssemble);
    sortRhsCub(_nDataPtsToAssemble, (void *) (_d_work), _d_rows, _d_data);
#else
    sortRhsThrust(_nDataPtsToAssemble, _d_rows, _d_data);
#endif
  }

  /* copy from kokkos to internal data structure ... which will be manipulated later */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rhs_indices, _d_kokkos_row_indices, _num_rows*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));

  /* Step 2 : reduce the array and create the RHS Vector */
  int num_threads=128;
  int threads_per_row = nextPowerOfTwo((_nDataPtsToAssemble + _num_rows - 1)/_num_rows);
  //int threads_per_row = 1;
  int num_rows_per_block = num_threads/threads_per_row;
  int num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;

  fillRhsVector<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_kokkos_row_start, _d_data, _d_rhs);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  /*******************************************************/
  /* Step 3 : split into owned and shared not-owned rows */
  /*******************************************************/

  /* Figure out the location of the lower boundary */
  thrust::device_ptr<HypreIntType> iLowerLoc = thrust::min_element(thrust::device,
							      thrust::device_pointer_cast(_d_rhs_indices),
							      thrust::device_pointer_cast(_d_rhs_indices+_num_rows),
							      min_abs_diff(_iLower));
  HypreIntType iLowLoc = thrust::raw_pointer_cast(iLowerLoc)-_d_rhs_indices;    
  
  /* Figure out the location of the upper boundary */
  thrust::device_ptr<HypreIntType> iUpperLoc = thrust::min_element(thrust::device,
							      thrust::device_pointer_cast(_d_rhs_indices),
							      thrust::device_pointer_cast(_d_rhs_indices+_num_rows),
							      min_abs_diff(_iUpper));
  HypreIntType iUppLoc = thrust::raw_pointer_cast(iUpperLoc)-_d_rhs_indices;    
  
  
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  std::vector<HypreIntType> tmp(_num_rows);
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(tmp.data(), _d_rhs_indices, _num_rows*sizeof(HypreIntType), cudaMemcpyDeviceToHost));

  HypreIntType smallest_row = tmp[0];
  HypreIntType smallest_row_this_rank = tmp[iLowLoc];
  HypreIntType largest_row = tmp[_num_rows-1];
  HypreIntType largest_row_this_rank = tmp[iUppLoc];

   printf("Rank %d %s %s %d : name=%s : iLowLoc=%lld, iUppLoc=%lld, all ranks: smallest=%lld, largest=%lld, this rank: smallest=%lld, largest=%lld\n",
  	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),iLowLoc,iUppLoc,smallest_row,largest_row,smallest_row_this_rank,largest_row_this_rank);

#endif      

  if (_iLower==0) {

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
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

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
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

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : CASE 3\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* need to throw an exception here as this is a catastrophic failure */
    assert(_num_rows_this_rank==iUppLoc+1-iLowLoc);

    /* shared matrix exists under the following condition */
    if (iLowLoc>0 || iUppLoc+1<_num_rows) _has_shared = true;

    /* assert that this is always true */
    _num_rows_owned = _num_rows_this_rank;
    _num_rows_shared = _num_rows - _num_rows_owned;

#define SWAP_RHS(x1,x2,x3,n1,n2,n3,s) ( {					           \
	HypreIntType bytes1 = n1*s;						           \
	HypreIntType bytes2 = n2*s;						           \
	ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x3, x1, bytes1, cudaMemcpyDeviceToDevice));    \
	ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x3+n3, x2, bytes2, cudaMemcpyDeviceToDevice)); \
	ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x1, x3+n3, bytes2, cudaMemcpyDeviceToDevice)); \
	ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(x1+n2, x3, bytes1, cudaMemcpyDeviceToDevice)); \
      }									                   \
      )									                          \

    /* Only swap if there are shared entries above */
    if (iLowLoc>0) {
      HypreIntType n1 = iLowLoc, n2=iUppLoc+1-iLowLoc, n3= _nDataPtsToAssemble;
      HypreIntType s = sizeof(HypreIntType);
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
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _assembleTime+=t;
  _nAssemble++;
}


}  // nalu
}  // sierra

#endif
