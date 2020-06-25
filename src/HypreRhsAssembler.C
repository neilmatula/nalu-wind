#include "HypreLinearSystem.h"
#include "HypreRhsAssembler.h"

namespace sierra {
namespace nalu {

/********************************************************************************/
/*  Factory method. Returns a particular derived class instance based on choice */
/*  Current options are Kokkos(0) or Cuda(1).                                   */
/********************************************************************************/
HypreRhsAssembler * HypreRhsAssembler::make_HypreRhsAssembler(int choice, std::string name, bool ensureReproducible,
							      HypreIntType iLower, HypreIntType iUpper,
							      HypreIntType global_num_rows, HypreIntType nDataPtsToAssemble,
							      int rank, HypreIntType num_rows,
							      Kokkos::View<HypreIntType *>& assembly_row_indices,
							      Kokkos::View<HypreIntType *>& assembly_row_start)
{
  if (choice == 0)
    return new HypreKokkosRhsAssembler(name, ensureReproducible, iLower, iUpper, global_num_rows, 
				       nDataPtsToAssemble, rank, num_rows, assembly_row_indices, assembly_row_start);
  else if (choice == 1)
#ifdef KOKKOS_ENABLE_CUDA
    return new HypreCudaRhsAssembler(name, ensureReproducible, iLower, iUpper, global_num_rows, 
				     nDataPtsToAssemble, rank, num_rows, assembly_row_indices, assembly_row_start);
#else
    throw std::runtime_error("Invalid choice for make_HypreRhsAssembler. Exiting.");
#endif
  else
    throw std::runtime_error("Invalid choice for make_HypreRhsAssembler. Exiting.");
}

/**********************************************************************************************************/
/*                            Hypre Kokkos Rhs Assembler implementations                                  */
/**********************************************************************************************************/
HypreKokkosRhsAssembler::HypreKokkosRhsAssembler(std::string name, bool ensureReproducible, HypreIntType iLower, HypreIntType iUpper,
						 HypreIntType global_num_rows, HypreIntType nDataPtsToAssemble, int rank,
						 HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
						 Kokkos::View<HypreIntType *>& assembly_row_start)
  : HypreRhsAssembler(name, ensureReproducible, iLower, iUpper, global_num_rows, nDataPtsToAssemble,
		      rank, num_rows, assembly_row_indices, assembly_row_start)
{
#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld, iLower=%lld, iUpper=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,_iLower,_iUpper);
#endif

  _num_rows_this_rank = _iUpper+1-_iLower;

  /* Step 7 : allocate space */
  _d_rhs = Kokkos::View<double *>("d_rhs",_num_rows);
  _d_rhs_indices = Kokkos::View<HypreIntType *>("d_rhs_indices",_num_rows);
  _h_rhs = Kokkos::create_mirror_view(_d_rhs);
  _h_rhs_indices = Kokkos::create_mirror_view(_d_rhs_indices);

  _d_int_workspace = Kokkos::View<HypreIntType *>("d_int_workspace",3*nDataPtsToAssemble);
  _d_double_workspace = Kokkos::View<double *>("d_double_workspace",nDataPtsToAssemble);

  _memoryUsed = _num_rows*(sizeof(HypreIntType) + sizeof(double))
    + 3*nDataPtsToAssemble*sizeof(HypreIntType) + nDataPtsToAssemble*sizeof(double);

#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
  printf("Done Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs());
#endif
}

HypreKokkosRhsAssembler::~HypreKokkosRhsAssembler() {
#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
  if (_nAssemble>0)
    printf("Mean RHS Assembly Time (%d samples)=%1.5f msec, Data Xfer Time To Host=%1.5f msec\n",
	   _nAssemble,_assembleTime/_nAssemble,_xferHostTime/_nAssemble);
#endif
}


double
HypreKokkosRhsAssembler::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}

void
HypreKokkosRhsAssembler::copyRhsVectorToHost() {

#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  if (getHasShared()) {
    copyOwnedRhsVectorToHost();
    copySharedRhsVectorToHost();
  } else {
    Kokkos::deep_copy(_h_rhs, _d_rhs);
    Kokkos::deep_copy(_h_rhs_indices, _d_rhs_indices);
  }

#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the stop time */
  gettimeofday(&_stop, NULL);
  float msec = (float)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((float)(_stop.tv_sec - _start.tv_sec));
  _xferHostTime+=msec;
#endif
}


void
HypreKokkosRhsAssembler::copyOwnedRhsVectorToHost() {
  Kokkos::deep_copy(_h_rhs_owned, _d_rhs_owned);
  Kokkos::deep_copy(_h_rhs_indices_owned, _d_rhs_indices_owned);
}


void
HypreKokkosRhsAssembler::copySharedRhsVectorToHost() {
  Kokkos::deep_copy(_h_rhs_shared, _d_rhs_shared);
  Kokkos::deep_copy(_h_rhs_indices_shared, _d_rhs_indices_shared);
}


void
HypreKokkosRhsAssembler::assemble(Kokkos::View<double **>& data, const int index) {

#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  /* team for parallel for loops */
  auto num_rows = _num_rows;
  auto team_exec = get_device_team_policy(num_rows, 0, 0);
  //auto nDataPtsToAssemble = _nDataPtsToAssemble;
  auto assembly_row_start = _d_assembly_row_start;
  auto assembly_row_indices = _d_assembly_row_indices;
  auto d_int_workspace = _d_int_workspace;
  auto d_double_workspace = _d_double_workspace;
  auto d_data = data;
  auto d_rhs = _d_rhs;

  if (_ensure_reproducible) {
    throw std::runtime_error("This option is NOT supported. Set ensure_reproducibile=no in the Hypre linear solver blocks.");

    // /* Create the row indices vector that is the same size as the _d_data buffer */
    // Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {

    //     auto rowId = team.league_rank();
    // 	auto begin = assembly_row_start(rowId);
    // 	auto end = assembly_row_start(rowId+1);
    // 	auto row = assembly_row_indices(rowId);
    // 	auto rowLen = end - begin;

    //     Kokkos::parallel_for(Kokkos::TeamThreadRange(team, rowLen),
    // 			     [&](const HypreIntType& i) {
    // 			       d_int_workspace(begin+i) = row;
    // 			     });
    //   });

    // /* absolute value and sequence for keys */
    // Kokkos::parallel_for("abs_sequence", nDataPtsToAssemble, KOKKOS_LAMBDA(const HypreIntType& i) {    
    // 	d_double_workspace(i) = abs(d_data(i, index));
    // 	d_int_workspace(nDataPtsToAssemble+i) = i;
    //   });
   
    // /* Next Sort */
    // typedef decltype(d_int_workspace) KeyViewType;
    // typedef Kokkos::BinOp1D<KeyViewType> BinOp;
    // BinOp binner((HypreIntType)nDataPtsToAssemble, (HypreIntType)0, (HypreIntType)nDataPtsToAssemble);

    // Kokkos::BinSort<KeyViewType, BinOp> Sorter(d_int_workspace,0,nDataPtsToAssemble,binner,true);
    // Sorter.create_permute_vector();
    // //Sorter.sort<double>(element_);

  }

  /* copy assembly_row_indices to rhs_indices so that rhs_indices can be manipulated by the owned/shared computation */
  Kokkos::deep_copy(_d_rhs_indices, _d_assembly_row_indices);

  /**************************************/
  /* reduce the right hand size elements*/
  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {

      auto rowId = team.league_rank();
      auto begin = assembly_row_start(rowId);
      auto end = assembly_row_start(rowId+1);
      auto row = assembly_row_indices(rowId);
      auto rowLen = end - begin;

      double sum = 0.;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, rowLen),
			       [=](HypreIntType& i, double& lsum) {
				lsum += d_data(begin+i, index);
			      }, sum);

      /* sync */
      team.team_barrier();

      /* write the result to memory by thread 0 only */
      if (team.team_rank() == 0) d_rhs(rowId) = sum;
    });      


  /*******************************************************/
  /* Step 3 : split into owned and shared not-owned rows */
  /*******************************************************/
  typedef Kokkos::MinLoc<HypreIntType,HypreIntType>::value_type minloc_type;
  auto d_rhs_indices = _d_rhs_indices;

  /* Figure out the location of the lower boundary */
  minloc_type iLowLoc;
  auto iLower = _iLower;
  Kokkos::parallel_reduce("MinLocReduceLower", num_rows, KOKKOS_LAMBDA(const HypreIntType& i, minloc_type& lminloc) {
      HypreIntType val = (HypreIntType)abs(d_rhs_indices(i)-iLower);
      if( val < lminloc.val ) { lminloc.val = val; lminloc.loc = i; }
    }, Kokkos::MinLoc<HypreIntType,HypreIntType>(iLowLoc));

  /* Figure out the location of the upper boundary */
  minloc_type iUppLoc;
  auto iUpper = _iUpper;
  Kokkos::parallel_reduce("MinLocReduceUpper", num_rows, KOKKOS_LAMBDA(const HypreIntType& i, minloc_type& lminloc) {
      HypreIntType val = (HypreIntType)abs(d_rhs_indices(i)-iUpper);
      if( val < lminloc.val ) { lminloc.val = val; lminloc.loc = i; }
    }, Kokkos::MinLoc<HypreIntType,HypreIntType>(iUppLoc));

#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
  Kokkos::deep_copy(_h_rhs_indices, _d_rhs_indices);

  HypreIntType smallest_row = _h_rhs_indices(0);
  HypreIntType smallest_row_this_rank = _h_rhs_indices(iLowLoc.loc);
  HypreIntType largest_row = _h_rhs_indices(_num_rows-1);
  HypreIntType largest_row_this_rank = _h_rhs_indices(iUppLoc.loc);

  printf("Rank %d %s %s %d : name=%s : iLowLoc=%lld, iUppLoc=%lld, all ranks: smallest=%lld, largest=%lld, this rank: smallest=%lld, largest=%lld\n",
  	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),iLowLoc.loc,iUppLoc.loc,smallest_row,largest_row,smallest_row_this_rank,largest_row_this_rank);

#endif

  /*******************************************************/
  /* Consistency check : make sure things are consistent */
  /*******************************************************/
  if (_num_rows_owned!=_num_rows)
    _has_shared = true;
  else {
    _has_shared=false;
    return;
  }

  /***************************************/
  /* allocations, if not already created */
  /***************************************/
  if (!_owned_shared_views_created) {
    _num_rows_owned = _num_rows_this_rank;
    _num_rows_shared = _num_rows - _num_rows_owned;

    _d_rhs_owned = Kokkos::View<double *>("d_rhs_owned",_num_rows_owned);
    _d_rhs_indices_owned = Kokkos::View<HypreIntType *>("d_rhs_indices_owned",_num_rows_owned);
    _h_rhs_owned = Kokkos::create_mirror_view(_d_rhs_owned);
    _h_rhs_indices_owned = Kokkos::create_mirror_view(_d_rhs_indices_owned);

    _d_rhs_shared = Kokkos::View<double *>("d_rhs_shared",_num_rows_shared);
    _d_rhs_indices_shared = Kokkos::View<HypreIntType *>("d_rhs_indices_shared",_num_rows_shared);
    _h_rhs_shared = Kokkos::create_mirror_view(_d_rhs_shared);
    _h_rhs_indices_shared = Kokkos::create_mirror_view(_d_rhs_indices_shared);

    /* set this flag */
    _owned_shared_views_created = true;
  }

  /***************************************/
  /* Move to owned/shared views          */
  /***************************************/

  auto d_rhs_indices_owned = _d_rhs_indices_owned;
  auto d_rhs_owned = _d_rhs_owned;
  auto d_rhs_indices_shared = _d_rhs_indices_shared;
  auto d_rhs_shared = _d_rhs_shared;
  auto num_rows_owned = _num_rows_owned;
  auto num_rows_shared = _num_rows_shared;
  auto iLowerLoc = iLowLoc.loc;

  if (_iLower==0) {

#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : CASE 1\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* copy owned */
    Kokkos::parallel_for("copy_owned", _num_rows_owned, KOKKOS_LAMBDA(const unsigned& i) {
	d_rhs_indices_owned(i) = d_rhs_indices(i);
	d_rhs_owned(i) = d_rhs(i);
      });
    
    /* copy shared */
    Kokkos::parallel_for("copy_shared", _num_rows_shared, KOKKOS_LAMBDA(const unsigned& i) {
	d_rhs_indices_shared(i) = d_rhs_indices(i+num_rows_owned);
	d_rhs_shared(i) = d_rhs(i+num_rows_owned);
      });
    
  } else if (_iUpper+1==_global_num_rows) {

#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : CASE 2\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* copy shared */
    Kokkos::parallel_for("copy_shared", _num_rows_shared, KOKKOS_LAMBDA(const unsigned& i) {
	d_rhs_indices_shared(i) = d_rhs_indices(i);
	d_rhs_shared(i) = d_rhs(i);
      });
    
    /* copy shared */
    Kokkos::parallel_for("copy_owned", _num_rows_owned, KOKKOS_LAMBDA(const unsigned& i) {
	d_rhs_indices_owned(i) = d_rhs_indices(i+num_rows_shared);
	d_rhs_owned(i) = d_rhs(i+num_rows_shared);
      });

  } else {

#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : CASE 3\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

    /* copy shared lower */
    Kokkos::parallel_for("copy_shared_lower", iLowerLoc, KOKKOS_LAMBDA(const unsigned& i) {
	d_rhs_indices_shared(i) = d_rhs_indices(i);
	d_rhs_shared(i) = d_rhs(i);
      });

    /* copy shared upper */
    auto shift = iLowerLoc+num_rows_owned;
    Kokkos::parallel_for("copy_shared_upper", num_rows_shared-iLowerLoc, KOKKOS_LAMBDA(const unsigned& i) {
	d_rhs_indices_shared(i+iLowerLoc) = d_rhs_indices(i+shift);
	d_rhs_shared(i+iLowerLoc) = d_rhs(i+shift);
      });

    /* copy owned */
    Kokkos::parallel_for("copy_owned", num_rows_owned, KOKKOS_LAMBDA(const unsigned& i) {
	d_rhs_indices_owned(i) = d_rhs_indices(i+iLowerLoc);
	d_rhs_owned(i) = d_rhs(i+iLowerLoc);
      });

  } 

#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the stop time */
  gettimeofday(&_stop, NULL);
  float msec = (float)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((float)(_stop.tv_sec - _start.tv_sec));
  _assembleTime+=msec;
  _nAssemble++;
#endif
}


#ifdef KOKKOS_ENABLE_CUDA

/**********************************************************************************************************/
/*                            Hypre Cuda Rhs Assembler implementations                                    */
/**********************************************************************************************************/

HypreCudaRhsAssembler::HypreCudaRhsAssembler(std::string name, bool ensureReproducible, 
					     HypreIntType iLower, HypreIntType iUpper,
					     HypreIntType global_num_rows, HypreIntType nDataPtsToAssemble, 
					     int rank, HypreIntType num_rows, 
					     Kokkos::View<HypreIntType *>& assembly_row_indices,
					     Kokkos::View<HypreIntType *>& assembly_row_start)
  : HypreRhsAssembler(name, ensureReproducible, iLower, iUpper, global_num_rows, nDataPtsToAssemble,
		      rank, num_rows, assembly_row_indices, assembly_row_start)
{
#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld, iLower=%lld, iUpper=%lld\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,_iLower,_iUpper);
#endif
  
  _num_rows_this_rank = _iUpper+1-_iLower;

  /* Step 7 : allocate space */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_rhs, _num_rows*sizeof(double)));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_rhs_indices, _num_rows*sizeof(HypreIntType)));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_rhs, _num_rows*sizeof(double)));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_rhs_indices, _num_rows*sizeof(HypreIntType)));
  _memoryUsed = _num_rows*(sizeof(HypreIntType) + sizeof(double));


#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
  size_t radix_sort_bytes=0;
  void     *_d_tmp1 = NULL;
  double * _d_dtmp2 = NULL;
  HypreIntType * _d_tmp3 = NULL;
  if (_ensure_reproducible)
    ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_tmp1, radix_sort_bytes, _d_dtmp2, _d_dtmp2, _d_tmp3, _d_tmp3, _nDataPtsToAssemble));
  HypreIntType bytes = 5*_nDataPtsToAssemble*sizeof(double) + radix_sort_bytes;
#else
  HypreIntType bytes = _nDataPtsToAssemble*sizeof(double);
#endif
  
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_workspace, bytes));
  _memoryUsed += bytes;

    
#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  /* create events */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_start));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_stop));
#endif

#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
  double free=0., total=0.;
  deviceMemoryInGBs(free,total);
  printf("Done Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf,  Free Device Memory=%1.6lf,  TotalDeviceMemory=%1.6lf \n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs(),free,total);
#endif
}

HypreCudaRhsAssembler::~HypreCudaRhsAssembler() {

#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
  if (_nAssemble>0)
    printf("Mean RHS Assembly Time (%d samples)=%1.5f msec, Data Xfer Time To Host=%1.5f msec\n",
	   _nAssemble,_assembleTime/_nAssemble,_xferHostTime/_nAssemble);
  /* destroy events */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_start));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_stop));
#endif

  /* free the data */
  if (_d_rhs) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_rhs)); _d_rhs=NULL; }
  if (_h_rhs) { ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_rhs)); _d_rhs=NULL; }
  if (_d_rhs_indices) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_rhs_indices)); _d_rhs_indices=NULL; }
  if (_h_rhs_indices) { ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_rhs_indices)); _d_rhs_indices=NULL; }
  if (_d_workspace) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_workspace)); _d_workspace=NULL; }
  
}


double 
HypreCudaRhsAssembler::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}


void
HypreCudaRhsAssembler::deviceMemoryInGBs(double & free, double & total) const {
  size_t f=0, t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemGetInfo(&f, &t));
  free = 1.0*f/(1024.*1024.*1024.);
  total = 1.0*t/(1024.*1024.*1024.);
}


void
HypreCudaRhsAssembler::copyRhsVectorToHost() {

#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
#endif
  
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_rhs_indices, _d_rhs_indices, _num_rows*sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_rhs, _d_rhs, _num_rows*sizeof(double), cudaMemcpyDeviceToHost));
  
#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
#endif
}


void
HypreCudaRhsAssembler::assemble(Kokkos::View<double **>& data, const int index) {

#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif

#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
#endif

  _d_data = &data(0,index);

  /*********************************************************************/
  /* Get the raw pointers of the assembly Kokkos views data structures */
  /*********************************************************************/
  HypreIntType * _d_assembly_row_start_ptr = _d_assembly_row_start.data();
  HypreIntType * _d_assembly_row_indices_ptr = _d_assembly_row_indices.data();

  /* Step 1 : sort if chosen */
  if (_ensure_reproducible) {
    int num_threads=128;
    int threads_per_row = nextPowerOfTwo((_nDataPtsToAssemble + _num_rows - 1)/_num_rows);
    //int threads_per_row = 1;
    int num_rows_per_block = num_threads/threads_per_row;
    int num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;
    HypreIntType * _d_rows = (HypreIntType *)(_d_workspace);
    sierra::nalu::fillRowsKernel<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_assembly_row_start_ptr, 
					       _d_assembly_row_indices_ptr, _d_rows);
#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
    void * _d_work = (void *)(_d_rows+_nDataPtsToAssemble);
    sortRhsCub(_nDataPtsToAssemble, (void *) (_d_work), _d_rows, _d_data);
#else
    sortRhsThrust(_nDataPtsToAssemble, _d_rows, _d_data);
#endif
  }

  /* copy from kokkos to internal data structure ... which will be manipulated later */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rhs_indices, _d_assembly_row_indices_ptr, 
				      _num_rows*sizeof(HypreIntType), cudaMemcpyDeviceToDevice));

  /* Step 2 : reduce the array and create the RHS Vector */
  int num_threads=128;
  int threads_per_row = nextPowerOfTwo((_nDataPtsToAssemble + _num_rows - 1)/_num_rows);
  //int threads_per_row = 1;
  int num_rows_per_block = num_threads/threads_per_row;
  int num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;

  fillRhsVector<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_assembly_row_start_ptr, _d_data, _d_rhs);
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
  
  
#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
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

#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
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

#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
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

#ifdef HYPRE_RHS_ASSEMBLER_DEBUG
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
      )									                   \

    /* Only swap if there are shared entries above */
    if (iLowLoc>0) {
      HypreIntType n1 = iLowLoc, n2=iUppLoc+1-iLowLoc, n3= _nDataPtsToAssemble;
      HypreIntType s = sizeof(HypreIntType);
      SWAP_RHS(_d_rhs_indices, _d_rhs_indices+iLowLoc, (HypreIntType *)_d_workspace, n1, n2, n3, s);

      s = sizeof(double);
      SWAP_RHS(_d_rhs, _d_rhs+iLowLoc, (HypreIntType *)_d_workspace, n1, n2, n3, s);
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

#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _assembleTime+=t;
  _nAssemble++;
#endif
}

#endif

}  // nalu
}  // sierra
