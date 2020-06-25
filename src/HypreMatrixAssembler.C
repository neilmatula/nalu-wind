#include "HypreLinearSystem.h"
#include "HypreMatrixAssembler.h"

namespace sierra {
namespace nalu {

/********************************************************************************/
/*  Factory method. Returns a particular derived class instance based on choice */
/*  Current options are Kokkos(0) or Cuda(1).                                   */
/********************************************************************************/
HypreMatrixAssembler * HypreMatrixAssembler::make_HypreMatrixAssembler(int choice, std::string name, bool ensureReproducible,
								       HypreIntType iLower, HypreIntType iUpper,
								       HypreIntType jLower, HypreIntType jUpper,
								       HypreIntType global_num_rows, HypreIntType global_num_cols,
								       HypreIntType nDataPtsToAssemble, int rank,
								       HypreIntType num_rows, 
								       Kokkos::View<HypreIntType *>& assembly_row_indices,
								       Kokkos::View<HypreIntType *>& assembly_row_start)
{
  if (choice == 0)
    return new HypreKokkosMatrixAssembler(name, ensureReproducible, iLower, iUpper,
					  jLower, jUpper, global_num_rows, global_num_cols,
					  nDataPtsToAssemble, rank, num_rows, 
					  assembly_row_indices, assembly_row_start);
  else if (choice == 1)
#ifdef KOKKOS_ENABLE_CUDA
    return new HypreCudaMatrixAssembler(name, ensureReproducible, iLower, iUpper,
					jLower, jUpper, global_num_rows, global_num_cols,
					nDataPtsToAssemble, rank, num_rows, 
					assembly_row_indices, assembly_row_start);
#else
    throw std::runtime_error("Invalid choice for make_HypreMatrixAssembler. Exiting.");
#endif
  else
    throw std::runtime_error("Invalid choice for make_HypreMatrixAssembler. Exiting.");
}


/**********************************************************************************************************/
/*                            Hypre Kokkos Matrix Assembler implementations                               */
/**********************************************************************************************************/

HypreKokkosMatrixAssembler::HypreKokkosMatrixAssembler(std::string name, bool ensureReproducible, HypreIntType iLower,
						       HypreIntType iUpper, HypreIntType jLower, HypreIntType jUpper,
						       HypreIntType global_num_rows, HypreIntType global_num_cols,
						       HypreIntType nDataPtsToAssemble, int rank,
						       HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
						       Kokkos::View<HypreIntType *>& assembly_row_start)
  : HypreMatrixAssembler(name, ensureReproducible, iLower, iUpper, jLower, jUpper, global_num_rows, global_num_cols,
			 nDataPtsToAssemble, rank, num_rows, assembly_row_indices, assembly_row_start)
{
#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld, iLower=%lld, iUpper=%lld, jLower=%lld, jUpper=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,_iLower,_iUpper,_jLower,_jUpper);
#endif

  _num_rows_this_rank = _iUpper+1-_iLower;
  _num_cols_this_rank = _jUpper+1-_jLower;

  /* allocate space */
  _d_row_indices = Kokkos::View<HypreIntType *>("d_row_indices",_num_rows);
  _d_row_counts = Kokkos::View<HypreIntType *>("d_row_counts",_num_rows);

  _h_row_indices = Kokkos::create_mirror_view(_d_row_indices);
  _h_row_counts = Kokkos::create_mirror_view(_d_row_counts);

  /* scratch space */
  _d_dense_keys  = Kokkos::View<HypreIntType *>("d_dense_keys",nDataPtsToAssemble);
  _d_mat_elem_bin_locs  = Kokkos::View<HypreIntType *>("d_mat_elem_bin_locs",nDataPtsToAssemble+1);
  _d_mat_elem_bins  = Kokkos::View<HypreIntType *>("d_mat_elem_bin_locs",nDataPtsToAssemble+1);
  _d_transitions  = Kokkos::View<HypreIntType *>("d_transitions",nDataPtsToAssemble+1);
  _d_row_counts_scanned  = Kokkos::View<HypreIntType *>("d_row_counts_scanned",num_rows+1);
  _h_row_counts_scanned = Kokkos::create_mirror_view(_d_row_counts_scanned);

  _memoryUsed = (3*_num_rows+1)*sizeof(HypreIntType) + (4*nDataPtsToAssemble+3)*sizeof(HypreIntType);
 
#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
  printf("Done Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs());
#endif
}

HypreKokkosMatrixAssembler::~HypreKokkosMatrixAssembler() {
#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
  if (_nAssemble>0) {
    printf("Mean Matrix Assembly Time (%d samples)=%1.5f msec, Data Xfer Time To Host=%1.5f msec\n",
	   _nAssemble,_assembleTime/_nAssemble,_xferHostTime/_nAssemble);
    printf("\tCompute Dense Keys Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_denseKeysTime/_nAssemble);
    printf("\tSort Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_sortTime/_nAssemble);
    printf("\tCompute Matrix Elem Boundaries Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_matElemBndryTime/_nAssemble);
    printf("\tAllocate Matrix Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_matAllocateTime/_nAssemble);
    printf("\tFill CSR Matrix Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_fillCSRMatTime/_nAssemble);
    printf("\tFind Owned/Shared Boundary Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_findOwnedSharedBndryTime/_nAssemble);
    printf("\tFill Owned/Shared Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_fillOwnedSharedTime/_nAssemble);
  }
#endif
}


double
HypreKokkosMatrixAssembler::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}


void
HypreKokkosMatrixAssembler::copyCSRMatrixToHost() {

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  if (getHasShared()) {
    copyOwnedCSRMatrixToHost();
    copySharedCSRMatrixToHost();
  } else {
    Kokkos::deep_copy(_h_row_indices, _d_row_indices);
    Kokkos::deep_copy(_h_row_counts, _d_row_counts);
    Kokkos::deep_copy(_h_col_indices, _d_col_indices);
    Kokkos::deep_copy(_h_values, _d_values);
  }

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the stop time */
  gettimeofday(&_stop, NULL);
  _xferHostTime += (float)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((float)(_stop.tv_sec - _start.tv_sec));
#endif
}


void
HypreKokkosMatrixAssembler::copyOwnedCSRMatrixToHost() {
  Kokkos::deep_copy(_h_row_indices_owned, _d_row_indices_owned);
  Kokkos::deep_copy(_h_row_counts_owned, _d_row_counts_owned);
  Kokkos::deep_copy(_h_col_indices_owned, _d_col_indices_owned);
  Kokkos::deep_copy(_h_values_owned, _d_values_owned);
}


void
HypreKokkosMatrixAssembler::copySharedCSRMatrixToHost() {
  Kokkos::deep_copy(_h_row_indices_shared, _d_row_indices_shared);
  Kokkos::deep_copy(_h_row_counts_shared, _d_row_counts_shared);
  Kokkos::deep_copy(_h_col_indices_shared, _d_col_indices_shared);
  Kokkos::deep_copy(_h_values_shared, _d_values_shared);
}


void
HypreKokkosMatrixAssembler::assemble(Kokkos::View<HypreIntType *>& cols, Kokkos::View<double *>& data) {

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the start time */
  gettimeofday(&_start, NULL);
  _nAssemble++;
#endif


  /* team for parallel for loops */
  auto num_rows = _num_rows;
  auto team_exec = get_device_team_policy(num_rows, 0, 0);
  auto nDataPtsToAssemble = _nDataPtsToAssemble;
  auto assembly_row_start = _d_assembly_row_start;
  auto assembly_row_indices = _d_assembly_row_indices;
  auto d_dense_keys = _d_dense_keys;
  auto d_mat_elem_bin_locs = _d_mat_elem_bin_locs;
  auto d_mat_elem_bins = _d_mat_elem_bins;
  auto d_transitions = _d_transitions;
  auto d_cols = cols;
  auto d_data = data;
  auto global_num_cols = _global_num_cols;

  /* initialize scratch space to 0 */
  Kokkos::deep_copy(d_mat_elem_bins, 0);
  Kokkos::deep_copy(d_mat_elem_bin_locs, 0);
  Kokkos::deep_copy(d_transitions, 0);


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the start time */
  gettimeofday(&_start_refined, NULL);
#endif


  /*************************************************************************************/
  /* Create the dense indices key vector that is the same size as the data/cols buffer */
  /*************************************************************************************/
  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {

      auto rowId = team.league_rank();
      auto begin = assembly_row_start(rowId);
      auto end = assembly_row_start(rowId+1);
      auto row = assembly_row_indices(rowId);
      auto rowLen = end - begin;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, rowLen),
			   [&](const HypreIntType& i) { d_dense_keys(begin+i) = row * global_num_cols + d_cols(begin+i); });
    });


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the stop time */
  gettimeofday(&_stop_refined, NULL);
  _denseKeysTime += (float)(_stop_refined.tv_usec - _start_refined.tv_usec) / 1.e3 + 1.e3*((float)(_stop_refined.tv_sec - _start_refined.tv_sec));

  /* record the start time */
  gettimeofday(&_start_refined, NULL);
#endif


  if (_ensure_reproducible) {
    /*********************************************************************************************************************/
    /* Ensure Reproducible : Here we sort on absolute value first, then on the dense index. This ensures reproducibility */
    /*********************************************************************************************************************/
    throw std::runtime_error("This option is NOT supported. Set ensure_reproducibile=no in the Hypre linear solver blocks.");

  } else {
    /****************************************/
    /* Sort : based on the dense index only */
    /****************************************/
    typedef decltype(d_dense_keys) KeyViewType;
    typedef Kokkos::BinOp1D<KeyViewType> CompType;
    Kokkos::MinMaxScalar<typename KeyViewType::non_const_value_type> result;
    Kokkos::MinMax<typename KeyViewType::non_const_value_type> reducer(result);
    Kokkos::parallel_reduce("findMinMax", Kokkos::RangePolicy<typename KeyViewType::execution_space>(0, d_dense_keys.extent(0)),
			    Kokkos::Impl::min_max_functor<KeyViewType>(d_dense_keys), reducer);
#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
    printf("Rank %d %s %s %d : name=%s : myMin=%lld, myMax=%lld\n",
	   _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),result.min_val,result.max_val);
#endif
    Kokkos::BinSort<KeyViewType, CompType> bin_sort(d_dense_keys, CompType(d_dense_keys.extent(0) / 2, result.min_val, result.max_val), true);
    bin_sort.create_permute_vector();
    bin_sort.sort(d_dense_keys);
    bin_sort.sort(d_cols);
    bin_sort.sort(d_data);
  }


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the stop time */
  gettimeofday(&_stop_refined, NULL);
  _sortTime += (float)(_stop_refined.tv_usec - _start_refined.tv_usec) / 1.e3 + 1.e3*((float)(_stop_refined.tv_sec - _start_refined.tv_sec));

  /* record the start time */
  gettimeofday(&_start_refined, NULL);
#endif


  /********************************************************/
  /* The next 3 kernels figures out the boundaries of the */
  /* matrix elements in linear memory.                    */   
  /********************************************************/

  /* The initial kernel for determining the matrix element locations in linear memory */
  Kokkos::parallel_for("matrix_elem_locations_initial", nDataPtsToAssemble, KOKKOS_LAMBDA(const HypreIntType& i) {    
      HypreIntType delta = d_dense_keys(i+1) - d_dense_keys(i); 
      if (i<nDataPtsToAssemble-1) {
	d_mat_elem_bin_locs(i+1) = (i+1) * (delta>0 ? 1 : 0);
	d_transitions(i+1) = (delta>0 ? 1 : 0);
      } else if (i==nDataPtsToAssemble-1) {
	d_mat_elem_bin_locs(i+1) = nDataPtsToAssemble;
	d_transitions(i+1) = 1;
      }
    });

  /* inclusive scan : this determines the number of nonzeros in the matrix*/
  Kokkos::parallel_scan("inclusive_scan", nDataPtsToAssemble+1, 
			KOKKOS_LAMBDA(const HypreIntType i, HypreIntType& update, const bool final) {
			  // Load old value in case we update it before accumulating
			  const HypreIntType val_i = d_transitions(i); 
			  update += val_i;
			  if (final) d_transitions(i) = update;
			}, _num_nonzeros);
  
#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : _num_nonzeros=%lld, _num_rows=%lld\n",
    _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_num_nonzeros,_num_rows);
#endif

  /* The final kernel for determining the matrix element locations in linear memory */
  auto num_nonzeros = _num_nonzeros;
  Kokkos::parallel_for("matrix_elem_locations_final", nDataPtsToAssemble+1, KOKKOS_LAMBDA(const HypreIntType& i) {    
      HypreIntType x=0, y=0; 
      if (i<=nDataPtsToAssemble) {
	x = d_mat_elem_bin_locs(i);
	y = d_transitions(i);
	if (x>0 && y>0 && y<=num_nonzeros) d_mat_elem_bins(y) = x;
      }
    });


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the stop time */
  gettimeofday(&_stop_refined, NULL);
  _matElemBndryTime += (float)(_stop_refined.tv_usec - _start_refined.tv_usec) / 1.e3 + 1.e3*((float)(_stop_refined.tv_sec - _start_refined.tv_sec));

  /* record the start time */
  gettimeofday(&_start_refined, NULL);
#endif


  /*****************************/
  /*     Allocate space        */
  /*****************************/
  if (!_csrMatMemoryAdded) {
    _d_col_indices = Kokkos::View<HypreIntType *>("d_col_indices",_num_nonzeros);
    _d_values = Kokkos::View<double *>("d_values",_num_nonzeros);

    _h_col_indices = Kokkos::create_mirror_view(_d_col_indices);
    _h_values = Kokkos::create_mirror_view(_d_values);
    _memoryUsed += (_num_nonzeros)*(sizeof(HypreIntType) + sizeof(double));
    _csrMatMemoryAdded = true;
  }


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the stop time */
  gettimeofday(&_stop_refined, NULL);
  _matAllocateTime += (float)(_stop_refined.tv_usec - _start_refined.tv_usec) / 1.e3 + 1.e3*((float)(_stop_refined.tv_sec - _start_refined.tv_sec));

  /* record the start time */
  gettimeofday(&_start_refined, NULL);
#endif


#ifdef HYPRE_CUDA_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs());
#endif

  /***********************************************/
  /*          CSR Matrix Computation             */
  /***********************************************/

  /* first reset the row counts */
  Kokkos::deep_copy(_d_row_counts, 0);

  /* copy from kokkos to internal data structure ... which can be manipulated later */
  Kokkos::deep_copy(_d_row_indices, _d_assembly_row_indices);
  
  /* Expand assembly_row_start into a vector of the same size as cols/data */
  team_exec = get_device_team_policy(num_rows, 0, 0);
  auto d_assembly_row_start_expanded = _d_dense_keys;
  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {

      auto rowId = team.league_rank();
      auto begin = assembly_row_start(rowId);
      auto end = assembly_row_start(rowId+1);
      auto rowLen = end - begin;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, rowLen),
			   [&](const HypreIntType& i) { d_assembly_row_start_expanded(begin+i) = rowId; });
    });

  /* Fill the CSR Matrix */
  team_exec = get_device_team_policy(num_nonzeros, 0, 0);
  auto d_row_counts = _d_row_counts;
  auto d_col_indices = _d_col_indices;
  auto d_values = _d_values;
  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {

      auto elemId = team.league_rank();
      auto begin = d_mat_elem_bins(elemId);
      auto end = d_mat_elem_bins(elemId+1);
      auto rowLen = end - begin;
      auto rbegin = d_assembly_row_start_expanded(begin);

      /* Inner reduction to create a matrix element */
      double sum = 0.0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, rowLen), [&](const HypreIntType i, double& update) {
	  update += d_data(begin+i); 
	}, sum);

      /* thread 0 writes the results */
      if (team.team_rank() == 0) {
	d_col_indices(elemId) = d_cols(begin);
	d_values(elemId) = sum;
	/* atomic accumulate for the row count */
	Kokkos::atomic_add(&d_row_counts(rbegin), (HypreIntType)1);
      }
    });


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the stop time */
  gettimeofday(&_stop_refined, NULL);
  _fillCSRMatTime += (float)(_stop_refined.tv_usec - _start_refined.tv_usec) / 1.e3 + 1.e3*((float)(_stop_refined.tv_sec - _start_refined.tv_sec));

  /* record the start time */
  gettimeofday(&_start_refined, NULL);
#endif


  /*******************************************************/
  /*      Split into owned and shared not-owned rows     */
  /*******************************************************/

  /* Next get the locations of the upper and lower boundaries */
  typedef Kokkos::MinLoc<HypreIntType,HypreIntType>::value_type minloc_type;
  auto d_row_indices = _d_row_indices;

  /* Figure out the location of the lower boundary */
  minloc_type iLowLoc;
  auto iLower = _iLower;
  Kokkos::parallel_reduce("MinLocReduceLower", num_rows, KOKKOS_LAMBDA(const HypreIntType& i, minloc_type& lminloc) {
      HypreIntType val = (HypreIntType)abs(d_row_indices(i)-iLower);
      if( val < lminloc.val ) { lminloc.val = val; lminloc.loc = i; }
    }, Kokkos::MinLoc<HypreIntType,HypreIntType>(iLowLoc));

  /* Figure out the location of the upper boundary */
  minloc_type iUppLoc;
  auto iUpper = _iUpper;
  Kokkos::parallel_reduce("MinLocReduceUpper", num_rows, KOKKOS_LAMBDA(const HypreIntType& i, minloc_type& lminloc) {
      HypreIntType val = (HypreIntType)abs(d_row_indices(i)-iUpper);
      if( val < lminloc.val ) { lminloc.val = val; lminloc.loc = i; }
    }, Kokkos::MinLoc<HypreIntType,HypreIntType>(iUppLoc));


#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : iLowLoc=%lld at %lld, iUppLoc=%lld at %lld\n",
  	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),iLowLoc.val,iLowLoc.loc,iUppLoc.val,iUppLoc.loc);
#endif


  /* exclusive scan : this will tell us where to split the cols/values data structure for owned/shared */
  auto d_row_counts_scanned = _d_row_counts_scanned;
  Kokkos::deep_copy(d_row_counts_scanned, 0);
  HypreIntType dummy;
  Kokkos::parallel_scan("exclusive_scan", num_rows+1, 
			KOKKOS_LAMBDA(const HypreIntType i, HypreIntType& update, const bool final) {
			  // Load old value in case we update it before accumulating
			  const HypreIntType val_i = d_row_counts(i); 
			  if (final) d_row_counts_scanned(i) = update;
			  update += val_i;
			}, dummy);
  /* copy this back to the host */
  Kokkos::deep_copy(_h_row_counts_scanned, _d_row_counts_scanned);
  
  /* retrieve these values */
  HypreIntType nnz_lower=_h_row_counts_scanned(iLowLoc.loc);
  HypreIntType nnz_upper=_h_row_counts_scanned(iUppLoc.loc+1);


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the stop time */
  gettimeofday(&_stop_refined, NULL);
  _findOwnedSharedBndryTime += (float)(_stop_refined.tv_usec - _start_refined.tv_usec) / 1.e3 + 1.e3*((float)(_stop_refined.tv_sec - _start_refined.tv_sec));

  /* record the start time */
  gettimeofday(&_start_refined, NULL);
#endif


#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : num_nonzeros=%lld, dummy=%lld, nnz_lower=%lld, nnz_upper=%lld\n",
  	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),num_nonzeros,dummy,nnz_lower,nnz_upper);
#endif

  /* Compute owned/shared meta data */
  _num_rows_owned = _num_rows_this_rank;
  _num_rows_shared = _num_rows - _num_rows_owned;

  _num_nonzeros_owned = nnz_upper - nnz_lower;
  _num_nonzeros_shared = _num_nonzeros - _num_nonzeros_owned;

  /*******************************************************/
  /* Consistency check : make sure things are consistent */
  /*******************************************************/
  if ((_num_rows_owned!=_num_rows && _num_nonzeros_owned==_num_nonzeros) ||
      (_num_rows_owned==_num_rows && _num_nonzeros_owned!=_num_nonzeros)) {
    /* This is inconsistent. Need to throw an exception */
    throw std::runtime_error("Inconsistency detected. This should not happen.");
  }

  _has_shared=false;
  if (_num_rows_owned!=_num_rows && _num_nonzeros_owned!=_num_nonzeros) {
    _has_shared = true;

    /***************************************/
    /* allocations, if not already created */
    /***************************************/
    if (!_owned_shared_views_created) {
      _d_values_owned = Kokkos::View<double *>("d_values_owned",_num_nonzeros_owned);
      _d_col_indices_owned = Kokkos::View<HypreIntType *>("d_col_indices_owned",_num_nonzeros_owned);
      _d_row_indices_owned = Kokkos::View<HypreIntType *>("d_row_indices_owned",_num_rows_owned);
      _d_row_counts_owned = Kokkos::View<HypreIntType *>("d_row_counts_owned",_num_rows_owned);
      
      _h_values_owned = Kokkos::create_mirror_view(_d_values_owned);
      _h_col_indices_owned = Kokkos::create_mirror_view(_d_col_indices_owned);
      _h_row_indices_owned = Kokkos::create_mirror_view(_d_row_indices_owned);
      _h_row_counts_owned = Kokkos::create_mirror_view(_d_row_counts_owned);
      
      _d_values_shared = Kokkos::View<double *>("d_values_shared",_num_nonzeros_shared);
      _d_col_indices_shared = Kokkos::View<HypreIntType *>("d_col_indices_shared",_num_nonzeros_shared);
      _d_row_indices_shared = Kokkos::View<HypreIntType *>("d_row_indices_shared",_num_rows_shared);
      _d_row_counts_shared = Kokkos::View<HypreIntType *>("d_row_counts_shared",_num_rows_shared);
      
      _h_values_shared = Kokkos::create_mirror_view(_d_values_shared);
      _h_col_indices_shared = Kokkos::create_mirror_view(_d_col_indices_shared);
      _h_row_indices_shared = Kokkos::create_mirror_view(_d_row_indices_shared);
      _h_row_counts_shared = Kokkos::create_mirror_view(_d_row_counts_shared);
      
      /* set this flag */
      _owned_shared_views_created = true;
    }

    /***************************************/
    /* Move to owned/shared views          */
    /***************************************/

    /* For device capture ... ugh */
    auto d_values_owned = _d_values_owned;
    auto d_values_shared = _d_values_shared;
    auto d_col_indices_owned = _d_col_indices_owned;
    auto d_col_indices_shared = _d_col_indices_shared;
    auto d_row_indices_owned = _d_row_indices_owned;
    auto d_row_indices_shared = _d_row_indices_shared;
    auto d_row_counts_owned = _d_row_counts_owned;
    auto d_row_counts_shared = _d_row_counts_shared;
    
    if (_iLower==0) {

#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
      printf("Rank %d %s %s %d : name=%s : CASE 1\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

      /* copy owned */
      Kokkos::parallel_for("CASE1_copy_owned1", _num_nonzeros_owned, KOKKOS_LAMBDA(const unsigned& i) {
	  d_values_owned(i) = d_values(i);
	  d_col_indices_owned(i) = d_col_indices(i);
	});
      Kokkos::parallel_for("CASE1_copy_owned2", _num_rows_owned, KOKKOS_LAMBDA(const unsigned& i) {
	  d_row_indices_owned(i) = d_row_indices(i);
	  d_row_counts_owned(i) = d_row_counts(i);
	});
      
      /* copy shared */
      auto shift1 = _num_nonzeros_owned;
      Kokkos::parallel_for("CASE1_copy_shared1", _num_nonzeros_shared, KOKKOS_LAMBDA(const unsigned& i) {
	  d_values_shared(i) = d_values(i+shift1);
	  d_col_indices_shared(i) = d_col_indices(i+shift1);
	});
      auto shift2 = _num_rows_owned;
      Kokkos::parallel_for("CASE1_copy_shared2", _num_rows_shared, KOKKOS_LAMBDA(const unsigned& i) {
	  d_row_indices_shared(i) = d_row_indices(i+shift2);
	  d_row_counts_shared(i) = d_row_counts(i+shift2);
	});
    
    } else if (_iUpper+1==_global_num_rows) {

#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
      printf("Rank %d %s %s %d : name=%s : CASE 2\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif      

      /* copy shared */
      Kokkos::parallel_for("CASE2_copy_shared1", _num_nonzeros_shared, KOKKOS_LAMBDA(const unsigned& i) {
	  d_values_shared(i) = d_values(i);
	d_col_indices_shared(i) = d_col_indices(i);
	});
      Kokkos::parallel_for("CASE2_copy_shared2", _num_rows_shared, KOKKOS_LAMBDA(const unsigned& i) {
	  d_row_indices_shared(i) = d_row_indices(i);
	  d_row_counts_shared(i) = d_row_counts(i);
	});

      /* copy owned */
      auto shift1 = _num_nonzeros_shared;
      Kokkos::parallel_for("CASE2_copy_owned1", _num_nonzeros_owned, KOKKOS_LAMBDA(const unsigned& i) {
	  d_values_owned(i) = d_values(i+shift1);
	  d_col_indices_owned(i) = d_col_indices(i+shift1);
	});
      auto shift2 = _num_rows_shared;
      Kokkos::parallel_for("CASE2_copy_owned2", _num_rows_owned, KOKKOS_LAMBDA(const unsigned& i) {
	  d_row_indices_owned(i) = d_row_indices(i+shift2);
	  d_row_counts_owned(i) = d_row_counts(i+shift2);
	});

    } else {

#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
      printf("Rank %d %s %s %d : name=%s : CASE 3\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif
     
      auto iLowerLoc = iLowLoc.loc;
      auto iUpperLoc = iUppLoc.loc;
      
      /* copy shared lower */
      Kokkos::parallel_for("CASE3_copy_shared1", nnz_lower, KOKKOS_LAMBDA(const unsigned& i) {
	  d_values_shared(i) = d_values(i);
	  d_col_indices_shared(i) = d_col_indices(i);
	});
      Kokkos::parallel_for("CASE3_copy_shared2", iLowerLoc, KOKKOS_LAMBDA(const unsigned& i) {
	  d_row_indices_shared(i) = d_row_indices(i);
	  d_row_counts_shared(i) = d_row_counts(i);
	});
      
      /* copy owned */
      auto shift = nnz_lower;
      Kokkos::parallel_for("CASE3_copy_owned1", _num_nonzeros_owned, KOKKOS_LAMBDA(const unsigned& i) {
	  d_values_owned(i) = d_values(i+shift);
	  d_col_indices_owned(i) = d_col_indices(i+shift);
	});
      Kokkos::parallel_for("CASE3_copy_owned2", _num_rows_owned, KOKKOS_LAMBDA(const unsigned& i) {
	  d_row_indices_owned(i) = d_row_indices(i+iLowerLoc);
	  d_row_counts_owned(i) = d_row_counts(i+iLowerLoc);
	});

      /* copy shared upper */
      auto shift1 = nnz_lower+_num_nonzeros_owned;
      Kokkos::parallel_for("CASE1_copy_shared3", _num_nonzeros - nnz_upper, KOKKOS_LAMBDA(const unsigned& i) {
	  d_values_shared(i+shift) = d_values(i+shift1);
	  d_col_indices_shared(i+shift) = d_col_indices(i+shift1);
	});
      Kokkos::parallel_for("CASE3_copy_shared4", _num_rows_shared-iLowerLoc, KOKKOS_LAMBDA(const unsigned& i) {
	  d_row_indices_shared(i+iLowerLoc) = d_row_indices(i+iUpperLoc+1);
	  d_row_counts_shared(i+iLowerLoc) = d_row_counts(i+iUpperLoc+1);
	});
    }
  }

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  Kokkos::fence();
  /* record the stop time */
  gettimeofday(&_stop_refined, NULL);
  _fillOwnedSharedTime += (float)(_stop_refined.tv_usec - _start_refined.tv_usec) / 1.e3 + 1.e3*((float)(_stop_refined.tv_sec - _start_refined.tv_sec));
  gettimeofday(&_stop, NULL);
  _assembleTime += (float)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((float)(_stop.tv_sec - _start.tv_sec));
#endif
}


#ifdef KOKKOS_ENABLE_CUDA

/**********************************************************************************************************/
/*                            Hypre Cuda Matrix Assembler implementations                               */
/**********************************************************************************************************/

HypreCudaMatrixAssembler::HypreCudaMatrixAssembler(std::string name, bool ensureReproducible, HypreIntType iLower,
						   HypreIntType iUpper, HypreIntType jLower, HypreIntType jUpper,
						   HypreIntType global_num_rows, HypreIntType global_num_cols,
						   HypreIntType nDataPtsToAssemble, int rank,
						   HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
						   Kokkos::View<HypreIntType *>& assembly_row_start)
  : HypreMatrixAssembler(name, ensureReproducible, iLower, iUpper, jLower, jUpper, global_num_rows, global_num_cols,
			 nDataPtsToAssemble, rank, num_rows, assembly_row_indices, assembly_row_start)
{
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("\nRank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld, iLower=%lld, iUpper=%lld, jLower=%lld, jUpper=%lld\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,_iLower,_iUpper,_jLower,_jUpper);
#endif

  _num_rows_this_rank = _iUpper+1-_iLower;
  _num_cols_this_rank = _jUpper+1-_jLower;

  /* allocate some space */
  _d_assembly_row_indices = assembly_row_indices;
  _d_assembly_row_start = assembly_row_start;

  /* Allocate these data structures now */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_row_indices, _num_rows*sizeof(HypreIntType)));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_row_indices, _num_rows*sizeof(HypreIntType)));
  
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_row_counts, _num_rows*sizeof(unsigned long long int)));    
  ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_row_counts, _num_rows*sizeof(HypreIntType)));    

  _memoryUsed = _num_rows*(sizeof(HypreIntType) + sizeof(unsigned long long int));    

#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
  void     *_d_tmp1 = NULL;
  double * _d_dtmp2 = NULL;
  HypreIntType * _d_tmp3 = NULL;
  size_t temp_bytes1=0;
  if (_ensure_reproducible)
    ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_tmp1, temp_bytes1, _d_dtmp2, _d_dtmp2,
							     _d_tmp3, _d_tmp3, _nDataPtsToAssemble));

  _d_tmp1 = NULL;
  HypreIntType * _d_tmp4 = NULL;
  size_t temp_bytes2=0;
  ASSEMBLER_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(_d_tmp1, temp_bytes2, _d_tmp4, _d_tmp4, _d_tmp4,
							   _d_tmp4, _nDataPtsToAssemble));

  HypreIntType _radix_sort_bytes = (HypreIntType)(temp_bytes1 > temp_bytes2 ? temp_bytes1 : temp_bytes2);
  HypreIntType bytes1 = 4*_nDataPtsToAssemble * sizeof(HypreIntType) + _radix_sort_bytes;
  HypreIntType bytes2 = 4*(_nDataPtsToAssemble+1)*sizeof(HypreIntType);
  HypreIntType bytes = bytes1 > bytes2 ? bytes1 : bytes2;
#else
  HypreIntType bytes = 4*(_nDataPtsToAssemble+1)*sizeof(HypreIntType);
#endif
  
  ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_workspace, bytes));
  _memoryUsed += bytes;


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* create events */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_start));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_start_refined));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_stop_refined));
#endif
 
#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  double free=0., total=0.;
  deviceMemoryInGBs(free,total);
  printf("Done Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf,  Free Device Memory=%1.6lf,  TotalDeviceMemory=%1.6lf \n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs(),free,total);
#endif
}



HypreCudaMatrixAssembler::~HypreCudaMatrixAssembler() {

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  printf("\nRank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
  if (_nAssemble>0) {
    printf("Mean Matrix Assembly Time (%d samples)=%1.5f msec, Data Xfer Time To Host=%1.5f msec\n",
	   _nAssemble,_assembleTime/_nAssemble,_xferHostTime/_nAssemble);
    printf("\tCompute Dense Keys Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_denseKeysTime/_nAssemble);
    printf("\tSort Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_sortTime/_nAssemble);
    printf("\tCompute Matrix Elem Boundaries Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_matElemBndryTime/_nAssemble);
    printf("\tAllocate Matrix Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_matAllocateTime/_nAssemble);
    printf("\tFill CSR Matrix Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_fillCSRMatTime/_nAssemble);
    printf("\tFind Owned/Shared Boundary Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_findOwnedSharedBndryTime/_nAssemble);
    printf("\tFill Owned/Shared Time (%d samples)=%1.5f msec\n",
	   _nAssemble,_fillOwnedSharedTime/_nAssemble);
  }

  /* destroy events */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_start));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_start_refined));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_stop_refined));
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

  if (_d_workspace) { ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_workspace)); _d_workspace=NULL; }

}


double
HypreCudaMatrixAssembler::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}


void
HypreCudaMatrixAssembler::deviceMemoryInGBs(double & free, double & total) const {
  size_t f=0, t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemGetInfo(&f, &t));
  free = 1.0*f/(1024.*1024.*1024.);
  total = 1.0*t/(1024.*1024.*1024.);
}


void
HypreCudaMatrixAssembler::copyCSRMatrixToHost() {

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
#endif

  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_row_indices, _d_row_indices, _num_rows*sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_row_counts, _d_row_counts, _num_rows*sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_col_indices, _d_col_indices, _num_nonzeros*sizeof(HypreIntType), cudaMemcpyDeviceToHost));
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_values, _d_values, _num_nonzeros*sizeof(double), cudaMemcpyDeviceToHost));

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
#endif
}


void
HypreCudaMatrixAssembler::assemble(Kokkos::View<HypreIntType *>& cols, Kokkos::View<double *>& data) {

#ifdef HYPRE_CUDA_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s\n",_rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  float t=0;
  _nAssemble++;
#endif

  _d_cols = cols.data();
  _d_data = data.data();

  /*********************************************************************/
  /* Get the raw pointers of the assembly Kokkos views data structures */
  /*********************************************************************/
  HypreIntType * _d_assembly_row_start_ptr = _d_assembly_row_start.data();
  HypreIntType * _d_assembly_row_indices_ptr = _d_assembly_row_indices.data();
  
  /* dense key */
  HypreIntType * _d_key = (HypreIntType *)_d_workspace;
  HypreIntType * _d_matelem_bin_ptrs = (HypreIntType *)(_d_key + (_nDataPtsToAssemble+1));
  HypreIntType * _d_locations = (HypreIntType *)(_d_matelem_bin_ptrs + (_nDataPtsToAssemble+1));
  HypreIntType * _d_matelem_bin_ptrs_final = (HypreIntType *)(_d_locations + (_nDataPtsToAssemble+1));


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* record the start time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start_refined));
#endif

  
  if (_ensure_reproducible) {
    /*********************************************************************************************************************/
    /* Ensure Reproducible : Here we sort on absolute value first, then on the dense index. This ensures reproducibility */
    /*********************************************************************************************************************/
#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
    sortCooAscendingCub(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_assembly_row_indices_ptr,
				 _d_assembly_row_start_ptr, _d_workspace, _d_cols, _d_data);
#else
    sortCooAscendingThrust(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_assembly_row_indices_ptr,
				    _d_assembly_row_start_ptr, _d_workspace, _d_cols, _d_data);
#endif
  } else {
    /****************************************/
    /* Sort : based on the dense index only */
    /****************************************/
#ifdef HYPRE_CUDA_ASSEMBLER_USE_CUB
    sortCooCub(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_assembly_row_indices_ptr,
			_d_assembly_row_start_ptr, _d_workspace, _d_cols, _d_data);
#else
    sortCooThrust(_nDataPtsToAssemble, _num_rows, _global_num_cols, _d_assembly_row_indices_ptr,
			   _d_assembly_row_start_ptr, _d_workspace, _d_cols, _d_data);
#endif
  }
  HypreIntType * ptr = (HypreIntType *)(_d_workspace)+_nDataPtsToAssemble;
  CUDA_SAFE_CALL(cudaMemset(ptr, 0, (3*(_nDataPtsToAssemble+1)+1)*sizeof(HypreIntType)));


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop_refined));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop_refined));
  t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start_refined, _stop_refined));
  _sortTime+=t;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start_refined));
#endif


  /********************************************************/
  /* The next 3 kernels figures out the boundaries of the */
  /* matrix elements in linear memory.                    */   
  /********************************************************/

  /* The initial kernel for determining the matrix element locations in linear memory */
  int num_threads=128;
  int num_blocks = (_nDataPtsToAssemble + num_threads - 1)/num_threads;  
#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : _nDataPtsToAssemble=%lld, num_blocks=%d\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,num_blocks);
#endif

  matrixElemLocationsInitialKernel<<<num_blocks,num_threads>>>(_d_key, _nDataPtsToAssemble, _global_num_cols, 
						_d_matelem_bin_ptrs, _d_locations);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());

  
  /* inclusive scan : this determines the number of nonzeros in the matrix*/
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

#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
  printf("Rank %d %s %s %d : name=%s : _num_nonzeros=%lld, _num_rows=%lld\n",
    _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_num_nonzeros,_num_rows);
#endif

  /* The final kernel for determining the matrix element locations in linear memory */
  num_blocks = (_nDataPtsToAssemble + 1 + num_threads - 1)/num_threads;  
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_matelem_bin_ptrs_final, 0, (_nDataPtsToAssemble+1)*sizeof(HypreIntType)));
  matrixElemLocationsFinalKernel<<<num_blocks,num_threads>>>(_num_nonzeros, _nDataPtsToAssemble,
						     _d_matelem_bin_ptrs, _d_locations, _d_matelem_bin_ptrs_final);  
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop_refined));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop_refined));
  t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start_refined, _stop_refined));
  _matElemBndryTime+=t;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start_refined));
#endif


  /*****************************/
  /*     Allocate space        */
  /*****************************/

  /* Step 8 : allocate space */
  if (!_csrMatMemoryAdded) {
    ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_col_indices, _num_nonzeros*sizeof(HypreIntType)));
    ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_values, _num_nonzeros*sizeof(double)));
    ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_col_indices, _num_nonzeros*sizeof(HypreIntType)));
    ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_values, _num_nonzeros*sizeof(double)));
    _memoryUsed += (_num_nonzeros)*(sizeof(HypreIntType) + sizeof(double));
    _csrMatMemoryAdded = true;
  }

#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
  double free=0., total=0.;
  deviceMemoryInGBs(free,total);
  printf("Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf,  Free Device Memory=%1.6lf,  TotalDeviceMemory=%1.6lf \n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs(),free,total);
#endif


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop_refined));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop_refined));
  t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start_refined, _stop_refined));
  _matAllocateTime+=t;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start_refined));
#endif


  /***********************************************/
  /*          CSR Matrix Computation             */
  /***********************************************/

  /* first reset the row counts */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_row_counts, 0, _num_rows*sizeof(unsigned long long int)));

  /* copy from kokkos to internal data structure ... which will be manipulated later */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_row_indices, _d_assembly_row_indices_ptr, _num_rows*sizeof(HypreIntType), 
					     cudaMemcpyDeviceToDevice));

  /* Expand assembly_row_start into a vector of the same size as cols/data */
  num_threads=128;
  num_blocks = _num_rows;
  HypreIntType * _d_assembly_row_start_expanded = (HypreIntType *)_d_workspace;
  fillRowStartExpandedKernel<<<num_blocks,num_threads>>>(_num_rows, _d_assembly_row_start_ptr, _d_assembly_row_start_expanded);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  
    
  /* Fill the CSR Matrix */
  num_threads=128;
  int threads_per_row = nextPowerOfTwo((_nDataPtsToAssemble + _num_nonzeros - 1)/_num_nonzeros);
  if (threads_per_row>32) threads_per_row=32;
  int num_rows_per_block = num_threads/threads_per_row;
  num_blocks = (_num_nonzeros + num_rows_per_block - 1)/num_rows_per_block;
  fillCSRMatrix<<<num_blocks,num_threads>>>(_num_nonzeros, threads_per_row,
					    _d_matelem_bin_ptrs_final, _d_assembly_row_start_expanded,
					    _d_cols, _d_data, _d_row_counts,
					    _d_col_indices, _d_values);
  ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop_refined));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop_refined));
  t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start_refined, _stop_refined));
  _fillCSRMatTime+=t;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start_refined));
#endif


  /*******************************************************/
  /*      Split into owned and shared not-owned rows     */
  /*******************************************************/

#ifdef HYPRE_CUDA_ASSEMBLER_CUSTOM_SCAN
  /* Use the custom exclusive_scan version since thrust allocates memory on device */
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_workspace, 0, (_num_rows+1)*sizeof(HypreIntType)));
  d_work = (HypreIntType*)_d_workspace + _nDataPtsToAssemble+1;
  exclusive_scan(_d_workspace, (HypreIntType *)_d_row_counts, d_work, _num_rows+1);    
#else
/* Use thrust which seems a little faster */
  thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(_d_row_counts),
   			 thrust::device_pointer_cast(_d_row_counts+_num_rows+1),
   			 thrust::device_pointer_cast((HypreIntType*)_d_workspace));
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
    
#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
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
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&nnz_lower, (HypreIntType *)_d_workspace+iLowLoc,
				      sizeof(HypreIntType), cudaMemcpyDeviceToHost));

  /* owned num_nonzeros is the value of the exclusive scan at iUppLoc */
  HypreIntType nnz_upper=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&nnz_upper, (HypreIntType *)_d_workspace+iUppLoc+1,
				      sizeof(HypreIntType), cudaMemcpyDeviceToHost));


#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop_refined));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop_refined));
  t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start_refined, _stop_refined));
  _findOwnedSharedBndryTime+=t;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start_refined));
#endif


#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
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
  if ((_num_rows_owned!=_num_rows && _num_nonzeros_owned==_num_nonzeros) ||
      (_num_rows_owned==_num_rows && _num_nonzeros_owned!=_num_nonzeros)) {
    /* This is inconsistent. Need to throw an exception */
    throw std::runtime_error("Inconsistency detected. This should not happen.");
  }

  _has_shared=false;
  if (_num_rows_owned!=_num_rows && _num_nonzeros_owned!=_num_nonzeros) {
    _has_shared = true;

    if (_iLower==0) {

#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
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

#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
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

#ifdef HYPRE_MATRIX_ASSEMBLER_DEBUG
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
	SWAP(_d_row_indices, _d_row_indices+iLowLoc, (HypreIntType *)_d_workspace, n1, n2, n3, s);
	
	s = sizeof(unsigned long long int);
	SWAP(_d_row_counts, _d_row_counts+iLowLoc, (HypreIntType *)_d_workspace, n1, n2, n3, s);
	
	n1 = nnz_lower, n2=nnz_upper-nnz_lower, n3= _nDataPtsToAssemble;
	s = sizeof(HypreIntType);
	SWAP(_d_col_indices, _d_col_indices+nnz_lower, (HypreIntType *)_d_workspace, n1, n2, n3, s);
	
	s = sizeof(double);
	SWAP(_d_values, _d_values+nnz_lower, (double *)_d_workspace, n1, n2, n3, s);
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
  }

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop_refined));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop_refined));
  t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start_refined, _stop_refined));
  _fillOwnedSharedTime+=t;

  /* record the stop time */
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  t=0;
  ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _assembleTime+=t;
#endif
}

#endif // KOKKOS_ENABLE_CUDA

}  // nalu
}  // sierra
