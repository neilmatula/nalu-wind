#include "HypreLinearSystem.h"
#include "HypreKokkosLinearSystemAssembler.h"

namespace sierra {
namespace nalu {



/* --------------------------------------------------------------------------------------------------------- */
/*                                     Kokkos Matrix Assembler Class                                         */
/* --------------------------------------------------------------------------------------------------------- */

KokkosMatrixAssembler::KokkosMatrixAssembler(std::string name, bool sort, HypreIntType iLower,
					     HypreIntType iUpper, HypreIntType jLower, HypreIntType jUpper,
					     HypreIntType global_num_rows, HypreIntType global_num_cols,
					     HypreIntType nDataPtsToAssemble, int rank,
					     HypreIntType num_rows, Kokkos::View<HypreIntType *>& kokkos_row_indices,
					     Kokkos::View<HypreIntType *>& kokkos_row_start)
  : _name(name), _sort(sort), _iLower(iLower), _iUpper(iUpper),
    _jLower(jLower), _jUpper(jUpper), _global_num_rows(global_num_rows), _global_num_cols(global_num_cols),
    _nDataPtsToAssemble(nDataPtsToAssemble), _rank(rank), _num_rows(num_rows),
    _d_kokkos_row_indices(kokkos_row_indices), _d_kokkos_row_start(kokkos_row_start)
{
#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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
 
#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Done Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs());
#endif
}

KokkosMatrixAssembler::~KokkosMatrixAssembler() {
}


double KokkosMatrixAssembler::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}


void KokkosMatrixAssembler::copyCSRMatrixToHost() {
  if (getHasShared()) {
    copyOwnedCSRMatrixToHost();
    copySharedCSRMatrixToHost();
  } else {
    Kokkos::deep_copy(_h_row_indices, _d_row_indices);
    Kokkos::deep_copy(_h_row_counts, _d_row_counts);
    Kokkos::deep_copy(_h_col_indices, _d_col_indices);
    Kokkos::deep_copy(_h_values, _d_values);
  }
}


void KokkosMatrixAssembler::copyOwnedCSRMatrixToHost() {
  Kokkos::deep_copy(_h_row_indices_owned, _d_row_indices_owned);
  Kokkos::deep_copy(_h_row_counts_owned, _d_row_counts_owned);
  Kokkos::deep_copy(_h_col_indices_owned, _d_col_indices_owned);
  Kokkos::deep_copy(_h_values_owned, _d_values_owned);
}


void KokkosMatrixAssembler::copySharedCSRMatrixToHost() {
  Kokkos::deep_copy(_h_row_indices_shared, _d_row_indices_shared);
  Kokkos::deep_copy(_h_row_counts_shared, _d_row_counts_shared);
  Kokkos::deep_copy(_h_col_indices_shared, _d_col_indices_shared);
  Kokkos::deep_copy(_h_values_shared, _d_values_shared);
}


void KokkosMatrixAssembler::assemble(Kokkos::View<HypreIntType *>& cols, Kokkos::View<double *>& data) {

  /* team for parallel for loops */
  auto num_rows = _num_rows;
  auto team_exec = get_device_team_policy(num_rows, 0, 0);
  auto nDataPtsToAssemble = _nDataPtsToAssemble;
  auto kokkos_row_start = _d_kokkos_row_start;
  auto kokkos_row_indices = _d_kokkos_row_indices;
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

  /*************************************************************************************/
  /* Create the dense indices key vector that is the same size as the data/cols buffer */
  /*************************************************************************************/
  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {

      auto rowId = team.league_rank();
      auto begin = kokkos_row_start(rowId);
      auto end = kokkos_row_start(rowId+1);
      auto row = kokkos_row_indices(rowId);
      auto rowLen = end - begin;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, rowLen),
			   [&](const HypreIntType& i) { d_dense_keys(begin+i) = row * global_num_cols + d_cols(begin+i); });
    });

  if (_sort) {
    /******************************************************************************************************/
    /* Sort : Here we sort on absolute value first, then on the dense index. This ensures reproducibility */
    /******************************************************************************************************/
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
#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
    printf("Rank %d %s %s %d : name=%s : myMin=%lld, myMax=%lld\n",
	   _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),result.min_val,result.max_val);
#endif
    Kokkos::BinSort<KeyViewType, CompType> bin_sort(d_dense_keys, CompType(d_dense_keys.extent(0) / 2, result.min_val, result.max_val), true);
    bin_sort.create_permute_vector();
    bin_sort.sort(d_dense_keys);
    bin_sort.sort(d_cols);
    bin_sort.sort(d_data);
  }

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
  
#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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
  Kokkos::deep_copy(_d_row_indices, _d_kokkos_row_indices);
  
  /* Expand kokkos_row_start into a vector of the same size as cols/data */
  team_exec = get_device_team_policy(num_rows, 0, 0);
  auto d_kokkos_row_start_expanded = _d_dense_keys;
  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {

      auto rowId = team.league_rank();
      auto begin = kokkos_row_start(rowId);
      auto end = kokkos_row_start(rowId+1);
      auto rowLen = end - begin;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, rowLen),
			   [&](const HypreIntType& i) { d_kokkos_row_start_expanded(begin+i) = rowId; });
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
      auto rbegin = d_kokkos_row_start_expanded(begin);

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


#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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

#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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

#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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

#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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

#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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


/* --------------------------------------------------------------------------------------------------------- */
/*                                     Kokkos RHS Assembler Class                                            */
/* --------------------------------------------------------------------------------------------------------- */

KokkosRhsAssembler::KokkosRhsAssembler(std::string name, bool sort, HypreIntType iLower, HypreIntType iUpper,
				       HypreIntType global_num_rows, HypreIntType nDataPtsToAssemble, int rank,
				       HypreIntType num_rows, Kokkos::View<HypreIntType *>& kokkos_row_indices,
				       Kokkos::View<HypreIntType *>& kokkos_row_start)
  : _name(name), _sort(sort), _iLower(iLower), _iUpper(iUpper),
    _global_num_rows(global_num_rows), _nDataPtsToAssemble(nDataPtsToAssemble), _rank(rank), _num_rows(num_rows),
    _d_kokkos_row_indices(kokkos_row_indices), _d_kokkos_row_start(kokkos_row_start)
{
#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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

#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Done Rank %d %s %s %d : name=%s : nDataPtsToAssemble=%lld,  Used Memory (this class) GBs=%1.6lf\n",
	 _rank,__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs());
#endif
}

KokkosRhsAssembler::~KokkosRhsAssembler() {
}

double KokkosRhsAssembler::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}

void KokkosRhsAssembler::copyRhsVectorToHost() {
  if (getHasShared()) {
    copyOwnedRhsVectorToHost();
    copySharedRhsVectorToHost();
  } else {
    Kokkos::deep_copy(_h_rhs, _d_rhs);
    Kokkos::deep_copy(_h_rhs_indices, _d_rhs_indices);
  }
}

void KokkosRhsAssembler::copyOwnedRhsVectorToHost() {
  Kokkos::deep_copy(_h_rhs_owned, _d_rhs_owned);
  Kokkos::deep_copy(_h_rhs_indices_owned, _d_rhs_indices_owned);
}

void KokkosRhsAssembler::copySharedRhsVectorToHost() {
  Kokkos::deep_copy(_h_rhs_shared, _d_rhs_shared);
  Kokkos::deep_copy(_h_rhs_indices_shared, _d_rhs_indices_shared);
}

void KokkosRhsAssembler::assemble(Kokkos::View<double **>& data, const int index) {

  /* team for parallel for loops */
  auto num_rows = _num_rows;
  auto team_exec = get_device_team_policy(num_rows, 0, 0);
  //auto nDataPtsToAssemble = _nDataPtsToAssemble;
  auto kokkos_row_start = _d_kokkos_row_start;
  auto kokkos_row_indices = _d_kokkos_row_indices;
  auto d_int_workspace = _d_int_workspace;
  auto d_double_workspace = _d_double_workspace;
  auto d_data = data;
  auto d_rhs = _d_rhs;

  if (_sort) {
    throw std::runtime_error("This option is NOT supported. Set ensure_reproducibile=no in the Hypre linear solver blocks.");

    // /* Create the row indices vector that is the same size as the _d_data buffer */
    // Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {

    //     auto rowId = team.league_rank();
    // 	auto begin = kokkos_row_start(rowId);
    // 	auto end = kokkos_row_start(rowId+1);
    // 	auto row = kokkos_row_indices(rowId);
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

  /* copy kokkos_row_indices to rhs_indices so that rhs_indices can be manipulated by the owned/shared computation */
  Kokkos::deep_copy(_d_rhs_indices, _d_kokkos_row_indices);

  /**************************************/
  /* reduce the right hand size elements*/
  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {

      auto rowId = team.league_rank();
      auto begin = kokkos_row_start(rowId);
      auto end = kokkos_row_start(rowId+1);
      auto row = kokkos_row_indices(rowId);
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

#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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

#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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

#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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

#ifdef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
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

  
}

}  // nalu
}  // sierra
