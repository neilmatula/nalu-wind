#ifndef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_H
#define HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_H

#ifndef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
#define HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
#endif // HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG
#undef HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_DEBUG

#include <Kokkos_Sort.hpp>

namespace sierra {
namespace nalu {

class KokkosMatrixAssembler {

public:
  /**
   * MatrixAssembler Constructor 
   *
   * @param sort whether or not to sort the CSR matrix (prior to full assembly) based on the element ids AND the values
   * @param iLower the first row
   * @param iUpper the ending row (inclusive)
   * @param jLower the first column
   * @param jUpper the ending column (inclusive)
   * @param global_num_rows the number of rows in the global matrix
   * @param global_num_cols the number of columns in the global matrix
   * @param nDataPtsToAssemble the number of data points to assemble into a CSR matrix
   * @param rank the mpi rank
   * @param num_rows the number of rows in the kokkos data structure
   * @param kokkos_row_indices Kokkos View for the row coordinates
   * @param kokkos_row_start Kokkos View to the start of the data for the rows
   */
  KokkosMatrixAssembler(std::string name, bool sort, HypreIntType iLower, HypreIntType iUpper, HypreIntType jLower, HypreIntType jUpper,
			HypreIntType global_num_rows, HypreIntType global_num_cols, HypreIntType nDataPtsToAssemble, int rank,
			HypreIntType num_rows, Kokkos::View<HypreIntType *>& kokkos_row_indices, Kokkos::View<HypreIntType *>& kokkos_row_start);

  /**
   *  Destructor 
   */
  virtual ~KokkosMatrixAssembler();

  /**
   * memoryInGBS computes the amount of device memory used in GBs
   *
   * @return the amount of device memory used in GBs
   */
  double memoryInGBs() const;

  /**
   * getHasShared gets whether or not this has a shared matrix
   *
   * @return whether or not this has a shared matrix
   */
  bool getHasShared() const { return _has_shared; }

  /**
   * getNumRows gets the number of rows in both the owned and shared parts
   *
   * @return the number of rows in both the owned and shared parts
   */
  HypreIntType getNumRows() const { return _num_rows; }

  /**
   * getNumRowsOwned gets the number of rows in the owned part
   *
   * @return the number of rows in the owned part
   */
  HypreIntType getNumRowsOwned() const { return _num_rows_owned; }

  /**
   * getNumRowsShared gets the number of rows in the shared part
   *
   * @return the number of rows in the shared part
   */
  HypreIntType getNumRowsShared() const { return _num_rows_shared; }

  /**
   * getNumNonzeros gets the number of nonzeros in both the owned and shared parts
   *
   * @return the number of nonzeros in both the owned and shared parts
   */
  HypreIntType getNumNonzeros() const { return _num_nonzeros; }

  /**
   * getNumNonzerosOwned gets the number of nonzeros in the owned part
   *
   * @return the number of nonzeros in the owned part
   */
  HypreIntType getNumNonzerosOwned() const { return _num_nonzeros_owned; }

  /**
   * getNumNonzerosShared gets the number of nonzeros in the shared part
   *
   * @return the number of nonzeros in he shared part
   */
  HypreIntType getNumNonzerosShared() const { return _num_nonzeros_shared; }

  /**
   * copyCSRMatrixToHost copies the assembled CSR matrix to the host (page locked memory)
   */
  void copyCSRMatrixToHost();

  /**
   * copyOwnedCSRMatrixToHost copies the assembled owned CSR matrix to the host (page locked memory)
   */
  void copyOwnedCSRMatrixToHost();

  /**
   * copySharedCSRMatrixToHost copies the assembled shared CSR matrix to the host (page locked memory)
   */
  void copySharedCSRMatrixToHost();

  /**
   * assemble : assemble the symbolic and numeric parts of the CSR matrix
   *
   */
  void assemble(Kokkos::View<HypreIntType *>& cols, Kokkos::View<double *>& data);

  /**
   * get the host row indices ptr in page locked memory
   *
   * @return the pointer to the host row indices
   */
  HypreIntType * getHostRowIndicesPtr() { return _h_row_indices.data(); }

  /**
   * get the host row counts ptr in page locked memory
   *
   * @return the pointer to the host row counts
   */
  HypreIntType * getHostRowCountsPtr() { return _h_row_counts.data(); }

  /**
   * get the host column indices ptr in page locked memory
   *
   * @return the pointer to the host column indices 
   */
  HypreIntType * getHostColIndicesPtr() { return _h_col_indices.data(); }

  /**
   * get the host values ptr in page locked memory
   *
   * @return the pointer to the host values
   */
  double * getHostValuesPtr() { return _h_values.data(); }

  /**
   * get the host owned row indices ptr in page locked memory
   *
   * @return the pointer to the host owned row indices
   */
  HypreIntType * getHostOwnedRowIndicesPtr() { return _h_row_indices_owned.data(); }

  /**
   * get the host owned row counts ptr in page locked memory
   *
   * @return the pointer to the host owned row counts
   */
  HypreIntType * getHostOwnedRowCountsPtr() { return _h_row_counts_owned.data(); }
  
  /**
   * get the host owned column indices ptr in page locked memory
   *
   * @return the pointer to the host owned column indices 
   */
  HypreIntType * getHostOwnedColIndicesPtr() { return _h_col_indices_owned.data(); }

  /**
   * get the host owned values ptr in page locked memory
   *
   * @return the pointer to the host owned values
   */
  double * getHostOwnedValuesPtr() { return _h_values_owned.data(); }

  /**
   * get the host shared row indices ptr in page locked memory
   *
   * @return the pointer to the host shared row indices
   */
  HypreIntType * getHostSharedRowIndicesPtr() { return _h_row_indices_shared.data(); }

  /**
   * get the host shared row counts ptr in page locked memory
   *
   * @return the pointer to the host shared row counts
   */
  HypreIntType * getHostSharedRowCountsPtr() { return _h_row_counts_shared.data(); }
  
  /**
   * get the host shared column indices ptr in page locked memory
   *
   * @return the pointer to the host shared column indices 
   */
  HypreIntType * getHostSharedColIndicesPtr() { return _h_col_indices_shared.data(); }

  /**
   * get the host shared values ptr in page locked memory
   *
   * @return the pointer to the host shared values
   */
  double * getHostSharedValuesPtr() { return _h_values_shared.data(); }


  /* amount of memory being used */
  HypreIntType _memoryUsed=0;

  /* The final csr matrix pointers */
  HypreIntType _num_rows=0;
  HypreIntType _num_nonzeros=0;
  Kokkos::View<HypreIntType *> _d_row_indices;
  Kokkos::View<HypreIntType *> _d_row_counts;
  Kokkos::View<HypreIntType *> _d_col_indices;
  Kokkos::View<double *> _d_values;

  Kokkos::View<HypreIntType *>::HostMirror _h_row_indices;
  Kokkos::View<HypreIntType *>::HostMirror _h_row_counts;
  Kokkos::View<HypreIntType *>::HostMirror _h_col_indices;
  Kokkos::View<double *>::HostMirror _h_values;

  /* owned CSR matrix */
  HypreIntType _num_rows_owned=0;
  HypreIntType _num_nonzeros_owned=0;  
  Kokkos::View<HypreIntType *> _d_row_indices_owned;
  Kokkos::View<HypreIntType *> _d_row_counts_owned;
  Kokkos::View<HypreIntType *> _d_col_indices_owned;
  Kokkos::View<double *> _d_values_owned;

  Kokkos::View<HypreIntType *>::HostMirror _h_row_indices_owned;
  Kokkos::View<HypreIntType *>::HostMirror _h_row_counts_owned;
  Kokkos::View<HypreIntType *>::HostMirror _h_col_indices_owned;
  Kokkos::View<double *>::HostMirror _h_values_owned;

  /* shared (not owned) CSR matrix */
  HypreIntType _num_rows_shared=0;
  HypreIntType _num_nonzeros_shared=0;  
  Kokkos::View<HypreIntType *> _d_row_indices_shared;
  Kokkos::View<HypreIntType *> _d_row_counts_shared;
  Kokkos::View<HypreIntType *> _d_col_indices_shared;
  Kokkos::View<double *> _d_values_shared;

  Kokkos::View<HypreIntType *>::HostMirror _h_row_indices_shared;
  Kokkos::View<HypreIntType *>::HostMirror _h_row_counts_shared;
  Kokkos::View<HypreIntType *>::HostMirror _h_col_indices_shared;
  Kokkos::View<double *>::HostMirror _h_values_shared;

  /* meta data */
  std::string _name="";
  bool _sort=false;
  HypreIntType _iLower=0;
  HypreIntType _iUpper=0;
  HypreIntType _jLower=0;
  HypreIntType _jUpper=0;
  HypreIntType _global_num_rows=0;
  HypreIntType _global_num_cols=0;
  HypreIntType _num_rows_this_rank=0;
  HypreIntType _num_cols_this_rank=0;
  HypreIntType _nDataPtsToAssemble=0;
  int _rank=0;
  bool _csrMatMemoryAdded=false;
  bool _has_shared=false;
  /* flag for allocating only once */
  bool _owned_shared_views_created=false;

  /* the kokkos pointers to the row indices and row starts */
  Kokkos::View<HypreIntType *> _d_kokkos_row_indices;
  Kokkos::View<HypreIntType *> _d_kokkos_row_start;

  /* temporaries/scratch space */
  Kokkos::View<HypreIntType *> _d_dense_keys;
  Kokkos::View<HypreIntType *> _d_mat_elem_bin_locs;
  Kokkos::View<HypreIntType *> _d_mat_elem_bins;
  Kokkos::View<HypreIntType *> _d_transitions;
  Kokkos::View<HypreIntType *> _d_row_counts_scanned;
  Kokkos::View<HypreIntType *>::HostMirror _h_row_counts_scanned;
};


class KokkosRhsAssembler {

public:

  /**
   * KokkosRhsAssembler Constructor 
   *
   * @param name of the linear system being assembled
   * @param sort whether or not to sort the rhs vector (prior to full assembly) based on the element ids AND the values
   * @param iLower the first row
   * @param iUpper the ending row (inclusive)
   * @param global_num_rows the number of rows in the global matrix
   * @param nDataPtsToAssemble the number of data points to assemble into a rhs vector
   * @param rank the mpi rank
   * @param num_rows the number of rows in the kokkos data structure
   * @param kokkos_row_indices Kokkos View for the row coordinates
   * @param kokkos_row_start Kokkos View to the start of data for the rowss
   */
  KokkosRhsAssembler(std::string name, bool sort, HypreIntType iLower, HypreIntType iUpper,
		     HypreIntType global_num_rows, HypreIntType nDataPtsToAssemble, int rank,
		     HypreIntType num_rows, Kokkos::View<HypreIntType *>& kokkos_row_indices,
		     Kokkos::View<HypreIntType *>& kokkos_row_start);

  /**
   *  Destructor 
   */
  virtual ~KokkosRhsAssembler();

  /**
   * memoryInGBS computes the amount of device memory used in GBs
   *
   * @return the amount of device memory used in GBs
   */
  double memoryInGBs() const;

  /**
   * getHasShared gets whether or not this has a shared matrix
   *
   * @return whether or not this has a shared matrix
   */
  bool getHasShared() const { return _has_shared; }

  /**
   * getNumRows gets the number of rows in both the owned and shared parts
   *
   * @return the number of rows in both the owned and shared parts
   */
  HypreIntType getNumRows() const { return _num_rows; }

  /**
   * getNumRowsOwned gets the number of rows in the owned part
   *
   * @return the number of rows in the owned part
   */
  HypreIntType getNumRowsOwned() const { return _num_rows_owned; }

  /**
   * getNumRowsShared gets the number of rows in the shared part
   *
   * @return the number of rows in the shared part
   */
  HypreIntType getNumRowsShared() const { return _num_rows_shared; }

  /**
   * copyRhsVectorToHost copies the assembled RhsVector to the host (page locked memory)
   */
  void copyRhsVectorToHost();

  /**
   * copyOwnedRhsVectorToHost copies the assembled owned RhsVector to the host (page locked memory)
   */
  void copyOwnedRhsVectorToHost();

  /**
   * copySharedRhsVectorToHost copies the assembled shared RhsVector to the host (page locked memory)
   */
  void copySharedRhsVectorToHost();

  /**
   * assemble : assemble the symbolic and numeric parts of the CSR matrix
   *
   */
  void assemble(Kokkos::View<double **>& data, const int index);

  /**
   * get the host rhs ptr in page locked memory
   *
   * @return the pointer to the host rhs
   */
  double * getHostRhsPtr() { return _h_rhs.data(); }

  /**
   * get the host rhs indices ptr in page locked memory
   *
   * @return the pointer to the host rhs indices
   */
  HypreIntType * getHostRhsIndicesPtr() { return _h_rhs_indices.data(); }

  /**
   * get the host owned rhs ptr in page locked memory
   *
   * @return the pointer to the host owned rhs
   */
  double * getHostOwnedRhsPtr() { return _h_rhs_owned.data(); }

  /**
   * get the host owned rhs indices ptr in page locked memory
   *
   * @return the pointer to the host owned rhs indices
   */
  HypreIntType * getHostOwnedRhsIndicesPtr() { return _h_rhs_indices_owned.data(); }

  /**
   * get the host shared rhs ptr in page locked memory
   *
   * @return the pointer to the host shared rhs
   */
  double * getHostSharedRhsPtr() { return _h_rhs_shared.data(); }

  /**
   * get the host shared rhs indices ptr in page locked memory
   *
   * @return the pointer to the host shared rhs indices
   */
  HypreIntType * getHostSharedRhsIndicesPtr() { return _h_rhs_indices_shared.data(); }

  /* amount of memory being used */
  HypreIntType _memoryUsed=0;

  /* The final rhs vector */
  HypreIntType _num_rows=0;
  Kokkos::View<double *> _d_rhs;
  Kokkos::View<HypreIntType *> _d_rhs_indices;
  Kokkos::View<double *>::HostMirror _h_rhs;
  Kokkos::View<HypreIntType *>::HostMirror _h_rhs_indices;

  /* The owned rhs vector */
  HypreIntType _num_rows_owned=0;
  Kokkos::View<double *> _d_rhs_owned;
  Kokkos::View<HypreIntType *> _d_rhs_indices_owned;
  Kokkos::View<double *>::HostMirror _h_rhs_owned;
  Kokkos::View<HypreIntType *>::HostMirror _h_rhs_indices_owned;

  /* The shared rhs vector */
  HypreIntType _num_rows_shared=0;
  Kokkos::View<double *> _d_rhs_shared;
  Kokkos::View<HypreIntType *> _d_rhs_indices_shared;
  Kokkos::View<double *>::HostMirror _h_rhs_shared;
  Kokkos::View<HypreIntType *>::HostMirror _h_rhs_indices_shared;

  /* flag for allocating only once */
  bool _owned_shared_views_created=false;

  /* meta data */
  std::string _name="";
  bool _sort=false;
  HypreIntType _iLower=0;
  HypreIntType _iUpper=0;
  HypreIntType _global_num_rows=0;
  HypreIntType _num_rows_this_rank=0;
  HypreIntType _nDataPtsToAssemble=0;
  int _rank=0;
  bool _has_shared=false;

  /* the kokkos pointers to the row indices and row starts */
  Kokkos::View<HypreIntType *> _d_kokkos_row_indices;
  Kokkos::View<HypreIntType *> _d_kokkos_row_start;

  /* temporaries/scratch space */
  Kokkos::View<HypreIntType *> _d_int_workspace;
  Kokkos::View<double *> _d_double_workspace;

};

}  // nalu
}  // sierra

#endif /* HYPRE_KOKKOS_LINEAR_SYSTEM_ASSEMBLER_H */
