#ifndef HYPRE_MATRIX_ASSEMBLER_H
#define HYPRE_MATRIX_ASSEMBLER_H

#ifndef HYPRE_MATRIX_ASSEMBLER_DEBUG
#define HYPRE_MATRIX_ASSEMBLER_DEBUG
#endif // HYPRE_MATRIX_ASSEMBLER_DEBUG
#undef HYPRE_MATRIX_ASSEMBLER_DEBUG

#ifndef HYPRE_MATRIX_ASSEMBLER_TIMER
#define HYPRE_MATRIX_ASSEMBLER_TIMER
#endif // HYPRE_MATRIX_ASSEMBLER_TIMER
#undef HYPRE_MATRIX_ASSEMBLER_TIMER

#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sys/time.h>

#include "HypreCudaAssembler.h"
#include <Kokkos_Sort.hpp>

namespace sierra {
namespace nalu {

class HypreMatrixAssembler {

public:

  /**
   * HypreKokkosMatrixAssembler Constructor 
   *
   * @param choice specific type of assembler to be used: choice==0, Kokkos, choice==1, Cuda
   * @param name of the linear system being assembled
   * @param ensureReproducible will sort the values based on absolute value ascending to ensure that when the input is the
   *   same, though in randomized order potentially, the same RHS is always produced.
   * @param iLower the first row
   * @param iUpper the ending row (inclusive)
   * @param jLower the first column
   * @param jUpper the ending column (inclusive)
   * @param global_num_rows the number of rows in the global matrix
   * @param global_num_cols the number of columns in the global matrix
   * @param nDataPtsToAssemble the number of data points to assemble into a CSR matrix
   * @param rank the mpi rank
   * @param num_rows the number of rows in the kokkos data structure
   * @param assembly_row_indices Kokkos View for the row coordinates
   * @param assembly_row_start Kokkos View to the start of the data for the rows
   *
   * @return a pointer to HypreMatrixAssembler
   */
  static HypreMatrixAssembler * make_HypreMatrixAssembler(int choice, std::string name, bool ensureReproducible,
							  HypreIntType iLower, HypreIntType iUpper,
							  HypreIntType jLower, HypreIntType jUpper, HypreIntType global_num_rows,
							  HypreIntType global_num_cols, HypreIntType nDataPtsToAssemble, int rank,
							  HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
							  Kokkos::View<HypreIntType *>& assembly_row_start);

  /**
   *  Destructor 
   */
  virtual ~HypreMatrixAssembler() {}

  /**
   * assemble : assemble the symbolic and numeric parts of the CSR matrix
   *
   * @param cols Kokkos View of the matrix column indices list
   * @param data Kokkos View of the matrix data list
   */
  virtual void assemble(Kokkos::View<HypreIntType *>& cols, Kokkos::View<double *>& data) = 0;

  /**
   * copyCSRMatrixToHost copies the assembled CSR matrix to the host
   */
  virtual void copyCSRMatrixToHost() = 0;

  /**
   * getHasShared gets whether or not this has a shared matrix
   *
   * @return whether or not this has a shared matrix
   */
  virtual bool getHasShared() const { return _has_shared; };

  /**
   * getNumRows gets the number of rows in both the owned and shared parts
   *
   * @return the number of rows in both the owned and shared parts
   */
  virtual HypreIntType getNumRows() const { return _num_rows; };

  /**
   * getNumRowsOwned gets the number of rows in the owned part
   *
   * @return the number of rows in the owned part
   */
  virtual HypreIntType getNumRowsOwned() const { return _num_rows_owned; };

  /**
   * getNumRowsShared gets the number of rows in the shared part
   *
   * @return the number of rows in the shared part
   */
  virtual HypreIntType getNumRowsShared() const { return _num_rows_shared; };

  /**
   * getNumNonzeros gets the number of nonzeros in both the owned and shared parts
   *
   * @return the number of nonzeros in both the owned and shared parts
   */
  virtual HypreIntType getNumNonzeros() const { return _num_nonzeros; }

  /**
   * getNumNonzerosOwned gets the number of nonzeros in the owned part
   *
   * @return the number of nonzeros in the owned part
   */
  virtual HypreIntType getNumNonzerosOwned() const { return _num_nonzeros_owned; }

  /**
   * getNumNonzerosShared gets the number of nonzeros in the shared part
   *
   * @return the number of nonzeros in he shared part
   */
  virtual HypreIntType getNumNonzerosShared() const { return _num_nonzeros_shared; }

  /**
   * get the host row indices ptr in page locked memory
   *
   * @return the pointer to the host row indices
   */
  virtual HypreIntType * getHostRowIndicesPtr() const = 0;

  /**
   * get the host row counts ptr in page locked memory
   *
   * @return the pointer to the host row counts
   */
  virtual HypreIntType * getHostRowCountsPtr() const = 0;

  /**
   * get the host column indices ptr in page locked memory
   *
   * @return the pointer to the host column indices 
   */
  virtual HypreIntType * getHostColIndicesPtr() const = 0;

  /**
   * get the host values ptr in page locked memory
   *
   * @return the pointer to the host values
   */
  virtual double * getHostValuesPtr() const = 0;

  /**
   * get the host owned row indices ptr in page locked memory
   *
   * @return the pointer to the host owned row indices
   */
  virtual HypreIntType * getHostOwnedRowIndicesPtr() const = 0;

  /**
   * get the host owned row counts ptr in page locked memory
   *
   * @return the pointer to the host owned row counts
   */
  virtual HypreIntType * getHostOwnedRowCountsPtr() const = 0;
  
  /**
   * get the host owned column indices ptr in page locked memory
   *
   * @return the pointer to the host owned column indices 
   */
  virtual HypreIntType * getHostOwnedColIndicesPtr() const = 0;

  /**
   * get the host owned values ptr in page locked memory
   *
   * @return the pointer to the host owned values
   */
  virtual double * getHostOwnedValuesPtr() const = 0;

  /**
   * get the host shared row indices ptr in page locked memory
   *
   * @return the pointer to the host shared row indices
   */
  virtual HypreIntType * getHostSharedRowIndicesPtr() const = 0;

  /**
   * get the host shared row counts ptr in page locked memory
   *
   * @return the pointer to the host shared row counts
   */
  virtual HypreIntType * getHostSharedRowCountsPtr() const = 0;
  
  /**
   * get the host shared column indices ptr in page locked memory
   *
   * @return the pointer to the host shared column indices 
   */
  virtual HypreIntType * getHostSharedColIndicesPtr() const = 0;

  /**
   * get the host shared values ptr in page locked memory
   *
   * @return the pointer to the host shared values
   */
  virtual double * getHostSharedValuesPtr() const = 0;


protected:

  /* Protected constructor. Should only be called when instantiating derived class types */
  HypreMatrixAssembler(std::string name, bool ensureReproducible, HypreIntType iLower, HypreIntType iUpper,
		       HypreIntType jLower, HypreIntType jUpper, HypreIntType global_num_rows,
		       HypreIntType global_num_cols, HypreIntType nDataPtsToAssemble, int rank,
		       HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
		       Kokkos::View<HypreIntType *>& assembly_row_start) :
    _name(name), _ensure_reproducible(ensureReproducible), _iLower(iLower), _iUpper(iUpper), _jLower(jLower), _jUpper(jUpper),
    _global_num_rows(global_num_rows), _global_num_cols(global_num_cols), _nDataPtsToAssemble(nDataPtsToAssemble), _rank(rank),
    _num_rows(num_rows), _d_assembly_row_indices(assembly_row_indices), _d_assembly_row_start(assembly_row_start) {}
  
  /* input meta data */
  std::string _name="";
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

  /* Input number of rows. Owned and shared are computed quantities */
  HypreIntType _num_rows=0;
  HypreIntType _num_rows_owned=0;
  HypreIntType _num_rows_shared=0;
  /* Number of nonzeros. Computed in these classes */
  HypreIntType _num_nonzeros=0;
  HypreIntType _num_nonzeros_owned=0;  
  HypreIntType _num_nonzeros_shared=0;  

  /* amount of memory being used */
  HypreIntType _memoryUsed=0;

  /* whether or not to sort the values data to ensure reproducibility */
  bool _ensure_reproducible=false;

  /* whether or not this class contains shared rows for other ranks */
  bool _has_shared=false;

  /* flag for allocating matrix space only once */
  bool _csrMatMemoryAdded=false;

  /* the kokkos pointers to the row indices and row starts */
  Kokkos::View<HypreIntType *> _d_assembly_row_indices;
  Kokkos::View<HypreIntType *> _d_assembly_row_start;

};


class HypreKokkosMatrixAssembler : public HypreMatrixAssembler {

public:

  /**
   * HypreKokkosMatrixAssembler Constructor 
   *
   * @param name of the linear system being assembled
   * @param ensureReproducible will sort the values based on absolute value ascending to ensure that when the input is the
   *   same, though in randomized order potentially, the same RHS is always produced.
   * @param iLower the first row
   * @param iUpper the ending row (inclusive)
   * @param jLower the first column
   * @param jUpper the ending column (inclusive)
   * @param global_num_rows the number of rows in the global matrix
   * @param global_num_cols the number of columns in the global matrix
   * @param nDataPtsToAssemble the number of data points to assemble into a CSR matrix
   * @param rank the mpi rank
   * @param num_rows the number of rows in the kokkos data structure
   * @param assembly_row_indices Kokkos View for the row coordinates
   * @param assembly_row_start Kokkos View to the start of the data for the rows
   */
  HypreKokkosMatrixAssembler(std::string name, bool ensureReproducible, HypreIntType iLower, HypreIntType iUpper,
			     HypreIntType jLower, HypreIntType jUpper, HypreIntType global_num_rows,
			     HypreIntType global_num_cols, HypreIntType nDataPtsToAssemble, int rank,
			     HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
			     Kokkos::View<HypreIntType *>& assembly_row_start);

  /**
   *  Destructor 
   */
  virtual ~HypreKokkosMatrixAssembler();

  /**
   * assemble : assemble the symbolic and numeric parts of the CSR matrix
   *
   * @param cols Kokkos View of the matrix column indices list
   * @param data Kokkos View of the matrix data list
   */
  virtual void assemble(Kokkos::View<HypreIntType *>& cols, Kokkos::View<double *>& data);

  /**
   * copyCSRMatrixToHost copies the assembled CSR matrix to the host
   */
  virtual void copyCSRMatrixToHost();

  /**
   * get the host row indices ptr in page locked memory
   *
   * @return the pointer to the host row indices
   */
  virtual HypreIntType * getHostRowIndicesPtr() const { return _h_row_indices.data(); }

  /**
   * get the host row counts ptr in page locked memory
   *
   * @return the pointer to the host row counts
   */
  virtual HypreIntType * getHostRowCountsPtr() const { return _h_row_counts.data(); }

  /**
   * get the host column indices ptr in page locked memory
   *
   * @return the pointer to the host column indices 
   */
  virtual HypreIntType * getHostColIndicesPtr() const { return _h_col_indices.data(); }

  /**
   * get the host values ptr in page locked memory
   *
   * @return the pointer to the host values
   */
  virtual double * getHostValuesPtr() const { return _h_values.data(); }

  /**
   * get the host owned row indices ptr in page locked memory
   *
   * @return the pointer to the host owned row indices
   */
  virtual HypreIntType * getHostOwnedRowIndicesPtr() const { return _h_row_indices_owned.data(); }

  /**
   * get the host owned row counts ptr in page locked memory
   *
   * @return the pointer to the host owned row counts
   */
  virtual HypreIntType * getHostOwnedRowCountsPtr() const { return _h_row_counts_owned.data(); }
  
  /**
   * get the host owned column indices ptr in page locked memory
   *
   * @return the pointer to the host owned column indices 
   */
  virtual HypreIntType * getHostOwnedColIndicesPtr() const { return _h_col_indices_owned.data(); }

  /**
   * get the host owned values ptr in page locked memory
   *
   * @return the pointer to the host owned values
   */
  virtual double * getHostOwnedValuesPtr() const { return _h_values_owned.data(); }

  /**
   * get the host shared row indices ptr in page locked memory
   *
   * @return the pointer to the host shared row indices
   */
  virtual HypreIntType * getHostSharedRowIndicesPtr() const { return _h_row_indices_shared.data(); }

  /**
   * get the host shared row counts ptr in page locked memory
   *
   * @return the pointer to the host shared row counts
   */
  virtual HypreIntType * getHostSharedRowCountsPtr() const { return _h_row_counts_shared.data(); }
  
  /**
   * get the host shared column indices ptr in page locked memory
   *
   * @return the pointer to the host shared column indices 
   */
  virtual HypreIntType * getHostSharedColIndicesPtr() const { return _h_col_indices_shared.data(); }

  /**
   * get the host shared values ptr in page locked memory
   *
   * @return the pointer to the host shared values
   */
  virtual double * getHostSharedValuesPtr() const { return _h_values_shared.data(); }

protected:

  /**
   * memoryInGBS computes the amount of device memory used in GBs
   *
   * @return the amount of device memory used in GBs
   */
  double memoryInGBs() const;

  /**
   * copyOwnedCSRMatrixToHost copies the assembled owned CSR matrix to the host (page locked memory)
   */
  void copyOwnedCSRMatrixToHost();

  /**
   * copySharedCSRMatrixToHost copies the assembled shared CSR matrix to the host (page locked memory)
   */
  void copySharedCSRMatrixToHost();

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* timers */
  struct timeval _start, _stop;
  float _assembleTime=0.f;
  float _sortTime=0.f;
  float _xferHostTime=0.f;
  int _nAssemble=0;
#endif

  /* flag for allocating only once */
  bool _owned_shared_views_created=false;

  /* The final csr matrix pointers */
  Kokkos::View<HypreIntType *> _d_row_indices;
  Kokkos::View<HypreIntType *> _d_row_counts;
  Kokkos::View<HypreIntType *> _d_col_indices;
  Kokkos::View<double *> _d_values;

  Kokkos::View<HypreIntType *>::HostMirror _h_row_indices;
  Kokkos::View<HypreIntType *>::HostMirror _h_row_counts;
  Kokkos::View<HypreIntType *>::HostMirror _h_col_indices;
  Kokkos::View<double *>::HostMirror _h_values;

  /* owned CSR matrix */
  Kokkos::View<HypreIntType *> _d_row_indices_owned;
  Kokkos::View<HypreIntType *> _d_row_counts_owned;
  Kokkos::View<HypreIntType *> _d_col_indices_owned;
  Kokkos::View<double *> _d_values_owned;

  Kokkos::View<HypreIntType *>::HostMirror _h_row_indices_owned;
  Kokkos::View<HypreIntType *>::HostMirror _h_row_counts_owned;
  Kokkos::View<HypreIntType *>::HostMirror _h_col_indices_owned;
  Kokkos::View<double *>::HostMirror _h_values_owned;

  /* shared (not owned) CSR matrix */
  Kokkos::View<HypreIntType *> _d_row_indices_shared;
  Kokkos::View<HypreIntType *> _d_row_counts_shared;
  Kokkos::View<HypreIntType *> _d_col_indices_shared;
  Kokkos::View<double *> _d_values_shared;

  Kokkos::View<HypreIntType *>::HostMirror _h_row_indices_shared;
  Kokkos::View<HypreIntType *>::HostMirror _h_row_counts_shared;
  Kokkos::View<HypreIntType *>::HostMirror _h_col_indices_shared;
  Kokkos::View<double *>::HostMirror _h_values_shared;

  /* temporaries/scratch space */
  Kokkos::View<HypreIntType *> _d_dense_keys;
  Kokkos::View<HypreIntType *> _d_mat_elem_bin_locs;
  Kokkos::View<HypreIntType *> _d_mat_elem_bins;
  Kokkos::View<HypreIntType *> _d_transitions;
  Kokkos::View<HypreIntType *> _d_row_counts_scanned;
  Kokkos::View<HypreIntType *>::HostMirror _h_row_counts_scanned;

};

#ifdef KOKKOS_ENABLE_CUDA

class HypreCudaMatrixAssembler : public HypreMatrixAssembler {

public:

  /* While a little dangerous, the CudaRhsAssembler will only access the
     temporary data/memory pool. We allocate here and let the Rhs access it. */
  friend class HypreCudaRhsAssembler;

  /**
   * HypreCudaMatrixAssembler Constructor 
   *
   * @param name of the linear system being assembled
   * @param ensureReproducible will sort the values based on absolute value ascending to ensure that when the input is the
   *   same, though in randomized order potentially, the same RHS is always produced.
   * @param iLower the first row
   * @param iUpper the ending row (inclusive)
   * @param jLower the first column
   * @param jUpper the ending column (inclusive)
   * @param global_num_rows the number of rows in the global matrix
   * @param global_num_cols the number of columns in the global matrix
   * @param nDataPtsToAssemble the number of data points to assemble into a CSR matrix
   * @param rank the mpi rank
   * @param num_rows the number of rows in the kokkos data structure
   * @param assembly_row_indices Kokkos View for the row coordinates
   * @param assembly_row_start Kokkos View to the start of the data for the rows
   */
  HypreCudaMatrixAssembler(std::string name, bool ensureReproducible, HypreIntType iLower, HypreIntType iUpper,
			   HypreIntType jLower, HypreIntType jUpper, HypreIntType global_num_rows,
			   HypreIntType global_num_cols, HypreIntType nDataPtsToAssemble, int rank,
			   HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
			   Kokkos::View<HypreIntType *>& assembly_row_start);

  /**
   *  Destructor 
   */
  virtual ~HypreCudaMatrixAssembler();

  /**
   * copyCSRMatrixToHost copies the assembled CSR matrix to the host (page locked memory)
   */
  virtual void copyCSRMatrixToHost();

  /**
   * assemble : assemble the symbolic and numeric parts of the CSR matrix
   *
   * @param cols host pointer for the column coordinates
   * @param data host pointer for the data values
   */
  virtual void assemble(Kokkos::View<HypreIntType *>& cols, Kokkos::View<double *>& data);

  /**
   * get the host row indices ptr in page locked memory
   *
   * @return the pointer to the host row indices
   */
  virtual HypreIntType * getHostRowIndicesPtr() const { return _h_row_indices; }

  /**
   * get the host row counts ptr in page locked memory
   *
   * @return the pointer to the host row counts
   */
  virtual HypreIntType * getHostRowCountsPtr() const { return _h_row_counts; }

  /**
   * get the host column indices ptr in page locked memory
   *
   * @return the pointer to the host column indices 
   */
  virtual HypreIntType * getHostColIndicesPtr() const { return _h_col_indices; }

  /**
   * get the host values ptr in page locked memory
   *
   * @return the pointer to the host values
   */
  virtual double * getHostValuesPtr() const { return _h_values; }

  /**
   * get the host owned row indices ptr in page locked memory
   *
   * @return the pointer to the host owned row indices
   */
  virtual HypreIntType * getHostOwnedRowIndicesPtr() const { return _h_row_indices_owned; }

  /**
   * get the host owned row counts ptr in page locked memory
   *
   * @return the pointer to the host owned row counts
   */
  virtual HypreIntType * getHostOwnedRowCountsPtr() const { return _h_row_counts_owned; }
  
  /**
   * get the host owned column indices ptr in page locked memory
   *
   * @return the pointer to the host owned column indices 
   */
  virtual HypreIntType * getHostOwnedColIndicesPtr() const { return _h_col_indices_owned; }

  /**
   * get the host owned values ptr in page locked memory
   *
   * @return the pointer to the host owned values
   */
  virtual double * getHostOwnedValuesPtr() const { return _h_values_owned; }

  /**
   * get the host shared row indices ptr in page locked memory
   *
   * @return the pointer to the host shared row indices
   */
  virtual HypreIntType * getHostSharedRowIndicesPtr() const { return _h_row_indices_shared; }

  /**
   * get the host shared row counts ptr in page locked memory
   *
   * @return the pointer to the host shared row counts
   */
  virtual HypreIntType * getHostSharedRowCountsPtr() const { return _h_row_counts_shared; }
  
  /**
   * get the host shared column indices ptr in page locked memory
   *
   * @return the pointer to the host shared column indices 
   */
  virtual HypreIntType * getHostSharedColIndicesPtr() const { return _h_col_indices_shared; }

  /**
   * get the host shared values ptr in page locked memory
   *
   * @return the pointer to the host shared values
   */
  virtual double * getHostSharedValuesPtr() const { return _h_values_shared; }

protected:

  /**
   * memoryInGBS computes the amount of device memory used in GBs
   *
   * @return the amount of device memory used in GBs
   */
  double memoryInGBs() const;

  /**
   * deviceMemoryInGBS gets the amount of free and total memory in GBs
   *
   * @param free the amount of free memory in GBs
   * @param total the amount of total memory in GBs
   */
  void deviceMemoryInGBs(double & free, double & total) const;

#ifdef HYPRE_MATRIX_ASSEMBLER_TIMER
  /* cuda timers */
  cudaEvent_t _start, _stop;
  float _assembleTime=0.f;
  float _sortTime=0.f;
  float _xferHostTime=0.f;
  int _nAssemble=0;
#endif
  
  /* The final csr matrix pointers */
  HypreIntType * _d_row_indices=NULL;
  unsigned long long int * _d_row_counts=NULL;
  HypreIntType * _d_col_indices=NULL;
  double *_d_values=NULL;
  /* host pointers in page locked memory */
  HypreIntType * _h_row_indices=NULL;
  HypreIntType * _h_row_counts=NULL;
  HypreIntType * _h_col_indices=NULL;
  double *_h_values=NULL;

  /* owned CSR matrix */
  HypreIntType * _d_row_indices_owned=NULL;
  unsigned long long int * _d_row_counts_owned=NULL;
  HypreIntType * _d_col_indices_owned=NULL;
  double *_d_values_owned=NULL;
  /* host pointers in page locked memory */
  HypreIntType * _h_row_indices_owned=NULL;
  HypreIntType * _h_row_counts_owned=NULL;
  HypreIntType * _h_col_indices_owned=NULL;
  double *_h_values_owned=NULL;

  /* shared (not owned) CSR matrix */
  HypreIntType * _d_row_indices_shared=NULL;
  unsigned long long int * _d_row_counts_shared=NULL;
  HypreIntType * _d_col_indices_shared=NULL;
  double *_d_values_shared=NULL;
  /* host pointers in page locked memory */
  HypreIntType * _h_row_indices_shared=NULL;
  HypreIntType * _h_row_counts_shared=NULL;
  HypreIntType * _h_col_indices_shared=NULL;
  double *_h_values_shared=NULL;

  /* Cuda pointers and allocations for temporaries */
  HypreIntType * _d_cols=NULL;
  double * _d_data=NULL;
  void * _d_workspace=NULL;

};

#endif

}  // nalu
}  // sierra

#endif /* HYPRE_MATRIX_ASSEMBLER_H */
