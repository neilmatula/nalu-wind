#ifndef HYPRE_RHS_ASSEMBLER_H
#define HYPRE_RHS_ASSEMBLER_H

#ifndef HYPRE_RHS_ASSEMBLER_DEBUG
#define HYPRE_RHS_ASSEMBLER_DEBUG
#endif // HYPRE_RHS_ASSEMBLER_DEBUG
#undef HYPRE_RHS_ASSEMBLER_DEBUG

#ifndef HYPRE_RHS_ASSEMBLER_TIMER
#define HYPRE_RHS_ASSEMBLER_TIMER
#endif // HYPRE_RHS_ASSEMBLER_TIMER
//#undef HYPRE_RHS_ASSEMBLER_TIMER

#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sys/time.h>

#include "HypreCudaAssembler.h"
#include <Kokkos_Sort.hpp>

namespace sierra {
namespace nalu {

class HypreRhsAssembler {

public:

  /**
   * HypreRhsAssembler Constructor 
   *
   * @param choice specific type of assembler to be used: choice==0, Kokkos, choice==1, Cuda
   * @param name of the linear system being assembled
   * @param ensureReproducible will sort the values based on absolute value ascending to ensure that when the input is the
   *   same, though in randomized order potentially, the same RHS is always produced.
   * @param iLower the first row
   * @param iUpper the ending row (inclusive)
   * @param global_num_rows the number of rows in the global matrix
   * @param nDataPtsToAssemble the number of data points to assemble into a rhs vector
   * @param rank the mpi rank
   * @param num_rows the number of rows in the kokkos data structure
   * @param assembly_row_indices Kokkos View for the row coordinates
   * @param assembly_row_start Kokkos View to the start of data for the rowss
   *
   * @return a pointer to HypreRhsAssembler
   */
  static HypreRhsAssembler * make_HypreRhsAssembler(int choice, std::string name, bool ensureReproducible,
						    HypreIntType iLower, HypreIntType iUpper,
						    HypreIntType global_num_rows, HypreIntType nDataPtsToAssemble, int rank,
						    HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
						    Kokkos::View<HypreIntType *>& assembly_row_start);

  /**
   *  Destructor 
   */
  virtual ~HypreRhsAssembler() {}

  /**
   * assemble : assemble the symbolic and numeric parts of the CSR matrix
   *
   * @param data 2D Kokkos of the rhs data lists
   * @param index rhs component being assembled
   */
  virtual void assemble(Kokkos::View<double **>& data, const int index) = 0;

  /**
   * copyRhsVectorToHost copies the assembled RhsVector to the host
   */
  virtual void copyRhsVectorToHost() = 0;

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
   * get the host rhs ptr in page locked memory
   *
   * @return the pointer to the host rhs
   */
  virtual double * getHostRhsPtr() const = 0;

  /**
   * get the host rhs indices ptr in page locked memory
   *
   * @return the pointer to the host rhs indices
   */
  virtual HypreIntType * getHostRhsIndicesPtr() const = 0;

  /**
   * get the host owned rhs ptr in page locked memory
   *
   * @return the pointer to the host owned rhs
   */
  virtual double * getHostOwnedRhsPtr() const = 0;

  /**
   * get the host owned rhs indices ptr in page locked memory
   *
   * @return the pointer to the host owned rhs indices
   */
  virtual HypreIntType * getHostOwnedRhsIndicesPtr() const = 0;

  /**
   * get the host shared rhs ptr in page locked memory
   *
   * @return the pointer to the host shared rhs
   */
  virtual double * getHostSharedRhsPtr() const = 0;

  /**
   * get the host shared rhs indices ptr in page locked memory
   *
   * @return the pointer to the host shared rhs indices
   */
  virtual HypreIntType * getHostSharedRhsIndicesPtr() const = 0;

protected:

  /* Protected constructor. Should only be called when instantiating derived class types */
  HypreRhsAssembler(std::string name, bool ensureReproducible, HypreIntType iLower, HypreIntType iUpper,
		    HypreIntType global_num_rows, HypreIntType nDataPtsToAssemble, int rank,
		    HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
		    Kokkos::View<HypreIntType *>& assembly_row_start) : 
    _name(name), _ensure_reproducible(ensureReproducible), _iLower(iLower), _iUpper(iUpper),
    _global_num_rows(global_num_rows), _nDataPtsToAssemble(nDataPtsToAssemble), _rank(rank),
    _num_rows(num_rows), _d_assembly_row_indices(assembly_row_indices), _d_assembly_row_start(assembly_row_start) {}

  /* input meta data */
  std::string _name="";
  /* whether or not to sort the values data to ensure reproducibility */
  bool _ensure_reproducible=false;
  /* row extents for this rank */
  HypreIntType _iLower=0;
  HypreIntType _iUpper=0;
  HypreIntType _global_num_rows=0;
  HypreIntType _num_rows_this_rank=0;
  HypreIntType _nDataPtsToAssemble=0;
  int _rank=0;

  /* Input number of rows. Owned and shared are computed quantities */
  HypreIntType _num_rows=0;
  HypreIntType _num_rows_owned=0;
  HypreIntType _num_rows_shared=0;

  /* amount of memory being used */
  HypreIntType _memoryUsed=0;

  /* whether or not this class contains shared rows for other ranks */
  bool _has_shared=false;

  /* Kokkos View for the row indices and row starts. Built in the assemble stage. */
  Kokkos::View<HypreIntType *> _d_assembly_row_indices;
  Kokkos::View<HypreIntType *> _d_assembly_row_start;

};


class HypreKokkosRhsAssembler : public HypreRhsAssembler {

public:

  /**
   * HypreKokkosRhsAssembler Constructor 
   *
   * @param name of the linear system being assembled
   * @param ensureReproducible will sort the values based on absolute value ascending to ensure that when the input is the
   *   same, though in randomized order potentially, the same RHS is always produced.
   * @param iLower the first row
   * @param iUpper the ending row (inclusive)
   * @param global_num_rows the number of rows in the global matrix
   * @param nDataPtsToAssemble the number of data points to assemble into a rhs vector
   * @param rank the mpi rank
   * @param num_rows the number of rows in the kokkos data structure
   * @param assembly_row_indices Kokkos View for the row coordinates
   * @param assembly_row_start Kokkos View to the start of data for the rowss
   */
  HypreKokkosRhsAssembler(std::string name, bool ensureReproducible, HypreIntType iLower, HypreIntType iUpper,
			  HypreIntType global_num_rows, HypreIntType nDataPtsToAssemble, int rank,
			  HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
			  Kokkos::View<HypreIntType *>& assembly_row_start);

  /**
   *  Destructor 
   */
  virtual ~HypreKokkosRhsAssembler();

  /**
   * assemble : assemble the symbolic and numeric parts of the CSR matrix
   *
   * @param data 2D Kokkos of the rhs data lists
   * @param index rhs component being assembled
   */
  virtual void assemble(Kokkos::View<double **>& data, const int index);

  /**
   * copyRhsVectorToHost copies the assembled RhsVector to the host (page locked memory)
   */
  virtual void copyRhsVectorToHost();

  /**
   * get the host rhs ptr in page locked memory
   *
   * @return the pointer to the host rhs
   */
  virtual double * getHostRhsPtr() const { return _h_rhs.data(); }

  /**
   * get the host rhs indices ptr in page locked memory
   *
   * @return the pointer to the host rhs indices
   */
  virtual HypreIntType * getHostRhsIndicesPtr() const { return _h_rhs_indices.data(); }

  /**
   * get the host owned rhs ptr in page locked memory
   *
   * @return the pointer to the host owned rhs
   */
  virtual double * getHostOwnedRhsPtr() const { return _h_rhs_owned.data(); }

  /**
   * get the host owned rhs indices ptr in page locked memory
   *
   * @return the pointer to the host owned rhs indices
   */
  virtual HypreIntType * getHostOwnedRhsIndicesPtr() const { return _h_rhs_indices_owned.data(); }

  /**
   * get the host shared rhs ptr in page locked memory
   *
   * @return the pointer to the host shared rhs
   */
  virtual double * getHostSharedRhsPtr() const { return _h_rhs_shared.data(); }

  /**
   * get the host shared rhs indices ptr in page locked memory
   *
   * @return the pointer to the host shared rhs indices
   */
  virtual HypreIntType * getHostSharedRhsIndicesPtr() const { return _h_rhs_indices_shared.data(); }

protected:

  /**
   * memoryInGBS computes the amount of device memory used in GBs
   *
   * @return the amount of device memory used in GBs
   */
  double memoryInGBs() const;

  /**
   * copyOwnedRhsVectorToHost copies the assembled owned RhsVector to the host (page locked memory)
   */
  void copyOwnedRhsVectorToHost();

  /**
   * copySharedRhsVectorToHost copies the assembled shared RhsVector to the host (page locked memory)
   */
  void copySharedRhsVectorToHost();

#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  /* timers */
  struct timeval _start, _stop;
  struct timeval _start_refined, _stop_refined;
  float _assembleTime=0.f;
  float _xferHostTime=0.f;
  float _fillRhsTime=0.f;
  float _findOwnedSharedBndryTime=0.f;
  float _fillOwnedSharedTime=0.f;
  int _nAssemble=0;
#endif

  /* flag for allocating only once */
  bool _owned_shared_views_created=false;

  /* The final rhs vector */
  Kokkos::View<double *> _d_rhs;
  Kokkos::View<HypreIntType *> _d_rhs_indices;
  Kokkos::View<double *>::HostMirror _h_rhs;
  Kokkos::View<HypreIntType *>::HostMirror _h_rhs_indices;

  /* The owned rhs vector */
  Kokkos::View<double *> _d_rhs_owned;
  Kokkos::View<HypreIntType *> _d_rhs_indices_owned;
  Kokkos::View<double *>::HostMirror _h_rhs_owned;
  Kokkos::View<HypreIntType *>::HostMirror _h_rhs_indices_owned;

  /* The shared rhs vector */
  Kokkos::View<double *> _d_rhs_shared;
  Kokkos::View<HypreIntType *> _d_rhs_indices_shared;
  Kokkos::View<double *>::HostMirror _h_rhs_shared;
  Kokkos::View<HypreIntType *>::HostMirror _h_rhs_indices_shared;

  /* temporaries/scratch space */
  Kokkos::View<HypreIntType *> _d_int_workspace;
  Kokkos::View<double *> _d_double_workspace;

};

#ifdef KOKKOS_ENABLE_CUDA

class HypreCudaRhsAssembler : public HypreRhsAssembler {

public:

  /**
   * HypreCudaRhsAssembler Constructor 
   *
   * @param name of the linear system being assembled
   * @param ensureReproducible will sort the values based on absolute value ascending to ensure that when the input is the
   *   same, though in randomized order potentially, the same RHS is always produced.
   * @param iLower the first row
   * @param iUpper the ending row (inclusive)
   * @param global_num_rows the number of rows in the global matrix
   * @param nDataPtsToAssemble the number of data points to assemble into a rhs vector
   * @param rank the mpi rank
   * @param num_rows the number of rows in the kokkos data structure
   * @param assembly_row_indices Kokkos View for the row coordinates
   * @param assembly_row_start Kokkos View to the start of the row insides (rows, data) structures
   */
  HypreCudaRhsAssembler(std::string name, bool ensureReproducible, HypreIntType iLower, HypreIntType iUpper,
			HypreIntType global_num_rows, HypreIntType nDataPtsToAssemble, int rank,
			HypreIntType num_rows, Kokkos::View<HypreIntType *>& assembly_row_indices,
			Kokkos::View<HypreIntType *>& assembly_row_start);
  /**
   *  Destructor 
   */
  virtual ~HypreCudaRhsAssembler();

  /**
   * assemble : assemble the symbolic and numeric parts of the CSR matrix
   *
   * @param data 2D Kokkos of the rhs data lists
   * @param index rhs component being assembled
   */
  virtual void assemble(Kokkos::View<double **>& data, const int index);

  /**
   * copyRhsVectorToHost copies the assembled RhsVector to the host (page locked memory)
   */
  virtual void copyRhsVectorToHost();

  /**
   * get the host rhs ptr in page locked memory
   *
   * @return the pointer to the host rhs
   */
  virtual double * getHostRhsPtr() const { return _h_rhs; }

  /**
   * get the host rhs indices ptr in page locked memory
   *
   * @return the pointer to the host rhs indices
   */
  virtual HypreIntType * getHostRhsIndicesPtr() const { return _h_rhs_indices; }

  /**
   * get the host owned rhs ptr in page locked memory
   *
   * @return the pointer to the host owned rhs
   */
  virtual double * getHostOwnedRhsPtr() const { return _h_rhs_owned; }

  /**
   * get the host owned rhs indices ptr in page locked memory
   *
   * @return the pointer to the host owned rhs indices
   */
  virtual HypreIntType * getHostOwnedRhsIndicesPtr() const { return _h_rhs_indices_owned; }

  /**
   * get the host shared rhs ptr in page locked memory
   *
   * @return the pointer to the host shared rhs
   */
  virtual double * getHostSharedRhsPtr() const { return _h_rhs_shared; }

  /**
   * get the host shared rhs indices ptr in page locked memory
   *
   * @return the pointer to the host shared rhs indices
   */
  virtual HypreIntType * getHostSharedRhsIndicesPtr() const { return _h_rhs_indices_shared; }

protected:

  /**
   * deviceMemoryInGBS gets the amount of free and total memory in GBs
   *
   * @param free the amount of free memory in GBs
   * @param total the amount of total memory in GBs
   */
  void deviceMemoryInGBs(double & free, double & total) const;

  /**
   * memoryInGBS computes the amount of device memory used in GBs
   *
   * @return the amount of device memory used in GBs
   */
  double memoryInGBs() const;

#ifdef HYPRE_RHS_ASSEMBLER_TIMER
  /* cuda timers */
  cudaEvent_t _start, _stop;
  cudaEvent_t _start_refined, _stop_refined;
  float _assembleTime=0.f;
  float _xferHostTime=0.f;
  float _fillRhsTime=0.f;
  float _findOwnedSharedBndryTime=0.f;
  float _fillOwnedSharedTime=0.f;
  int _nAssemble=0;
#endif

  /* The final rhs vector */
  double *_d_rhs=NULL;
  HypreIntType *_d_rhs_indices=NULL;
  double *_h_rhs=NULL;
  HypreIntType *_h_rhs_indices=NULL;

  /* The owned rhs vector */
  double *_d_rhs_owned=NULL;
  HypreIntType *_d_rhs_indices_owned=NULL;
  double *_h_rhs_owned=NULL;
  HypreIntType *_h_rhs_indices_owned=NULL;

  /* The shared rhs vector */
  double *_d_rhs_shared=NULL;
  HypreIntType *_d_rhs_indices_shared=NULL;
  double *_h_rhs_shared=NULL;
  HypreIntType *_h_rhs_indices_shared=NULL;

  /* temporaries/scratch space */
  double * _d_data=NULL;
  void * _d_workspace=NULL;

};

#endif // KOKKOS_ENABLE_CUDA

}  // nalu
}  // sierra

#endif /* HYPRE_RHS_ASSEMBLER_H */
