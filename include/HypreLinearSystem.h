// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef HYPRELINEARSYSTEM_H
#define HYPRELINEARSYSTEM_H

#include "LinearSystem.h"
#include "XSDKHypreInterface.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "krylov.h"
#include "HYPRE.h"

#include <unordered_set>

#include "HypreCudaLinearSystemAssembler.h"
#include "Kokkos_UnorderedMap.hpp"

#ifndef HYPRE_LINEAR_SYSTEM_TIMER
#define HYPRE_LINEAR_SYSTEM_TIMER
#endif // HYPRE_LINEAR_SYSTEM_TIMER
#undef HYPRE_LINEAR_SYSTEM_TIMER

namespace sierra {
namespace nalu {

using EntityToHypreIntTypeView = Kokkos::View<HypreIntType*, Kokkos::LayoutRight>;
using EntityToHypreIntTypeViewHost = EntityToHypreIntTypeView::HostMirror;

using DoubleView = Kokkos::View<double*>;
using DoubleViewHost = DoubleView::HostMirror;

using DoubleView2D = Kokkos::View<double**>;
using DoubleView2DHost = DoubleView2D::HostMirror;

using HypreIntTypeView = Kokkos::View<HypreIntType*>;
using HypreIntTypeViewHost = HypreIntTypeView::HostMirror;

using IntTypeView2D = Kokkos::View<int**>;
using IntTypeView2DHost = IntTypeView2D::HostMirror;

using HypreIntTypeView2D = Kokkos::View<HypreIntType**>;
using HypreIntTypeView2DHost = HypreIntTypeView2D::HostMirror;

using HypreIntTypeViewScalar = Kokkos::View<HypreIntType>;
using HypreIntTypeViewScalarHost = HypreIntTypeViewScalar::HostMirror;

using HypreIntTypeUnorderedMap = Kokkos::UnorderedMap<HypreIntType, HypreIntType>;
using HypreIntTypeUnorderedMapHost = HypreIntTypeUnorderedMap::HostMirror;

typedef struct hypreIntTypeMapStruct
{
  HypreIntType mat;
  HypreIntType rhs;
  HypreIntType counter;
} HypreIntTypeMapStruct;

using HypreIntTypeMapUnorderedMap = Kokkos::UnorderedMap<HypreIntType, HypreIntTypeMapStruct>;
using HypreIntTypeMapUnorderedMapHost = HypreIntTypeMapUnorderedMap::HostMirror;

/** Nalu interface to populate a Hypre Linear System
 *
 *  This class provides an interface to the HYPRE IJMatrix and IJVector data
 *  structures. It is responsible for creating, resetting, and destroying the
 *  Hypre data structures and provides the HypreLinearSystem::sumInto interface
 *  used by Nalu Kernels and SupplementalAlgorithms to populate entries into the
 *  linear system. The HypreLinearSystem::solve method interfaces with
 *  sierra::nalu::HypreDirectSolver that is responsible for the actual solution
 *  of the system using the required solver and preconditioner combination.
 */
class HypreLinearSystem : public LinearSystem
{
public:
  std::string name_;

  /* data structures for accumulating the matrix elements */
  std::vector<std::vector<HypreIntType> > partitionRowCount_;
  std::vector<HypreIntType> num_nodes_per_partition_;
  float _hypreAssembleTime=0.f;
  int _nHypreAssembles=0;

  EntityToHypreIntTypeView entityToLID_;
  void fill_entity_to_row_mapping();

  HypreIntTypeMapUnorderedMap memory_map_;
  HypreIntTypeView row_indices_;
  HypreIntTypeView mat_row_start_;
  HypreIntTypeView rhs_row_start_;
  HypreIntTypeView periodic_bc_rows_;
  HypreIntTypeUnorderedMap skippedRowsMap_;
  HypreIntType numMatPtsToAssembleTotal_;
  HypreIntType numRhsPtsToAssembleTotal_;
  void fill_device_data_structures(const HypreIntType N);

  // Quiet "partially overridden" compiler warnings.
  using LinearSystem::buildDirichletNodeGraph;
  /**
   * @param[in] realm The realm instance that holds the EquationSystem being solved
   * @param[in] numDof The degrees of freedom for the equation system created (Default: 1)
   * @param[in] eqSys The equation system instance
   * @param[in] linearSolver Handle to the HypreDirectSolver instance
   */
  HypreLinearSystem(
    Realm& realm,
    const unsigned numDof,
    EquationSystem *eqSys,
    LinearSolver *linearSolver);

  virtual ~HypreLinearSystem();

  // Graph/Matrix Construction
  virtual void buildNodeGraph(const stk::mesh::PartVector & parts);// for nodal assembly (e.g., lumped mass and source)
  virtual void buildFaceToNodeGraph(const stk::mesh::PartVector & parts);// face->node assembly
  virtual void buildEdgeToNodeGraph(const stk::mesh::PartVector & parts);// edge->node assembly
  virtual void buildElemToNodeGraph(const stk::mesh::PartVector & parts);// elem->node assembly
  virtual void buildReducedElemToNodeGraph(const stk::mesh::PartVector&);// elem (nearest nodes only)->node assembly
  virtual void buildFaceElemToNodeGraph(const stk::mesh::PartVector & parts);// elem:face->node assembly
  virtual void buildNonConformalNodeGraph(const stk::mesh::PartVector&);// nonConformal->elem_node assembly
  virtual void buildOversetNodeGraph(const stk::mesh::PartVector&);// overset->elem_node assembly
  virtual void finalizeLinearSystem();

  /** Tag rows that must be handled as a Dirichlet BC node
   *
   *  @param[in] partVec List of parts that contain the Dirichlet nodes
   */
  virtual void buildDirichletNodeGraph(const stk::mesh::PartVector&);

  /** Tag rows that must be handled as a Dirichlet  node
   *
   *  @param[in] entities List of nodes where Dirichlet conditions are applied
   *
   *  \sa sierra::nalu::FixPressureAtNodeAlgorithm
   */
  virtual void buildDirichletNodeGraph(const std::vector<stk::mesh::Entity>&);
  virtual void buildDirichletNodeGraph(const stk::mesh::NgpMesh::ConnectedNodes);

  sierra::nalu::CoeffApplier* get_coeff_applier();

  void scanNodes(std::vector<HypreIntType> & nodeCount);

  /***************************************************************************************************/
  /*                     Beginning of HypreLinSysCoeffApplier definition                             */
  /***************************************************************************************************/

  class HypreLinSysCoeffApplier : public CoeffApplier
  {
  public:

    HypreLinSysCoeffApplier(bool ensureReproducible, unsigned numDof,
			    HypreIntType globalNumRows, int rank, 
			    HypreIntType iLower, HypreIntType iUpper,
			    HypreIntType jLower, HypreIntType jUpper,
			    HypreIntTypeMapUnorderedMap memory_map,
			    HypreIntTypeView row_indices,
			    HypreIntTypeView mat_row_start,
			    HypreIntTypeView rhs_row_start,
			    HypreIntType numMatPtsToAssembleTotal,
			    HypreIntType numRhsPtsToAssembleTotal,
			    HypreIntTypeView periodic_bc_rows,
			    EntityToHypreIntTypeView entityToLID,
			    HypreIntTypeUnorderedMap skippedRowsMap);

    KOKKOS_FUNCTION
    virtual ~HypreLinSysCoeffApplier() {
#ifdef KOKKOS_ENABLE_CUDA
      if (MemController_) { delete MemController_; MemController_=NULL; }
      if (MatAssembler_) { delete MatAssembler_; MatAssembler_=NULL; }
      if (RhsAssembler_) { delete RhsAssembler_; RhsAssembler_=NULL; }
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
      if (_nAssembleMat>0) {
	printf("\tMean HYPRE_IJMatrixSetValues Time (%d samples)=%1.5f   Total=%1.5f\n",
	       _nAssembleMat, _assembleMatTime/_nAssembleMat,_assembleMatTime);
	printf("\tMean HYPRE_IJVectorSetValues Time (%d samples)=%1.5f   Total=%1.5f\n",
	       _nAssembleRhs, _assembleRhsTime/_nAssembleRhs,_assembleRhsTime);
      }
#endif
#endif
    }

    KOKKOS_FUNCTION
    virtual void resetRows(unsigned,
                           const stk::mesh::Entity*,
                           const unsigned,
                           const unsigned,
                           const double,
                           const double) { checkSkippedRows_()=0; }

    KOKKOS_FUNCTION
    virtual void sum_into(unsigned numEntities,
			  const stk::mesh::NgpMesh::ConnectedNodes& entities,
			  const SharedMemView<int*,DeviceShmem> & localIds,
			  const SharedMemView<const double*,DeviceShmem> & rhs,
			  const SharedMemView<const double**,DeviceShmem> & lhs,
			  unsigned numDof);

    KOKKOS_FUNCTION
    virtual void sum_into_1DoF(unsigned numEntities,
			       const stk::mesh::NgpMesh::ConnectedNodes& entities,
			       const SharedMemView<int*,DeviceShmem> & localIds,
			       const SharedMemView<const double*,DeviceShmem> & rhs,
			       const SharedMemView<const double**,DeviceShmem> & lhs);


    KOKKOS_FUNCTION
    virtual void operator()(unsigned numEntities,
                            const stk::mesh::NgpMesh::ConnectedNodes& entities,
                            const SharedMemView<int*,DeviceShmem> & localIds,
                            const SharedMemView<int*,DeviceShmem> &,
                            const SharedMemView<const double*,DeviceShmem> & rhs,
                            const SharedMemView<const double**,DeviceShmem> & lhs,
                            const char * trace_tag);

    virtual void free_device_pointer();

    virtual sierra::nalu::CoeffApplier* device_pointer();

    virtual void resetInternalData();

    virtual void applyDirichletBCs(Realm & realm, 
				   stk::mesh::FieldBase * solutionField,
				   stk::mesh::FieldBase * bcValuesField,
				   const stk::mesh::PartVector& parts);
    
    virtual void finishAssembly(void * mat, std::vector<void *> rhs, std::string name);

    //! whether or not to enforce reproducibility
    bool ensureReproducible_=false;
    //! number of degrees of freedom
    unsigned numDof_=0;
    //! Maximum Row ID in the Hypre linear system
    HypreIntType globalNumRows_;
    //! Maximum Row ID in the Hypre linear system
    int rank_;
    //! The lowest row owned by this MPI rank
    HypreIntType iLower_=0;
    //! The highest row owned by this MPI rank
    HypreIntType iUpper_=0;
    //! The lowest column owned by this MPI rank; currently jLower_ == iLower_
    HypreIntType jLower_=0;
    //! The highest column owned by this MPI rank; currently jUpper_ == iUpper_
    HypreIntType jUpper_=0;

    //! the starting position(s) of the matrix/rhs/counter in one map
    HypreIntTypeMapUnorderedMap memory_map_;
    //! the row indices
    HypreIntTypeView row_indices_;
    //! the starting position(s) of the matrix list partitions
    HypreIntTypeView mat_row_start_;
    //! the starting position(s) of the rhs list partitions
    HypreIntTypeView rhs_row_start_;
    //! an upper bound on the total number of matrix list points
    HypreIntType numMatPtsToAssembleTotal_;
    //! an upper bound on the total number of rhs list points
    HypreIntType numRhsPtsToAssembleTotal_;
    //! rows for the periodic boundary conditions
    HypreIntTypeView periodic_bc_rows_;

    //! A way to map the entity local offset to the hypre id
    EntityToHypreIntTypeView entityToLID_;
    //! unordered map for skipped rows
    HypreIntTypeUnorderedMap skippedRowsMap_;

    //! this is the pointer to the device function ... that assembles the lists
    HypreLinSysCoeffApplier* devicePointer_;

    /* flag to reinitialize or not */
    bool reinitialize_=true;

    //! 2D data structure to atomically update for augmenting the list */
    HypreIntTypeView mat_counter_;
    HypreIntTypeView rhs_counter_;
    
    //! list for the column indices ... later to be assembled to the CSR matrix in Hypre
    HypreIntTypeView cols_;
    //! list for the values ... later to be assembled to the CSR matrix in Hypre
    DoubleView vals_;
    //! list for the rhs values ... later to be assembled to the rhs vector in Hypre
    DoubleView2D rhs_vals_;

    //! Total number of rows owned by this particular MPI rank
    HypreIntType numRows_;

    //! Flag indicating that sumInto should check to see if rows must be skipped
    HypreIntTypeViewScalar checkSkippedRows_;

#ifdef KOKKOS_ENABLE_CUDA
    //! The Memory Controller ... used for temporaries that can be shared between Matrix and Rhs assemblies
    MemoryController<HypreIntType> * MemController_=nullptr;
    //! The Matrix assembler
    MatrixAssembler<HypreIntType> * MatAssembler_=nullptr;
    //! The Rhs assembler
    RhsAssembler<HypreIntType> * RhsAssembler_=nullptr;

    float _assembleMatTime=0.f;
    float _assembleRhsTime=0.f;
    int _nAssembleMat=0;
    int _nAssembleRhs=0;
#endif
  };

  /***************************************************************************************************/
  /*                        End of of HypreLinSysCoeffApplier definition                             */
  /***************************************************************************************************/

  /** Reset the matrix and rhs data structures for the next iteration/timestep
   *
   */
  virtual void zeroSystem();

  /** Update coefficients of a particular row(s) in the linear system
   *
   *  The core method of this class, it updates the matrix and RHS based on the
   *  inputs from the various algorithms. Note that, unlike TpetraLinearSystem,
   *  this method skips over the fringe points of Overset mesh and the Dirichlet
   *  nodes rather than resetting them afterward.
   *
   *  This overloaded method deals with Kernels designed with Kokkos::View arrays.
   *
   *  @param[in] numEntities The total number of nodes where data is to be updated
   *  @param[in] entities A list of STK node entities
   *
   *  @param[in] rhs Array containing RHS entries to be summed into
   *      [numEntities * numDof]
   *
   *  @param[in] lhs Array containing LHS entries to be summed into.
   *      [numEntities * numDof, numEntities * numDof]
   *
   *  @param[in] localIds Work array for storing local row IDs
   *  @param[in] sortPermutation Work array for sorting row IDs
   *  @param[in] trace_tag Debugging message
   */
  virtual void sumInto(
    unsigned numEntities,
    const stk::mesh::NgpMesh::ConnectedNodes& entities,
    const SharedMemView<const double*, DeviceShmem> & rhs,
    const SharedMemView<const double**, DeviceShmem> & lhs,
    const SharedMemView<int*, DeviceShmem> & localIds,
    const SharedMemView<int*, DeviceShmem> & sortPermutation,
    const char * trace_tag
  );

  /** Update coefficients of a particular row(s) in the linear system
   *
   *  The core method of this class, it updates the matrix and RHS based on the
   *  inputs from the various algorithms. Note that, unlike TpetraLinearSystem,
   *  this method skips over the fringe points of Overset mesh and the Dirichlet
   *  nodes rather than resetting them afterward.
   *
   *  This overloaded method deals with classic SupplementalAlgorithms
   *
   *  @param[in] sym_meshobj A list of STK node entities
   *  @param[in] scratchIds Work array for row IDs
   *  @param[in] scratchVals Work array for row entries
   *
   *  @param[in] rhs Array containing RHS entries to be summed into
   *      [numEntities * numDof]
   *
   *  @param[in] lhs Array containing LHS entries to be summed into.
   *      [numEntities * numDof * numEntities * numDof]
   *
   *  @param[in] trace_tag Debugging message
   */
  virtual void sumInto(
    const std::vector<stk::mesh::Entity> & sym_meshobj,
    std::vector<int> &scratchIds,
    std::vector<double> &scratchVals,
    const std::vector<double> & rhs,
    const std::vector<double> & lhs,
    const char *trace_tag
  );

  /** Populate the LHS and RHS for the Dirichlet rows in linear system
   */
  virtual void applyDirichletBCs(
    stk::mesh::FieldBase * solutionField,
    stk::mesh::FieldBase * bcValuesField,
    const stk::mesh::PartVector & parts,
    const unsigned beginPos,
    const unsigned endPos);

  /** Prepare assembly for Dirichlet-type rows
   *
   *  Dirichlet rows are skipped over by the sumInto method when the interior
   *  parts are processed. This method toggles the flag alerting the sumInto
   *  method that the Dirichlet rows will be processed next and sumInto can
   *  proceed.
   */
  virtual void resetRows(
    const std::vector<stk::mesh::Entity>&,
    const unsigned,
    const unsigned,
    const double,
    const double)
  {
    checkSkippedRows_ = false;
  }

  virtual void resetRows(
    unsigned /*numNodes*/,
    const stk::mesh::Entity* /*nodeList*/,
    const unsigned,
    const unsigned,
    const double,
    const double)
  {
    checkSkippedRows_ = false;
  }

  /** Solve the system Ax = b
   *
   *  The solution vector is returned in linearSolutionField
   *
   *  @param[out] linearSolutionField STK field where the solution is populated
   */
  virtual int solve(stk::mesh::FieldBase * linearSolutionField);

  /** Finalize construction of the linear system matrix and rhs vector
   *
   *  This method calls the appropriate Hypre functions to assemble the matrix
   *  and rhs in a parallel run, as well as registers the matrix and rhs with
   *  the solver preconditioner.
   */
  virtual void loadComplete();

  virtual void writeToFile(const char * /* filename */, bool /* useOwned */ =true) {}
  virtual void writeSolutionToFile(const char * /* filename */, bool /* useOwned */ =true) {}

protected:

  /** Prepare the instance for system construction
   *
   *  During initialization, this creates the hypre data structures via API
   *  calls. It also synchronizes hypreGlobalId across shared and ghosted data
   *  so that hypre row ID lookups succeed during initialization and assembly.
   */
  virtual void beginLinearSystemConstruction();

  virtual void finalizeSolver();

  virtual void loadCompleteSolver();

  /** Return the Hypre ID corresponding to the given STK node entity
   *
   *  @param[in] entity The STK node entity object
   *
   *  @return The HYPRE row ID
   */
  HypreIntType get_entity_hypre_id(const stk::mesh::Entity&);

  //! Helper method to transfer the solution from a HYPRE_IJVector instance to
  //! the STK field data instance.
  double copy_hypre_to_stk(stk::mesh::FieldBase*);

  /** Flags indicating whether a particular row in the HYPRE matrix has been
   * filled or not.
   */
  enum RowFillStatus
  {
    RS_UNFILLED = 0, //!< Default status
    RS_FILLED        //!< sumInto filps to filled status once a row has been acted on
  };

  /** Flag indicating the type of row.
   *
   *  This flag is used to determine if the normal sumInto approach is used to
   *  populate the row, or a special method is used to handle that row. sumInto
   *  method will skip over the rows not marked RT_NORMAL and must be dealt with
   *  separately by other algorithms.
   */
  enum RowStatus
  {
    RT_NORMAL = 0, //!< A normal row that is summed into using sumInto
    RT_DIRICHLET,  //!< Rows with Dirichlet BC; no off-diagonal entries
    RT_OVERSET     //!< Overset fringe points; interpolation weights from other mesh
  };

  /** Dummy method to satisfy inheritance
   */
  void checkError(
    const int,
    const char*) {}

  //! The HYPRE matrix data structure
  mutable HYPRE_IJMatrix mat_;

  //! Track rows that have been updated during the assembly process
  std::vector<RowFillStatus> rowFilled_;

  //! Track the status of rows
  std::vector<RowStatus> rowStatus_;

  //! Track which rows are skipped
  std::unordered_set<HypreIntType> skippedRows_;

  //! Buffer for handling Global Row IDs for use in sumInto methods
  std::vector<HypreIntType> idBuffer_;

  //! The lowest row owned by this MPI rank
  HypreIntType iLower_;
  //! The highest row owned by this MPI rank
  HypreIntType iUpper_;
  //! The lowest column owned by this MPI rank; currently jLower_ == iLower_
  HypreIntType jLower_;
  //! The highest column owned by this MPI rank; currently jUpper_ == iUpper_
  HypreIntType jUpper_;
  //! Total number of rows owned by this particular MPI rank
  HypreIntType numRows_;
  //! Total number of rows owned by this particular MPI rank
  HypreIntType globalNumRows_;
  //! Store the rank as class data so it's easy to reference
  int rank_;
  //! Maximum Row ID in the Hypre linear system
  HypreIntType maxRowID_;

  //! Flag indicating whether IJMatrixAssemble has been called on the system
  bool matrixAssembled_{false};

  //! Flag indicating whether the linear system has been initialized
  bool systemInitialized_{false};

  //! Flag indicating that sumInto should check to see if rows must be skipped
  bool checkSkippedRows_{false};

  //! Flag indicating that dirichlet and/or overset rows are present for this system
  bool hasSkippedRows_{false};

private:
  //! HYPRE right hand side data structure
  mutable HYPRE_IJVector rhs_;

  //! HYPRE solution vector
  mutable HYPRE_IJVector sln_;

};

}  // nalu
}  // sierra


#endif /* HYPRELINEARSYSTEM_H */
