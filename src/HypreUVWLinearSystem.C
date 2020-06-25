// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "HypreUVWLinearSystem.h"
#include "HypreUVWSolver.h"
#include "NaluEnv.h"
#include "Realm.h"
#include "EquationSystem.h"

#include <utils/CreateDeviceExpression.h>

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE.h"
#include "HYPRE_config.h"

#include <limits>
#include <vector>
#include <string>
#include <cmath>

namespace sierra {
namespace nalu {

HypreUVWLinearSystem::HypreUVWLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver
) : HypreLinearSystem(realm, 1, eqSys, linearSolver),
    rhs_(numDof, nullptr),
    sln_(numDof, nullptr),
    nDim_(numDof)
{}

HypreUVWLinearSystem::~HypreUVWLinearSystem()
{
  if (systemInitialized_) {
    HYPRE_IJMatrixDestroy(mat_);

    for (unsigned i=0; i<nDim_; ++i) {
      HYPRE_IJVectorDestroy(rhs_[i]);
      HYPRE_IJVectorDestroy(sln_[i]);
    }
  }
  systemInitialized_ = false;
}


void
HypreUVWLinearSystem::finalizeLinearSystem()
{
  ThrowRequire(inConstruction_);
  inConstruction_ = false;

  // Prepare for matrix assembly and set all entry flags to "unfilled"
  for (HypreIntType i=0; i < numRows_; i++)
    rowFilled_[i] = RS_UNFILLED;

  finalizeSolver();

  /* create these mappings */
  fill_entity_to_row_mapping();

  /* fill the various device data structures need in device coeff applier */
  fill_device_data_structures(1);

  // Set flag to indicate whether rows must be skipped during normal sumInto
  // process. For this to be activated, the linear system must have Dirichlet or
  // overset rows and they must be present on this processor
  if (hasSkippedRows_ && !skippedRows_.empty())
    checkSkippedRows_ = true;

  // At this stage the LHS and RHS data structures are ready for
  // sumInto/assembly.
  systemInitialized_ = true;
}

void
HypreUVWLinearSystem::finalizeSolver()
{

  MPI_Comm comm = realm_.bulk_data().parallel();
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  for (unsigned i=0; i<nDim_; ++i) {
    HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_[i]);
    HYPRE_IJVectorSetObjectType(rhs_[i], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void**)&(solver->parRhsU_[i]));

    HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_[i]);
    HYPRE_IJVectorSetObjectType(sln_[i], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void**)&(solver->parSlnU_[i]));
  }
}

void
HypreUVWLinearSystem::loadComplete()
{
  // All algorithms have called sumInto and populated LHS/RHS. Now we are ready
  // to finalize the matrix at the HYPRE end. However, before we do that we need
  // to process unfilled rows and process them appropriately. Any row acted on
  // by sumInto method will have toggled the rowFilled_ array to RS_FILLED
  // status. Before finalizing assembly, we process rows that still have an
  // RS_UNFILLED status and set their diagonal entries to 1.0 (dummy row)
  //
  // TODO: Alternate design to eliminate dummy rows. This will require
  // load-balancing on HYPRE end.

#ifdef KOKKOS_ENABLE_CUDA

  std::vector<void *> rhs(nDim_);
  for (unsigned i=0; i<nDim_; ++i) rhs[i] = (void*)(&rhs_[i]);

  hostCoeffApplier->finishAssembly((void*)&mat_, rhs, name_);

#else

  HypreIntType hnrows = 1;
  HypreIntType hncols = 1;
  double getval;
  double setval = 1.0;
  for (HypreIntType i=0; i < numRows_; i++) {
    if (rowFilled_[i] == RS_FILLED) continue;
    HypreIntType lid = iLower_ + i;
    HYPRE_IJMatrixGetValues(mat_, hnrows, &hncols, &lid, &lid, &getval);
    if (std::fabs(getval) < 1.0e-12) {
      HYPRE_IJMatrixSetValues(mat_, hnrows, &hncols, &lid, &lid, &setval);
    }
  }

#endif

  loadCompleteSolver();
}


void
HypreUVWLinearSystem::loadCompleteSolver()
{
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  for (unsigned i=0; i<nDim_; ++i) {
    HYPRE_IJVectorAssemble(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void**)&(solver->parRhsU_[i]));

    HYPRE_IJVectorAssemble(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void**)&(solver->parSlnU_[i]));
  }

  solver->comm_ = realm_.bulk_data().parallel();

  matrixAssembled_ = true;
}

void
HypreUVWLinearSystem::zeroSystem()
{
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  if (matrixAssembled_) {
    HYPRE_IJMatrixInitialize(mat_);
    for (unsigned i=0; i<nDim_; ++i) {
      HYPRE_IJVectorInitialize(rhs_[i]);
      HYPRE_IJVectorInitialize(sln_[i]);
    }

    matrixAssembled_ = false;
  }

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  for (unsigned i=0; i<nDim_; ++i) {
    HYPRE_ParVectorSetConstantValues((solver->parRhsU_[i]), 0.0);
    HYPRE_ParVectorSetConstantValues((solver->parSlnU_[i]), 0.0);
  }

  // Prepare for matrix assembly and set all entry flags to "unfilled"
  for (HypreIntType i=0; i < numRows_; i++)
    rowFilled_[i] = RS_UNFILLED;

  // Set flag to indicate whether rows must be skipped during normal sumInto
  // process. For this to be activated, the linear system must have Dirichlet or
  // overset rows and they must be present on this processor
  if (hasSkippedRows_ && !skippedRows_.empty())
    checkSkippedRows_ = true;
}

void
HypreUVWLinearSystem::sumInto(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<int*, DeviceShmem>&,
  const char*  /* trace_tag */)
{
#ifndef KOKKOS_ENABLE_CUDA
  HypreIntType numRows = numEntities;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  if (bufSize < numRows) {
    idBuffer_.resize(numRows);
    scratchRowVals_.resize(numRows);
  }

  for (size_t in=0; in < numEntities; in++) {
    idBuffer_[in] = get_entity_hypre_id(entities[in]);
  }

  for (size_t in=0; in < numEntities; in++) {
    int ix = in * nDim_;
    HypreIntType hid = idBuffer_[in];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) {
	continue;
      }
    }

    int offset = 0;
    for (int c=0; c < numRows; c++) {
      scratchRowVals_[c] = lhs(ix, offset);
      offset += nDim_;
    }
    HYPRE_IJMatrixAddToValues(
      mat_, 1, &numRows, &hid, &idBuffer_[0], &scratchRowVals_[0]);

    for (unsigned d=0; d<nDim_; ++d) {
      int ir = ix + d;
      HYPRE_IJVectorAddToValues(rhs_[d], 1, &hid, &rhs[ir]);
    }

    if ((hid >= iLower_) && (hid <= iUpper_))
      rowFilled_[hid - iLower_] = RS_FILLED;
  }
#endif
}

void
HypreUVWLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>&  /* scratchIds */,
  std::vector<double>& scratchVals,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char*  /* trace_tag */)
{
  const size_t n_obj = entities.size();
  HypreIntType numRows = n_obj;
  const HypreIntType bufSize = idBuffer_.size();

#ifndef NDEBUG
  size_t vecSize = numRows * nDim_;
  ThrowAssert(vecSize == rhs.size());
  ThrowAssert(vecSize*vecSize == lhs.size());
#endif
  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    idBuffer_[in] = get_entity_hypre_id(entities[in]);
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * nDim_;
    HypreIntType hid = get_entity_hypre_id(entities[in]);

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) {
	continue;
      }
    }

    int offset = 0;
    int ic = ix * numRows * nDim_;
    for (int c=0; c < numRows; c++) {
      scratchVals[c] = lhs[ic + offset];
      offset += nDim_;
    }
    HYPRE_IJMatrixAddToValues(
      mat_, 1, &numRows, &hid, &idBuffer_[0], &scratchVals[0]);

    for (unsigned d = 0; d<nDim_; ++d) {
      int ir = ix + d;
      HYPRE_IJVectorAddToValues(rhs_[d], 1, &hid, &rhs[ir]);
    }

    if ((hid >= iLower_) && (hid <= iUpper_))
      rowFilled_[hid - iLower_] = RS_FILLED;
  }
}

void
HypreUVWLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase* solutionField,
  stk::mesh::FieldBase* bcValuesField,
  const stk::mesh::PartVector& parts,
  const unsigned,
  const unsigned)
{
  double adbc_time = -NaluEnv::self().nalu_time();

#ifdef KOKKOS_ENABLE_CUDA

  hostCoeffApplier->applyDirichletBCs(realm_, solutionField, bcValuesField, parts);

#else 

  auto& meta = realm_.meta_data();

  const stk::mesh::Selector sel = (
    meta.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm_.get_inactive_selector()));

  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

  HypreIntType ncols = 1;
  double diag_value = 1.0;
  for (auto b: bkts) {
    const double* solution = (double*)stk::mesh::field_data(
      *solutionField, *b);
    const double* bcValues = (double*)stk::mesh::field_data(
      *bcValuesField, *b);

    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);

      HYPRE_IJMatrixSetValues(mat_, 1, &ncols, &hid, &hid, &diag_value);

      for (unsigned d=0; d<nDim_; ++d) {
        double bcval = bcValues[in*nDim_ + d] - solution[in*nDim_ + d];
        HYPRE_IJVectorSetValues(rhs_[d], 1, &hid, &bcval);
      }
      rowFilled_[hid - iLower_] = RS_FILLED;
    }
  }
#endif

  adbc_time += NaluEnv::self().nalu_time();
}

int
HypreUVWLinearSystem::solve(stk::mesh::FieldBase* slnField)
{
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string matFile = eqSysName_ + ".IJM." + writeCounter + ".mat";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());

    for (unsigned d=0; d<nDim_; ++d) {
      const std::string rhsFile =
        eqSysName_ + std::to_string(d) + ".IJV." + writeCounter + ".rhs";
      HYPRE_IJVectorPrint(rhs_[d], rhsFile.c_str());
    }
  }

  int status = 0;
  std::vector<int> iters(nDim_, 0);
  std::vector<double> finalNorm(nDim_, 1.0);
  std::vector<double> rhsNorm(nDim_, std::numeric_limits<double>::max());

  for (unsigned d=0; d<nDim_; ++d) {
    status = solver->solve(d, iters[d], finalNorm[d], realm_.isFinalOuterIter_);
  }
  copy_hypre_to_stk(slnField, rhsNorm);
  sync_field(slnField);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    for (unsigned d=0; d < nDim_; ++d) {
      std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
      const std::string slnFile = eqSysName_ + std::to_string(d) + ".IJV." + writeCounter + ".sln";
      HYPRE_IJVectorPrint(sln_[d], slnFile.c_str());
    }
    ++eqSys_->linsysWriteCounter_;
  }

  {
    linearSolveIterations_ = 0;
    linearResidual_ = 0.0;
    nonLinearResidual_ = 0.0;
    double linres, nonlinres, scaledres, tmp, scaleFac = 0.0;

    for (unsigned d=0; d<nDim_; ++d) {
      linres = finalNorm[d] * rhsNorm[d];
      nonlinres = realm_.l2Scaling_ * rhsNorm[d];

      if (eqSys_->firstTimeStepSolve_)
        firstNLR_[d] = nonlinres;

      tmp = std::max(std::numeric_limits<double>::epsilon(), firstNLR_[d]);
      scaledres = nonlinres / tmp;
      scaleFac += tmp * tmp;

      linearResidual_ += linres * linres;
      nonLinearResidual_ += nonlinres * nonlinres;
      scaledNonLinearResidual_ += scaledres * scaledres;
      linearSolveIterations_ += iters[d];

      if (provideOutput_) {
        const int nameOffset = eqSysName_.length() + 10;

        NaluEnv::self().naluOutputP0()
          << std::setw(nameOffset) << std::right << eqSysName_+"_"+vecNames_[d]
          << std::setw(32 - nameOffset) << std::right << iters[d] << std::setw(18)
          << std::right << linres << std::setw(15) << std::right
          << nonlinres << std::setw(14) << std::right
          << scaledres << std::endl;
      }
    }
    linearResidual_ = std::sqrt(linearResidual_);
    nonLinearResidual_ = std::sqrt(nonLinearResidual_);
    scaledNonLinearResidual_ = nonLinearResidual_ / std::sqrt(scaleFac);

    if (provideOutput_) {
      const int nameOffset = eqSysName_.length() + 8;
      NaluEnv::self().naluOutputP0()
        << std::setw(nameOffset) << std::right << eqSysName_
        << std::setw(32 - nameOffset) << std::right << linearSolveIterations_ << std::setw(18)
        << std::right << linearResidual_ << std::setw(15) << std::right
        << nonLinearResidual_ << std::setw(14) << std::right
        << scaledNonLinearResidual_ << std::endl;
    }
  }

  eqSys_->firstTimeStepSolve_ = false;
  return status;
}


void
HypreUVWLinearSystem::copy_hypre_to_stk(
  stk::mesh::FieldBase* stkField, std::vector<double>& rhsNorm)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const auto sel = stk::mesh::selectField(*stkField)
    & meta.locally_owned_part()
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  const auto& bkts = bulk.get_buckets(
    stk::topology::NODE_RANK, sel);

  std::vector<double> lclnorm(nDim_, 0.0);
  std::vector<double> gblnorm(nDim_, 0.0);
  double rhsVal = 0.0;

  for (auto b: bkts) {
    double* field = (double*) stk::mesh::field_data(*stkField, *b);
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = get_entity_hypre_id(node);

      for (unsigned d=0; d<nDim_; ++d) {
        int sid = in * nDim_ + d;
        HYPRE_IJVectorGetValues(sln_[d], 1, &hid, &field[sid]);
        HYPRE_IJVectorGetValues(rhs_[d], 1, &hid, &rhsVal);
        lclnorm[d] += rhsVal * rhsVal;
      }
    }
  }

  NGPDoubleFieldType ngpField = realm_.ngp_field_manager().get_field<double>(stkField->mesh_meta_data_ordinal());
  ngpField.modify_on_host();
  ngpField.sync_to_device();

  stk::all_reduce_sum(bulk.parallel(), lclnorm.data(), gblnorm.data(), nDim_);

  for (unsigned d=0; d<nDim_; ++d)
    rhsNorm[d] = std::sqrt(gblnorm[d]);
}



sierra::nalu::CoeffApplier* HypreUVWLinearSystem::get_coeff_applier()
{
#ifdef KOKKOS_ENABLE_CUDA

  if (!hostCoeffApplier) {
    /***************************/
    /* Build the coeff applier */
    HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);
    bool ensureReproducible = solver->getConfig()->ensureReproducible();
    bool useNativeCudaAssembly = solver->getConfig()->useNativeCudaAssembly();

    hostCoeffApplier.reset(new HypreUVWLinSysCoeffApplier(useNativeCudaAssembly, ensureReproducible, nDim_, globalNumRows_, 
							  rank_, iLower_, iUpper_, jLower_, jUpper_,
							  memory_map_, row_indices_, mat_row_start_, rhs_row_start_,
							  numMatPtsToAssembleTotal_, numRhsPtsToAssembleTotal_,
							  periodic_bc_rows_, entityToLID_, skippedRowsMap_));
    deviceCoeffApplier = hostCoeffApplier->device_pointer();
  }

  /* reset the internal counters */
  hostCoeffApplier->resetInternalData();
  return deviceCoeffApplier;

#else

  return LinearSystem::get_coeff_applier();

#endif
}

/********************************************************************************************************/
/*                     Beginning of HypreUVWLinSysCoeffApplier implementations                          */
/********************************************************************************************************/

  HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::HypreUVWLinSysCoeffApplier(bool useNativeCudaAssembly, bool ensureReproducible, 
									       unsigned numDof, HypreIntType globalNumRows, int rank, 
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
									       HypreIntTypeUnorderedMap skippedRowsMap)
  : HypreLinSysCoeffApplier(useNativeCudaAssembly, ensureReproducible, numDof, globalNumRows, rank,
			    iLower, iUpper, jLower, jUpper, memory_map, row_indices, mat_row_start, rhs_row_start,
			    numMatPtsToAssembleTotal, numRhsPtsToAssembleTotal,
			    periodic_bc_rows, entityToLID, skippedRowsMap) {
  
  nDim_ = numDof;
}

KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::sum_into(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  unsigned numDof) {

  unsigned nDim = numDof;

  for(unsigned i=0; i<numEntities; ++i)
    localIds[i] = entityToLID_[entities[i].local_offset()];

  for (unsigned i=0; i<numEntities; ++i) {
    int ix = i * nDim;
    HypreIntType hid = localIds[i];
    if (checkSkippedRows_()) {
      if (skippedRowsMap_.exists(hid)) continue;
    }

    HypreIntTypeMapStruct s = memory_map_.value_at(memory_map_.find(hid));
    HypreIntType mat_row_start = s.mat;
    HypreIntType rhs_row_start = s.rhs;
    HypreIntType counter = s.counter;
    HypreIntType matIndex = mat_row_start + Kokkos::atomic_fetch_add(&mat_counter_(counter), numEntities);
    HypreIntType rhsIndex = rhs_row_start + Kokkos::atomic_fetch_add(&rhs_counter_(counter), 1);     

    int offset = 0;
    for (unsigned k=0; k<numEntities; ++k) {
      cols_(matIndex) = localIds[k];
      vals_(matIndex) = lhs(ix, offset);
      offset += nDim;
      matIndex++;
    }

    for (unsigned d=0; d<nDim; ++d) {
      int ir = ix + d;
      rhs_vals_(rhsIndex,d) = rhs[ir];
    }
  }
}

KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::operator()(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  sum_into(numEntities,entities,localIds,rhs,lhs,nDim_);
}

void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::applyDirichletBCs(Realm & realm, 
								    stk::mesh::FieldBase * solutionField,
								    stk::mesh::FieldBase * bcValuesField,
								    const stk::mesh::PartVector& parts) {
  resetInternalData();

  /************************************************************/
  /* this is a hack to get dirichlet bcs working consistently */

  /* Step 1: execute the old CPU code */
  auto& meta = realm.meta_data();

  const stk::mesh::Selector sel = (
    meta.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm.get_inactive_selector()));

  const auto& bkts = realm.get_buckets(
    stk::topology::NODE_RANK, sel);

  double diag_value = 1.0;
  std::vector<HypreIntType> tCols(0);
  std::vector<double> tVals(0);
  std::vector<std::vector<double> >trhsVals(nDim_);
  for (unsigned i=0;i<nDim_;++i) {
    trhsVals[i].resize(0);
  }

  NGPDoubleFieldType ngpSolutionField = realm.ngp_field_manager().get_field<double>(solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField = realm.ngp_field_manager().get_field<double>(bcValuesField->mesh_meta_data_ordinal());

  ngpSolutionField.sync_to_host();
  ngpBCValuesField.sync_to_host();

  for (auto b: bkts) {
    const double* solution = (double*)stk::mesh::field_data(
      *solutionField, *b);
    const double* bcValues = (double*)stk::mesh::field_data(
      *bcValuesField, *b);

    for (unsigned in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm.hypreGlobalId_, node);

      /* fill these temp values */
      tCols.push_back(hid);
      tVals.push_back(diag_value);
      
      for (unsigned d=0; d<nDim_; d++) {
        double bcval = bcValues[in*nDim_ + d] - solution[in*nDim_ + d];
	trhsVals[d].push_back(bcval);
      }
    }
  }

  /* Step 2 : allocate space in which to push the temporaries */
  HypreIntTypeView c("c",tCols.size());
  HypreIntTypeViewHost ch  = Kokkos::create_mirror_view(c);

  DoubleView v("v",tVals.size());
  DoubleViewHost vh  = Kokkos::create_mirror_view(v);

  Kokkos::View<double**> rv("rv",trhsVals[0].size(),nDim_);
  Kokkos::View<double**>::HostMirror rvh  = Kokkos::create_mirror_view(rv);

  /* Step 3 : next copy the std::vectors into the host mirrors */
  for (unsigned int i=0; i<tCols.size(); ++i) {
    ch(i) = tCols[i];
    vh(i) = tVals[i];
    for (unsigned j=0; j<nDim_;++j) {
      rvh(i,j) = trhsVals[j][i];
    }
  }

  /* Step 4 : deep copy this to device */
  Kokkos::deep_copy(c,ch);
  Kokkos::deep_copy(v,vh);
  Kokkos::deep_copy(rv,rvh);

  /* For device capture */
  auto memory_map = memory_map_;
  auto cols = cols_;
  auto vals = vals_;
  auto rhs_vals = rhs_vals_;
  auto nDim = nDim_;

  /* Step 5 : append this to the existing data structure */
  int N = (int) tCols.size();
  kokkos_parallel_for("dirichlet_bcs_UVW", N, KOKKOS_LAMBDA(const unsigned& i) {
      HypreIntType hid = c(i);
      HypreIntTypeMapStruct s = memory_map.value_at(memory_map.find(hid));
      HypreIntType matIndex = s.mat;
      HypreIntType rhsIndex = s.rhs;
      cols(matIndex)=c(i);
      vals(matIndex)=v(i);
      for (unsigned d=0; d<nDim; ++d) {
	rhs_vals(rhsIndex,d) = rv(i,d);
      }
    });
}

void HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::free_device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
#endif
}

sierra::nalu::CoeffApplier* HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (devicePointer_ != nullptr) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
  devicePointer_ = sierra::nalu::create_device_expression(*this);
#else
  devicePointer_ = this;
#endif
  return devicePointer_;
}


/*********************************************************************************************************/
/*                           End of HypreUVWLinSysCoeffApplier implementations                           */
/*********************************************************************************************************/

void
HypreUVWLinearSystem::buildNodeGraph(const stk::mesh::PartVector & parts)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_owned );

  /* counter for the number elements */
  HypreIntType count=-1;
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      count = count<0 ? 1 : count;

      stk::mesh::Entity node = b[k];
      HypreIntType hid = get_entity_hypre_id(node);
      rowCount[hid]++;
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}


void
HypreUVWLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(realm_.meta_data().side_rank(), s_owned);

  /* counter for the number elements */
  HypreIntType count=-1;
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      const HypreIntType numNodes = (HypreIntType)b.num_nodes(k);
      count = count<0 ? (HypreIntType)(numNodes) : count;

      /* get the first nodes hid */
      if (numNodes) {
	stk::mesh::Entity const * nodes = b.begin_nodes(k);
	for (HypreIntType i=0; i<numNodes; ++i) {
	  HypreIntType hid = get_entity_hypre_id(nodes[i]);
	  rowCount[hid]++;
	}
      }
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}

void
HypreUVWLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
  beginLinearSystemConstruction();

  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::EDGE_RANK, s_owned);

  /* counter for the number elements */
  HypreIntType count=-1;
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      const HypreIntType numNodes = (HypreIntType)b.num_nodes(k);
      count = count<0 ? (HypreIntType)(numNodes) : count;

      /* get the first nodes hid */
      if (numNodes) {
	stk::mesh::Entity const * nodes = b.begin_nodes(k);
	for (HypreIntType i=0; i<numNodes; ++i) {
	  HypreIntType hid = get_entity_hypre_id(nodes[i]);
	  rowCount[hid]++;
	}
      }
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}

void
HypreUVWLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::ELEM_RANK, s_owned);

  /* counter for the number elements */
  HypreIntType count=-1;
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      const HypreIntType numNodes = (HypreIntType)b.num_nodes(k);
      count = count<0 ? (HypreIntType)(numNodes) : count;

      /* get the first nodes hid */
      if (numNodes) {
	stk::mesh::Entity const * nodes = b.begin_nodes(k);
	for (HypreIntType i=0; i<numNodes; ++i) {
	  HypreIntType hid = get_entity_hypre_id(nodes[i]);
	  rowCount[hid]++;
	}
      }
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}

void
HypreUVWLinearSystem::buildFaceElemToNodeGraph(
  const stk::mesh::PartVector & parts)
{
  beginLinearSystemConstruction();
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( metaData.side_rank(), s_owned );

  /* counter for the number elements */
  HypreIntType count=-1;
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for(size_t ib=0; ib<face_buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *face_buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      const stk::mesh::Entity face = b[k];

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
      ThrowAssert( bulkData.num_elements(face) == 1 );

      // get connected element and nodal relations
      stk::mesh::Entity element = face_elem_rels[0];
      const stk::mesh::Entity* elem_nodes = bulkData.begin_nodes(element);

      // figure out the global dof ids for each dof on each node
      const HypreIntType numNodes = (HypreIntType)bulkData.num_nodes(element);
      count = count < numNodes ? (HypreIntType)(numNodes) : count;

      /* get the first nodes hid */
      if (numNodes) {
	for (HypreIntType i=0; i<numNodes; ++i) {
	  HypreIntType hid = get_entity_hypre_id(elem_nodes[i]);
	  rowCount[hid]++;
	}
      }
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}

void
HypreUVWLinearSystem::buildReducedElemToNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreUVWLinearSystem::buildNonConformalNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::PartVector& parts)
{
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  // Grab nodes regardless of whether they are owned or shared
  const stk::mesh::Selector sel = stk::mesh::selectUnion(parts);
  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

  /* counter for the number elements */
  HypreIntType count=1;
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
      skippedRows_.insert(hid);

      /* augment the counter */
      rowCount[hid]++;
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}

void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  /* counter for the number elements */
  HypreIntType count=1;
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for (const auto& node: nodeList) {
    HypreIntType hid = get_entity_hypre_id(node);
    skippedRows_.insert(hid);

    /* augment the counter */
    rowCount[hid]++;
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}

void 
HypreUVWLinearSystem::buildDirichletNodeGraph(const stk::mesh::NgpMesh::ConnectedNodes nodeList) {
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  /* counter for the number elements */
  HypreIntType count=1;
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for (unsigned i=0; i<nodeList.size();++i) {
    HypreIntType hid = get_entity_hypre_id(nodeList[i]);
    skippedRows_.insert(hid);

    /* augment the counter */
    rowCount[hid]++;
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}



}  // nalu
}  // sierra
