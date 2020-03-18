// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "HypreLinearSystem.h"
#include "HypreDirectSolver.h"
#include "Realm.h"
#include "EquationSystem.h"
#include "LinearSolver.h"
#include "PeriodicManager.h"
#include "NaluEnv.h"
#include "NonConformalManager.h"
#include "overset/OversetManager.h"
#include "overset/OversetInfo.h"

#include <utils/StkHelpers.h>
#include <utils/CreateDeviceExpression.h>

// NGP Algorithms
#include "ngp_utils/NgpLoopUtils.h"

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

#include <Kokkos_Macros.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Parallel.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>

namespace sierra {
namespace nalu {

HypreLinearSystem::HypreLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver)
  : LinearSystem(realm, numDof, eqSys, linearSolver),
    name_(eqSys->name_),
    rowFilled_(0),
    rowStatus_(0),
    idBuffer_(0)
{
  rank_ = realm_.bulk_data().parallel_rank();
  partitionRowCount_.clear();
  num_nodes_per_partition_.clear();
}

HypreLinearSystem::~HypreLinearSystem()
{
  if (systemInitialized_) {
    HYPRE_IJMatrixDestroy(mat_);
    HYPRE_IJVectorDestroy(rhs_);
    HYPRE_IJVectorDestroy(sln_);
    systemInitialized_ = false;
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  if (_nHypreAssembles>0) {
    printf("\tMean HYPRE_IJMatrix/VectorAssemble Time (%d samples)=%1.5f   Total=%1.5f\n",
	   _nHypreAssembles, _hypreAssembleTime/_nHypreAssembles, _hypreAssembleTime);
  }
#endif
}

void
HypreLinearSystem::beginLinearSystemConstruction()
{
  if (inConstruction_) return;
  inConstruction_ = true;

#ifndef HYPRE_BIGINT
  // Make sure that HYPRE is compiled with 64-bit integer support when running
  // O(~1B) linear systems.
  uint64_t totalRows = (static_cast<uint64_t>(realm_.hypreNumNodes_) *
                        static_cast<uint64_t>(numDof_));
  uint64_t maxHypreSize = static_cast<uint64_t>(std::numeric_limits<HypreIntType>::max());

  if (totalRows > maxHypreSize)
    throw std::runtime_error(
      "The linear system size is greater than what HYPRE is compiled for. "
      "Please recompile with bigint support and link to Nalu");
#endif

  if (rank_ == 0) {
    iLower_ = realm_.hypreILower_;
  } else {
    iLower_ = realm_.hypreILower_ * numDof_ ;
  }

  iUpper_ = realm_.hypreIUpper_  * numDof_ - 1;
  // For now set column indices the same as row indices
  jLower_ = iLower_;
  jUpper_ = iUpper_;

  // The total number of rows handled by this MPI rank for Hypre
  numRows_ = (iUpper_ - iLower_ + 1);
  // Total number of global rows in the system
  maxRowID_ = realm_.hypreNumNodes_ * numDof_ - 1;
  globalNumRows_ = maxRowID_ + 1;

#if 0
  if (numDof_ > 0)
    std::cerr << rank_ << "\t" << numDof_ << "\t"
              << realm_.hypreILower_ << "\t" << realm_.hypreIUpper_ << "\t"
                << iLower_ << "\t" << iUpper_ << "\t"
                << numRows_ << "\t" << maxRowID_ << std::endl;
#endif
  // Allocate memory for the arrays used to track row types and row filled status.
  rowFilled_.resize(numRows_);
  rowStatus_.resize(numRows_);
  skippedRows_.clear();
  // All nodes start out as NORMAL; "build*NodeGraph" methods might alter the
  // row status to modify behavior of sumInto method.
  for (HypreIntType i=0; i < numRows_; i++)
    rowStatus_[i] = RT_NORMAL;

  auto& bulk = realm_.bulk_data();
  std::vector<const stk::mesh::FieldBase*> fVec{realm_.hypreGlobalId_};

  stk::mesh::copy_owned_to_shared(bulk, fVec);
  stk::mesh::communicate_field_data(bulk.aura_ghosting(), fVec);

  if (realm_.oversetManager_ != nullptr &&
      realm_.oversetManager_->oversetGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.oversetManager_->oversetGhosting_, fVec);

  if (realm_.nonConformalManager_ != nullptr &&
      realm_.nonConformalManager_->nonConformalGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.nonConformalManager_->nonConformalGhosting_, fVec);
  
  if (realm_.periodicManager_ != nullptr &&
      realm_.periodicManager_->periodicGhosting_ != nullptr) {
    realm_.periodicManager_->parallel_communicate_field(realm_.hypreGlobalId_);
    realm_.periodicManager_->periodic_parallel_communicate_field(
      realm_.hypreGlobalId_);
  }
}


void
HypreLinearSystem::buildNodeGraph(
  const stk::mesh::PartVector & parts)
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
      for (unsigned d=0; d<numDof_; ++d) {
	rowCount[hid*numDof_+d]++;
      }
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}


void
HypreLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & parts)
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
	  for (unsigned d=0; d<numDof_; ++d) {
	    rowCount[hid*numDof_+d]++;
	  }
	}
      }
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}

void
HypreLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const auto& ngpMesh = realm_.ngp_mesh();
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
	  for (unsigned d=0; d<numDof_; ++d) {
	    rowCount[hid*numDof_+d]++;
	  }
	}
      }
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}

void
HypreLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & parts)
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
	  for (unsigned d=0; d<numDof_; ++d) {
	    rowCount[hid*numDof_+d]++;
	  }
	}
      }
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}

void
HypreLinearSystem::buildFaceElemToNodeGraph(
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
      count = count< numNodes ? (HypreIntType)(numNodes) : count;

      /* get the first nodes hid */
      if (numNodes) {
	for (HypreIntType i=0; i<numNodes; ++i) {
	  HypreIntType hid = get_entity_hypre_id(elem_nodes[i]);
	  for (unsigned d=0; d<numDof_; ++d) {
	    rowCount[hid*numDof_+d]++;
	  }
	}
      }
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(count);
}

void
HypreLinearSystem::buildReducedElemToNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildNonConformalNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildOversetNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  // Mark all the fringe nodes as skipped so that sumInto doesn't add into these
  // rows during assembly process
  for(auto* oinfo: realm_.oversetManager_->oversetInfoVec_) {
    auto node = oinfo->orphanNode_;
    HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
    skippedRows_.insert(hid * numDof_);
  } 
}

void
HypreLinearSystem::buildDirichletNodeGraph(
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
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);

      /* augment the counter */
      for (unsigned d=0; d<numDof_; ++d) {
	HypreIntType lid = hid * numDof_ + d;
	skippedRows_.insert(lid);
	rowCount[lid]++;
      }
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(1);
}

void
HypreLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  /* counter for the number elements */
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for (const auto& node: nodeList) {
    HypreIntType hid = get_entity_hypre_id(node);

    /* augment the counter */
    for (unsigned d=0; d<numDof_; ++d) {
      HypreIntType lid = hid * numDof_ + d;
      skippedRows_.insert(lid);
      rowCount[lid]++;
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(1);
}

void 
HypreLinearSystem::buildDirichletNodeGraph(const stk::mesh::NgpMesh::ConnectedNodes nodeList) {
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  /* counter for the number elements */
  std::vector<HypreIntType> rowCount(globalNumRows_);
  std::fill(rowCount.begin(), rowCount.end(), 0);

  for (unsigned i=0; i<nodeList.size(); ++i) {
    HypreIntType hid = get_entity_hypre_id(nodeList[i]);

    /* augment the counter */
    for (unsigned d=0; d<numDof_; ++d) {
      HypreIntType lid = hid * numDof_ + d;
      skippedRows_.insert(lid);
      rowCount[lid]++;
    }
  }

  /* save these */
  partitionRowCount_.push_back(rowCount);
  num_nodes_per_partition_.push_back(1);
}

void
HypreLinearSystem::finalizeLinearSystem()
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
  fill_device_data_structures(numDof_);

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
HypreLinearSystem::finalizeSolver()
{
  MPI_Comm comm = realm_.bulk_data().parallel();
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_);
  HYPRE_IJVectorSetObjectType(rhs_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_);
  HYPRE_IJVectorSetObjectType(sln_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));
}


void HypreLinearSystem::fill_entity_to_row_mapping()
{
#ifdef KOKKOS_ENABLE_CUDA
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  stk::mesh::Selector selector = bulk.mesh_meta_data().universal_part() & !(realm_.get_inactive_selector());
  EntityToHypreIntTypeViewHost entityToLIDHost = EntityToHypreIntTypeViewHost("entityToRowLID",bulk.get_size_of_entity_index_space());
  entityToLID_ = EntityToHypreIntTypeView("entityToRowLID",bulk.get_size_of_entity_index_space());

  const stk::mesh::BucketVector& nodeBuckets = realm_.get_buckets(stk::topology::NODE_RANK, selector);
  for(const stk::mesh::Bucket* bptr : nodeBuckets) {
    const stk::mesh::Bucket& b = *bptr;
    for(size_t i=0; i<b.size(); ++i) {
      stk::mesh::Entity node = b[i];
      const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
      const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluId);
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);
      entityToLIDHost[node.local_offset()] = hid;
    }
  }
  Kokkos::deep_copy(entityToLID_, entityToLIDHost);
#endif
}


void HypreLinearSystem::fill_device_data_structures(const HypreIntType N)
{
#ifdef KOKKOS_ENABLE_CUDA
  /* Figure out the number of partitions, i.e. the number of build*NodeGraph calls */
  HypreIntType numPartitions = partitionRowCount_.size();

  /***************************************************************************/
  /* Construct the device data structures for where to  write into the lists */
  std::vector<HypreIntType> matGlobalCount(0);
  std::vector<HypreIntType> rhsGlobalCount(0);
  std::vector<HypreIntType> periodicBCs(0);
  std::vector<HypreIntType> validRows(0);
  
  numMatPtsToAssembleTotal_ = 0;
  numRhsPtsToAssembleTotal_ = 0;

  for (HypreIntType j=0; j<globalNumRows_; ++j) {
    HypreIntType matRowCount = 0;
    HypreIntType rhsRowCount = 0;
    
    /* normal rows with multiple entries */
    for (HypreIntType i=0; i<numPartitions; ++i) {
      matRowCount += partitionRowCount_[i][j]*num_nodes_per_partition_[i]*N;
      rhsRowCount += partitionRowCount_[i][j];
    }

    /* Deal with periodic BCs */
    if (j>=iLower_ && j<=iUpper_ && matRowCount==0) {
      matRowCount = 1;
      rhsRowCount = 1;
      periodicBCs.push_back(j);
    }

    /* Deal with dirichlet BCs */
    if (skippedRows_.find(j) != skippedRows_.end()) {
      if (j>=iLower_ && j<=iUpper_) {
	matRowCount = 1;
	rhsRowCount = 1;
      } else {
	matRowCount = 0;
	rhsRowCount = 0;
      }
    }

    /* Figure out the row indices affected by this rank */
    if (matRowCount>0) {
      validRows.push_back(j);
      matGlobalCount.push_back(matRowCount);
      rhsGlobalCount.push_back(rhsRowCount);
      numMatPtsToAssembleTotal_ += matRowCount;
      numRhsPtsToAssembleTotal_ += rhsRowCount;
    }
  }

  /* Get the row_indices */
  memory_map_ = HypreIntTypeMapUnorderedMap(validRows.size());
  HypreIntTypeMapUnorderedMapHost memory_map_host(validRows.size());

  row_indices_ = HypreIntTypeView("row_indices",validRows.size());
  HypreIntTypeViewHost row_indices_host = Kokkos::create_mirror_view(row_indices_);

  mat_row_start_ = HypreIntTypeView("mat_row_start",validRows.size()+1);
  HypreIntTypeViewHost mat_row_start_host = Kokkos::create_mirror_view(mat_row_start_);

  rhs_row_start_ = HypreIntTypeView("rhs_row_start",validRows.size()+1);
  HypreIntTypeViewHost rhs_row_start_host = Kokkos::create_mirror_view(rhs_row_start_);

  /* create the maps */
  mat_row_start_host(0) = 0;
  rhs_row_start_host(0) = 0;
  for (unsigned i=0; i<validRows.size(); ++i) {
    row_indices_host(i) = validRows[i];

    HypreIntTypeMapStruct s;
    s.mat = mat_row_start_host(i);
    s.rhs = rhs_row_start_host(i);
    s.counter = i;
    memory_map_host.insert(validRows[i],s);
    
    mat_row_start_host(i+1) = mat_row_start_host(i) + matGlobalCount[i];
    rhs_row_start_host(i+1) = rhs_row_start_host(i) + rhsGlobalCount[i];
  }
  Kokkos::deep_copy(memory_map_, memory_map_host);
  Kokkos::deep_copy(row_indices_, row_indices_host);
  Kokkos::deep_copy(mat_row_start_, mat_row_start_host);
  Kokkos::deep_copy(rhs_row_start_, rhs_row_start_host);

  /* Handle periodic boundary conditions */
  periodic_bc_rows_ = HypreIntTypeView("periodic_bc_rows",periodicBCs.size());
  HypreIntTypeViewHost periodic_bc_rows_host = Kokkos::create_mirror_view(periodic_bc_rows_);
  for (unsigned i=0; i<periodicBCs.size(); ++i) periodic_bc_rows_host(i) = periodicBCs[i];
  Kokkos::deep_copy(periodic_bc_rows_, periodic_bc_rows_host);

  /* skipped rows data structure */
  skippedRowsMap_ = HypreIntTypeUnorderedMap(skippedRows_.size());
  HypreIntTypeUnorderedMapHost skippedRowsMapHost(skippedRows_.size());
  for (auto t : skippedRows_) skippedRowsMapHost.insert(t);
  Kokkos::deep_copy(skippedRowsMap_, skippedRowsMapHost);

  /* clear this data so that the next time a coeffApplier is built, these get rebuilt from scratch */
  partitionRowCount_.clear();
  num_nodes_per_partition_.clear();
#endif
}


void
HypreLinearSystem::loadComplete()
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

  /* this class only has 1 rhs vector regardless of the numDofs */
  std::vector<void *> rhs(1);
  rhs[0] = (void*)(&rhs_);

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
HypreLinearSystem::loadCompleteSolver()
{
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorAssemble(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorAssemble(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));

  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  _hypreAssembleTime+=msec;
  _nHypreAssembles++;


  solver->comm_ = realm_.bulk_data().parallel();

  // Set flag to indicate zeroSystem that the matrix must be reinitialized
  // during the next invocation.
  matrixAssembled_ = true;
}

void
HypreLinearSystem::zeroSystem()
{
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  // It is unsafe to call IJMatrixInitialize multiple times without intervening
  // call to IJMatrixAssemble. This occurs during the first outer iteration (of
  // first timestep in static application and every timestep in moving mesh
  // applications) when the data structures have been created but never used and
  // zeroSystem is called for a reset. Include a check to ensure we only
  // initialize if it was previously assembled.
  
  if (matrixAssembled_) {
    HYPRE_IJMatrixInitialize(mat_);
    HYPRE_IJVectorInitialize(rhs_);
    HYPRE_IJVectorInitialize(sln_);

    // Set flag to false until next invocation of IJMatrixAssemble in loadComplete
    matrixAssembled_ = false;
  }

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parRhs_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parSln_, 0.0);

  // Prepare for matrix assembly and set all entry flags to "unfilled"
  for (HypreIntType i=0; i < numRows_; i++)
    rowFilled_[i] = RS_UNFILLED;

  // Set flag to indicate whether rows must be skipped during normal sumInto
  // process. For this to be activated, the linear system must have Dirichlet or
  // overset rows and they must be present on this processor
  if (hasSkippedRows_ && !skippedRows_.empty())
    checkSkippedRows_ = true;
}

sierra::nalu::CoeffApplier* HypreLinearSystem::get_coeff_applier()
{

#ifdef KOKKOS_ENABLE_CUDA
  if (!hostCoeffApplier) {
    /***************************/
    /* Build the coeff applier */
    HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);
    bool ensureReproducible = solver->getConfig()->ensureReproducible();

    hostCoeffApplier.reset(new HypreLinSysCoeffApplier(ensureReproducible, numDof_, globalNumRows_, rank_,
						       iLower_, iUpper_, jLower_, jUpper_,
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
/*                     Beginning of HypreLinSysCoeffApplier implementations                             */
/********************************************************************************************************/
HypreLinearSystem::HypreLinSysCoeffApplier::HypreLinSysCoeffApplier(bool ensureReproducible, unsigned numDof,
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
								    HypreIntTypeUnorderedMap skippedRowsMap)
  : ensureReproducible_(ensureReproducible), numDof_(numDof), globalNumRows_(globalNumRows),
    rank_(rank), iLower_(iLower), iUpper_(iUpper), jLower_(jLower), jUpper_(jUpper),
    memory_map_(memory_map), row_indices_(row_indices), mat_row_start_(mat_row_start), rhs_row_start_(rhs_row_start),
    numMatPtsToAssembleTotal_(numMatPtsToAssembleTotal), numRhsPtsToAssembleTotal_(numRhsPtsToAssembleTotal),
    periodic_bc_rows_(periodic_bc_rows), entityToLID_(entityToLID), skippedRowsMap_(skippedRowsMap), devicePointer_(nullptr)
{
  /* The total number of rows handled by this MPI rank for Hypre */
  numRows_ = row_indices_.extent(0);
  
  /* This 2D array gets atomically incremented each time we read the hid of the first
     node in each group of entities */
  mat_counter_ = HypreIntTypeView("mat_counter_", numRows_);
  rhs_counter_ = HypreIntTypeView("rhs_counter_", numRows_);

  /* storage for the matrix lists */
  cols_ = HypreIntTypeView("cols_",numMatPtsToAssembleTotal_);
  vals_ = DoubleView("vals_",numMatPtsToAssembleTotal_);

  /* storage for the rhs lists */
  rhs_vals_ = DoubleView2D("rhs_vals_", numRhsPtsToAssembleTotal_, numDof_);

  /* check skipped rows */
  checkSkippedRows_ = HypreIntTypeViewScalar("checkSkippedRows_");
  checkSkippedRows_() = skippedRowsMap_.size()>0 ? 1 : 0;
}

KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::sum_into(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  unsigned numDof) {
  
  unsigned numRows = numEntities * numDof;

  for(unsigned i=0; i<numEntities; ++i) {
    HypreIntType hid = entityToLID_[entities[i].local_offset()];
    for(unsigned d=0; d<numDof; ++d) {
      unsigned lid = i*numDof + d;
      localIds[lid] = hid*numDof + d;
    }
  }

  for (unsigned i=0; i<numEntities; ++i) {
    int ix = i * numDof;
    HypreIntType hid = localIds[ix];
    if (checkSkippedRows_()) {
      if (skippedRowsMap_.exists(hid)) continue;
    }

    for (unsigned d=0; d<numDof; ++d) {
      unsigned ir = ix + d;
      HypreIntTypeMapStruct s = memory_map_.value_at(memory_map_.find(localIds[ir]));
      HypreIntType mat_row_start = s.mat;
      HypreIntType rhs_row_start = s.rhs;
      HypreIntType counter = s.counter;
      HypreIntType matIndex = mat_row_start + Kokkos::atomic_fetch_add(&mat_counter_(counter), numRows);
      HypreIntType rhsIndex = rhs_row_start + Kokkos::atomic_fetch_add(&rhs_counter_(counter), 1); 
      
      const double* cur_lhs = &lhs(ir, 0);

      /* fill the matrix values */
      for (unsigned k=0; k<numRows; ++k) {
	cols_(matIndex+k) = localIds[k];
	vals_(matIndex+k) = cur_lhs[k];
      }
      /* fill the right hand side values */
      rhs_vals_(rhsIndex,0) = rhs[ir];

    }
  }
}


KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::sum_into_1DoF(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs) {
  
  for(unsigned i=0; i<numEntities; ++i)
    localIds[i] = entityToLID_[entities[i].local_offset()];

  for (unsigned i=0; i<numEntities; ++i) {
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

    const double* cur_lhs = &lhs(i, 0);

    /* fill the matrix values */
    for (unsigned k=0; k<numEntities; ++k) {
      cols_(matIndex) = localIds[k];
      vals_(matIndex) = cur_lhs[k];
      matIndex++;
    }

    /* fill the right hand side values */
    rhs_vals_(rhsIndex,0) = rhs[i];
  }
}

KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::operator()(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  if (numDof_==1)
    sum_into_1DoF(numEntities,entities,localIds,rhs,lhs);
  else
    sum_into(numEntities,entities,localIds,rhs,lhs,numDof_);
}


void
HypreLinearSystem::HypreLinSysCoeffApplier::applyDirichletBCs(Realm & realm, 
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
  std::vector<double> trhsVals(0);

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

      for (unsigned d=0; d<numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        double bcval = bcValues[in*numDof_ + d] - solution[in*numDof_ + d];
	
	/* fill these temp values */
	tCols.push_back(lid);
	tVals.push_back(diag_value);
	trhsVals.push_back(bcval);
      }
    }
  }

  /* Step 2 : allocate space in which to push the temporaries */
  HypreIntTypeView c("c",tCols.size());
  HypreIntTypeViewHost ch = Kokkos::create_mirror_view(c);

  DoubleView v("v",tVals.size());
  DoubleViewHost vh = Kokkos::create_mirror_view(v);

  DoubleView rv("rv",trhsVals.size());
  DoubleViewHost rvh = Kokkos::create_mirror_view(rv);

  /* Step 3 : next copy the std::vectors into the host mirrors */
  for (unsigned int i=0; i<tCols.size(); ++i) {
    ch(i) = tCols[i];
    vh(i) = tVals[i];
    rvh(i) = trhsVals[i];
  }
  /* Step 4 : deep copy this to device */
  Kokkos::deep_copy(c,ch);
  Kokkos::deep_copy(v,vh);
  Kokkos::deep_copy(rv,rvh);

  /* Step 5 : append this to the existing data structure */
  /* for some reason, Kokkos::parallel_for with a LAMBDA function does not compile. */

  /* For device capture */
  auto memory_map = memory_map_;
  auto cols = cols_;
  auto vals = vals_;
  auto rhs_vals = rhs_vals_;

  int N = (int) tCols.size();
  Kokkos::parallel_for("dirichlet_bcs", N, KOKKOS_LAMBDA(const unsigned& i) {
      HypreIntType hid = c(i);
      HypreIntTypeMapStruct s = memory_map.value_at(memory_map.find(hid));
      HypreIntType matIndex = s.mat;
      HypreIntType rhsIndex = s.rhs;
      cols(matIndex)=c(i);
      vals(matIndex)=v(i);
      rhs_vals(rhsIndex,0) = rv(i);
    });
}

void
HypreLinearSystem::HypreLinSysCoeffApplier::finishAssembly(void * mat, std::vector<void *> rhs, std::string name) {
  
#ifdef KOKKOS_ENABLE_CUDA

  /*********************************************************************/
  /* Memory Controller : shares temporaries between the Matrix and Rhs */
  /*********************************************************************/

  HypreIntType n1 = numMatPtsToAssembleTotal_;
  HypreIntType n2 = numRhsPtsToAssembleTotal_;

  if (!MemController_)
    MemController_ = new MemoryController<HypreIntType>(name,n1>n2 ? n1 : n2, rank_);

  /**********/
  /* Matrix */
  /**********/

  /* Build the assembler objects */
  if (!MatAssembler_)
    MatAssembler_ = new MatrixAssembler<HypreIntType>(name,ensureReproducible_,iLower_,iUpper_,jLower_,jUpper_,
						      globalNumRows_,globalNumRows_,n1,rank_,
						      row_indices_.extent(0), row_indices_.data(), mat_row_start_.data());

  /* set the temporaries from the memory controller ... ugly but it works for the time beign */
  MatAssembler_->setTemporaryDataArrayPtrs(MemController_->get_d_workspace());
  MatAssembler_->copySrcDataFromKokkos(cols_.data(), vals_.data());
  MatAssembler_->assemble();
  MatAssembler_->copyCSRMatrixToHost();  

  /* Cast these to their types ... ugly */
  HYPRE_IJMatrix hmat = *((HYPRE_IJMatrix *)mat);

  HypreIntType nr;
  HypreIntType * row_indices;
  HypreIntType * row_counts;
  HypreIntType * col_indices;
  double * values;    

  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);

  if (MatAssembler_->getHasShared()) {
    /* Set the owned part */
    nr = MatAssembler_->getNumRowsOwned();
    row_indices = MatAssembler_->getHostOwnedRowIndicesPtr();
    row_counts = MatAssembler_->getHostOwnedRowCountsPtr();
    col_indices = MatAssembler_->getHostOwnedColIndicesPtr();
    values = MatAssembler_->getHostOwnedValuesPtr();    
    HYPRE_IJMatrixSetValues(hmat, nr, row_counts, row_indices, col_indices, values);  

    /* Add the shared part */
    nr = MatAssembler_->getNumRowsShared();
    row_indices = MatAssembler_->getHostSharedRowIndicesPtr();
    row_counts = MatAssembler_->getHostSharedRowCountsPtr();
    col_indices = MatAssembler_->getHostSharedColIndicesPtr();
    values = MatAssembler_->getHostSharedValuesPtr();    
    HYPRE_IJMatrixAddToValues(hmat, nr, row_counts, row_indices, col_indices, values);  

  } else {
    /* No shared part so do the whole thing */
    nr = MatAssembler_->getNumRows();
    row_indices = MatAssembler_->getHostRowIndicesPtr();
    row_counts = MatAssembler_->getHostRowCountsPtr();
    col_indices = MatAssembler_->getHostColIndicesPtr();
    values = MatAssembler_->getHostValuesPtr();    
    HYPRE_IJMatrixSetValues(hmat, nr, row_counts, row_indices, col_indices, values);  
  }

  /* record the stop time */
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  _assembleMatTime+=msec;
  _nAssembleMat++;

  /********/
  /* Rhs */
  /********/

  /* Build the assembler objects */
  if (!RhsAssembler_)
    RhsAssembler_ = new RhsAssembler<HypreIntType>(name,ensureReproducible_,iLower_,iUpper_,globalNumRows_,n2,rank_,
						   row_indices_.extent(0), row_indices_.data(), rhs_row_start_.data());

  /* set the temporaries from the memory controller ... ugly but it works for the time beign */
  RhsAssembler_->setTemporaryDataArrayPtrs(MemController_->get_d_workspace());

  for (unsigned i=0; i<rhs.size(); ++i) {

    /* get the src data from the kokkos views */
    RhsAssembler_->copySrcDataFromKokkos(&rhs_vals_(0,i));
    RhsAssembler_->assemble();
    RhsAssembler_->copyRhsVectorToHost();  

    /* record the start time */
    gettimeofday(&_start, NULL);

    /* Cast these to their types ... ugly */
    HYPRE_IJVector hrhs = *((HYPRE_IJVector *)rhs[i]);
    HypreIntType nr;
    HypreIntType * rhs_indices;
    double * rhs_values;

    if (RhsAssembler_->getHasShared()) {
      /* Set the owned part */
      nr = RhsAssembler_->getNumRowsOwned();
      rhs_indices = RhsAssembler_->getHostOwnedRhsIndicesPtr();
      rhs_values = RhsAssembler_->getHostOwnedRhsPtr();    
      HYPRE_IJVectorSetValues(hrhs, nr, rhs_indices, rhs_values);
      
      /* Add the shared part */
      nr = RhsAssembler_->getNumRowsShared();
      rhs_indices = RhsAssembler_->getHostSharedRhsIndicesPtr();
      rhs_values = RhsAssembler_->getHostSharedRhsPtr();    
      HYPRE_IJVectorAddToValues(hrhs, nr, rhs_indices, rhs_values);

    } else {
      /* No shared part so do the whole thing */
      nr = RhsAssembler_->getNumRows();
      rhs_indices = RhsAssembler_->getHostRhsIndicesPtr();
      rhs_values = RhsAssembler_->getHostRhsPtr();
      HYPRE_IJVectorSetValues(hrhs, nr, rhs_indices, rhs_values);
    }

    /* record the stop time */
    gettimeofday(&_stop, NULL);
    double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
    _assembleRhsTime+=msec;
    _nAssembleRhs++;
  }

#endif //KOKKOS_ENABLE_CUDA

  /* Reset after assembly */
  reinitialize_= true;

}


void
HypreLinearSystem::HypreLinSysCoeffApplier::resetInternalData() {

  Kokkos::deep_copy(checkSkippedRows_, 1);
  if (reinitialize_) {
    reinitialize_ = false;
    
    /* For device capture */
    auto periodic_bc_rows = periodic_bc_rows_;
    auto mat_counter = mat_counter_;
    auto rhs_counter = rhs_counter_;
    auto cols = cols_;
    auto vals = vals_;
    auto rhs_vals = rhs_vals_;
    auto numDof = numDof_;
    
    /* These seem slightly faster than deep copies */
    Kokkos::parallel_for("init1", numRows_, KOKKOS_LAMBDA(const unsigned& i) {
	mat_counter(i) = 0;
	rhs_counter(i) = 0;
      });
    

    /* These seem slightly faster than deep copies */
    Kokkos::parallel_for("init2", numMatPtsToAssembleTotal_, KOKKOS_LAMBDA(const unsigned& i) {
	cols(i) = -1;
	vals(i) = 0;
      });
    
    /* These seem slightly faster than deep copies */
    Kokkos::parallel_for("init3", numRhsPtsToAssembleTotal_, KOKKOS_LAMBDA(const unsigned& i) {
	for (unsigned j=0; j<numDof; ++j) {
	  rhs_vals(i,j) = 0;
	}
      });

    /* Apply periodic boundary conditions */
    int N = periodic_bc_rows_.extent(0); 
    auto memory_map = memory_map_;
    Kokkos::parallel_for("periodic_bcs", N, KOKKOS_LAMBDA(const unsigned& i) {
	HypreIntType hid = periodic_bc_rows(i);
	HypreIntTypeMapStruct s = memory_map.value_at(memory_map.find(hid));
	HypreIntType matIndex = s.mat;
	HypreIntType rhsIndex = s.rhs;
	cols(matIndex) = hid;
	vals(matIndex) = 1.0;
	for (unsigned d=0; d<numDof; ++d)
	  rhs_vals(rhsIndex,d) = 0.0;
      }); 
  }
}

void HypreLinearSystem::HypreLinSysCoeffApplier::free_device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
#endif
}

sierra::nalu::CoeffApplier* HypreLinearSystem::HypreLinSysCoeffApplier::device_pointer()
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

/********************************************************************************************************/
/*                           End of HypreLinSysCoeffApplier implementations                             */
/********************************************************************************************************/


void
HypreLinearSystem::sumInto(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<int*, DeviceShmem>&,
  const char*  /* trace_tag */)
{
#ifndef KOKKOS_ENABLE_CUDA
  const size_t n_obj = numEntities;
  HypreIntType numRows = n_obj * numDof_;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    HypreIntType hid = get_entity_hypre_id(entities[in]);
    HypreIntType localOffset = hid * numDof_;
    for (size_t d=0; d < numDof_; d++) {
      size_t lid = in * numDof_ + d;
      idBuffer_[lid] = localOffset + d;
    }
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * numDof_;
    HypreIntType hid = idBuffer_[ix];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) {
	continue;
      }
    }

    for (size_t d=0; d < numDof_; d++) {
      int ir = ix + d;
      HypreIntType lid = idBuffer_[ir];

      const double* cur_lhs = &lhs(ir, 0);
      HYPRE_IJMatrixAddToValues(mat_, 1, &numRows, &lid,
                                &idBuffer_[0], cur_lhs);
      HYPRE_IJVectorAddToValues(rhs_, 1, &lid, &rhs[ir]);

      if ((lid >= iLower_) && (lid <= iUpper_))
        rowFilled_[lid - iLower_] = RS_FILLED;
    }
  }
#endif
}


void
HypreLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>&  /* scratchIds */,
  std::vector<double>& scratchVals,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char*  /* trace_tag */)
{
  const size_t n_obj = entities.size();
  HypreIntType numRows = n_obj * numDof_;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssert(numRows == static_cast<HypreIntType>(rhs.size()));
  ThrowAssert(numRows*numRows == static_cast<HypreIntType>(lhs.size()));

  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    HypreIntType hid = get_entity_hypre_id(entities[in]);
    HypreIntType localOffset = hid * numDof_;
    for (size_t d=0; d < numDof_; d++) {
      size_t lid = in * numDof_ + d;
      idBuffer_[lid] = localOffset + d;
    }
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * numDof_;
    HypreIntType hid = idBuffer_[ix];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) {
	continue;
      }
    }

    for (size_t d=0; d < numDof_; d++) {
      int ir = ix + d;
      HypreIntType lid = idBuffer_[ir];

      for (int c=0; c < numRows; c++)
        scratchVals[c] = lhs[ir * numRows + c];

      HYPRE_IJMatrixAddToValues(mat_, 1, &numRows, &lid,
                                &idBuffer_[0], &scratchVals[0]);
      HYPRE_IJVectorAddToValues(rhs_, 1, &lid, &rhs[ir]);

      if ((lid >= iLower_) && (lid <= iUpper_))
        rowFilled_[lid - iLower_] = RS_FILLED;
    }
  }
}

void
HypreLinearSystem::applyDirichletBCs(
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

      for (size_t d=0; d<numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        double bcval = bcValues[in*numDof_ + d] - solution[in*numDof_ + d];
        HYPRE_IJMatrixSetValues(mat_, 1, &ncols, &lid, &lid, &diag_value);
        HYPRE_IJVectorSetValues(rhs_, 1, &lid, &bcval);
        rowFilled_[lid - iLower_] = RS_FILLED;
      }
    }
  }
#endif
  adbc_time += NaluEnv::self().nalu_time();
}

HypreIntType
HypreLinearSystem::get_entity_hypre_id(const stk::mesh::Entity& node)
{
  auto& bulk = realm_.bulk_data();
  const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
  const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluId);
#ifndef NDEBUG
  if (!bulk.is_valid(node))
    throw std::runtime_error("BAD STK NODE");
#endif
  HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);

#ifndef NDEBUG
  HypreIntType chk = ((hid+1) * numDof_ - 1);
  if ((hid < 0) || (chk > maxRowID_)) {
    std::cerr << bulk.parallel_rank() << "\t"
              << hid << "\t" << iLower_ << "\t" << iUpper_ << std::endl;
    throw std::runtime_error("BAD STK to hypre conversion");
  }
#endif

  return hid;
}

int
HypreLinearSystem::solve(stk::mesh::FieldBase* linearSolutionField)
{
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(
    linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string matFile = eqSysName_ + ".IJM." + writeCounter + ".mat";
    const std::string rhsFile = eqSysName_ + ".IJV." + writeCounter + ".rhs";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());
    HYPRE_IJVectorPrint(rhs_, rhsFile.c_str());
  }

  int iters = 0;
  double finalResidNorm = 0.0;

  // Call solve
  int status = 0;

  status = solver->solve(iters, finalResidNorm, realm_.isFinalOuterIter_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string slnFile = eqSysName_ + ".IJV." + writeCounter + ".sln";
    HYPRE_IJVectorPrint(sln_, slnFile.c_str());
    ++eqSys_->linsysWriteCounter_;
  }

  double norm2 = copy_hypre_to_stk(linearSolutionField);
  sync_field(linearSolutionField);

  linearSolveIterations_ = iters;
  // Hypre provides relative residuals not the final residual, so multiply by
  // the non-linear residual to obtain a final residual that is comparable to
  // what is reported by TpetraLinearSystem. Note that this assumes the initial
  // solution vector is set to 0 at the start of linear iterations.
  linearResidual_ = finalResidNorm * norm2;
  nonLinearResidual_ = realm_.l2Scaling_ * norm2;

  if (eqSys_->firstTimeStepSolve_)
    firstNonLinearResidual_ = nonLinearResidual_;

  scaledNonLinearResidual_ =
    nonLinearResidual_ /
    std::max(std::numeric_limits<double>::epsilon(), firstNonLinearResidual_);

  if (provideOutput_) {
    const int nameOffset = eqSysName_.length() + 8;
    NaluEnv::self().naluOutputP0()
      << std::setw(nameOffset) << std::right << eqSysName_
      << std::setw(32 - nameOffset) << std::right << iters << std::setw(18)
      << std::right << linearResidual_ << std::setw(15) << std::right
      << nonLinearResidual_ << std::setw(14) << std::right
      << scaledNonLinearResidual_ << std::endl;
  }

  eqSys_->firstTimeStepSolve_ = false;
  return status;
}

double
HypreLinearSystem::copy_hypre_to_stk(
  stk::mesh::FieldBase* stkField)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const auto sel = stk::mesh::selectField(*stkField)
    & meta.locally_owned_part()
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  const auto& bkts = bulk.get_buckets(
    stk::topology::NODE_RANK, sel);

  double lclnorm2 = 0.0;
  double rhsVal = 0.0;

  for (auto b: bkts) {
    double* field = (double*) stk::mesh::field_data(*stkField, *b);
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = get_entity_hypre_id(node);

      for (size_t d=0; d < numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        int sid = in * numDof_ + d;
        HYPRE_IJVectorGetValues(sln_, 1, &lid, &field[sid]);
        HYPRE_IJVectorGetValues(rhs_, 1, &lid, &rhsVal);
        lclnorm2 += rhsVal * rhsVal;
      }
    }
  }
  NGPDoubleFieldType ngpField = realm_.ngp_field_manager().get_field<double>(stkField->mesh_meta_data_ordinal());  
  ngpField.modify_on_host();
  ngpField.sync_to_device();

  double gblnorm2 = 0.0;
  stk::all_reduce_sum(bulk.parallel(), &lclnorm2, &gblnorm2, 1);

  return std::sqrt(gblnorm2);
}

}  // nalu
}  // sierra
