// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral stage-execution contracts for ANN inference.
//!
//! This module defines the vocabulary a backend-agnostic orchestrator uses to
//! describe and run an ANN forward pass as an ordered graph of *stages*:
//!
//! * **Describe:** [`StageKind`] names what a stage computes, [`AnnStage`]
//!   binds an identity ([`StageId`]) and input wiring ([`StageSource`]) to a
//!   kind, and [`AnnTopology`] is a validated ordered list of stages.
//! * **Capabilities:** [`AnnCapabilities`] reports which stage kinds
//!   ([`StageKindTag`]) and [`DType`]s a backend can execute, mirroring the
//!   `snn` module's `SnnCapabilities` contract.
//! * **Execute:** the [`AnnExecutor`] trait runs one stage at a time against
//!   backend-owned tensors, and [`run_topology`] composes an executor over a
//!   topology using an [`ExternalBindings`] source for ingress.
//!
//! The public execution contract is deliberately generic: [`AnnExecutor`]
//! never names [`crate::Tensor`] or `Vec<f32>`, only its associated
//! [`AnnExecutor::Tensor`] type (bounded by [`StageTensor`]). This keeps the
//! contract usable by future accelerated backends.
//!
//! # Intentional omissions
//!
//! Stage descriptors ([`AnnStage`], [`StageSource`], [`AnnTopology`]) describe
//! identity, kind, and wiring; they do not encode a persisted wire format.
//! [`TensorMeta`] describes shape and dtype without promising native storage
//! placement or layout. At the copy boundary, [`HiddenStateBuffer`] describes
//! a dense row-major CPU `f32` payload through its validated
//! [`crate::adapter::HiddenStateMeta`].

mod descriptor;
mod exec;
mod reference;
mod topology;

#[cfg(test)]
mod test_support;

#[cfg(feature = "burn")]
pub use crate::adapter::BurnAdapter;
pub use crate::adapter::{
    AdapterCapabilities, BackendLimitations, DeviceClass, ExternalAdapter, ExternalAdapterMarker,
    HiddenStateBuffer, NegotiationDocument,
};
pub use descriptor::{
    AnnStage, DType, MlpKind, NormKind, StageId, StageKind, StageKindTag, StageSource, StageTensor,
    TensorMeta,
};
pub use exec::{
    AnnCapabilities, AnnExecutor, ExternalBindings, StageInput, run_topology,
    run_topology_with_egress,
};
pub use reference::{ReferenceExecutor, ReferenceStageParams};
pub use topology::{AnnTopology, SubSpan};
