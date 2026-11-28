// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{AnnStage, StageId, StageKind, StageSource};
use crate::transformer::{TransformerConfig, TransformerLM};

pub(super) fn stage(id: &str, kind: StageKind, inputs: Vec<StageSource>) -> AnnStage {
    AnnStage {
        id: StageId::new(id).expect("valid id"),
        kind,
        inputs,
    }
}

/// A minimal, valid model so executor tests stay fast.
pub(super) fn micro_model() -> TransformerLM {
    let cfg = TransformerConfig {
        vocab_size: 6,
        dim: 8,
        num_heads: 2,
        num_layers: 2,
        ff_dim: 16,
        max_seq_len: 4,
    };
    TransformerLM::try_new(cfg).expect("valid micro model")
}
