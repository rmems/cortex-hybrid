// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{CortexError, Result};
use std::fmt;

/// Element type reported through the stage boundary.
///
/// The reference backend is `f32`-only ([`DType::F32`]); the other variants
/// exist so capability negotiation and dtype checks can describe backends that
/// support reduced precision without this crate implementing them.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    /// 32-bit IEEE-754 float (the reference backend's only dtype).
    F32,
    /// 16-bit IEEE-754 half precision.
    F16,
    /// 16-bit brain floating point.
    BF16,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl DType {
    /// Stable negotiation token (`"f32"`, `"f16"`, `"bf16"`).
    pub fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
        }
    }
}

/// Shape + dtype description of a tensor crossing the stage boundary.
///
/// This is metadata only: it never carries data, so an executor can report an
/// output's shape and dtype without exposing its concrete storage.
///
/// `#[non_exhaustive]` so additional fields (a device field is deferred to
/// RM-1827) can be added without a second breaking release. Construct with
/// [`TensorMeta::new`]; fields stay readable.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    /// Row-major extents, outermost dimension first.
    pub shape: Vec<usize>,
    /// Element type.
    pub dtype: DType,
}

impl TensorMeta {
    /// Shape + dtype metadata for `shape` (row-major extents) and `dtype`.
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        Self { shape, dtype }
    }
}

/// A tensor an [`super::AnnExecutor`] can carry across the stage boundary.
///
/// Implementors expose only shape and dtype through [`TensorMeta`]; the
/// composition driver uses this to validate dtypes without touching storage.
pub trait StageTensor {
    /// Report this tensor's shape and dtype.
    fn meta(&self) -> TensorMeta;
}

/// Which normalization a [`StageKind::Normalization`] stage applies.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// Layer normalization (mean/variance over the feature axis, with bias).
    Layer,
    /// Root-mean-square normalization (no mean subtraction, no bias).
    Rms,
}

/// Which MLP a [`StageKind::Mlp`] stage applies.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    /// Dense feed-forward (single up-projection then down-projection).
    Dense,
    /// Gated feed-forward (gate × value, e.g. SwiGLU-style).
    Gated,
}

/// What a stage computes.
///
/// Executable kinds map 1:1 to a [`StageKindTag`] via [`StageKind::tag`];
/// dense and gated MLPs map to *distinct* tags so capability negotiation can
/// advertise them independently.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageKind {
    /// Token (and position) embedding lookup.
    Embedding,
    /// Multi-head self-attention.
    Attention,
    /// Normalization; see [`NormKind`].
    Normalization {
        /// Normalization flavour.
        kind: NormKind,
    },
    /// Feed-forward network; see [`MlpKind`].
    Mlp {
        /// MLP flavour.
        kind: MlpKind,
    },
    /// Mixture-of-Experts router selecting `top_k` of `num_experts`.
    MoeRouter {
        /// Number of experts the router gates over.
        num_experts: usize,
        /// Number of experts selected per token.
        top_k: usize,
    },
    /// A single Mixture-of-Experts expert.
    MoeExpert {
        /// Index of this expert within its router.
        expert_index: usize,
    },
    /// Elementwise addition of exactly two inputs (e.g. a residual
    /// connection). Any other arity is [`CortexError::StageInputArity`].
    Add,
    /// Final projection to logits / outputs.
    Readout,
    /// A backend-specific stage identified by name.
    Custom {
        /// Backend-defined stage name.
        name: String,
    },
}

/// Fieldless tag identifying an executable [`StageKind`] for capability checks.
///
/// Dense and gated MLPs get separate tags ([`StageKindTag::DenseMlp`] /
/// [`StageKindTag::GatedMlp`]), as do layer and RMS normalization, so a backend
/// can advertise support for one flavour without the other.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageKindTag {
    /// Tag for [`StageKind::Embedding`].
    Embedding,
    /// Tag for [`StageKind::Attention`].
    Attention,
    /// Tag for [`StageKind::Normalization`] with [`NormKind::Layer`].
    LayerNorm,
    /// Tag for [`StageKind::Normalization`] with [`NormKind::Rms`].
    RmsNorm,
    /// Tag for [`StageKind::Mlp`] with [`MlpKind::Dense`].
    DenseMlp,
    /// Tag for [`StageKind::Mlp`] with [`MlpKind::Gated`].
    GatedMlp,
    /// Tag for [`StageKind::MoeRouter`].
    MoeRouter,
    /// Tag for [`StageKind::MoeExpert`].
    MoeExpert,
    /// Tag for [`StageKind::Add`].
    Add,
    /// Tag for [`StageKind::Readout`].
    Readout,
    /// Tag for [`StageKind::Custom`].
    Custom,
}

impl StageKindTag {
    /// Stable negotiation token, independent of [`Debug`] formatting.
    pub fn as_str(self) -> &'static str {
        match self {
            StageKindTag::Embedding => "embedding",
            StageKindTag::Attention => "attention",
            StageKindTag::LayerNorm => "layer_norm",
            StageKindTag::RmsNorm => "rms_norm",
            StageKindTag::DenseMlp => "dense_mlp",
            StageKindTag::GatedMlp => "gated_mlp",
            StageKindTag::MoeRouter => "moe_router",
            StageKindTag::MoeExpert => "moe_expert",
            StageKindTag::Add => "add",
            StageKindTag::Readout => "readout",
            StageKindTag::Custom => "custom",
        }
    }
}

impl fmt::Display for StageKindTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl StageKind {
    /// Map this kind to its fieldless [`StageKindTag`] for capability checks.
    pub fn tag(&self) -> StageKindTag {
        match self {
            StageKind::Embedding => StageKindTag::Embedding,
            StageKind::Attention => StageKindTag::Attention,
            StageKind::Normalization {
                kind: NormKind::Layer,
            } => StageKindTag::LayerNorm,
            StageKind::Normalization {
                kind: NormKind::Rms,
            } => StageKindTag::RmsNorm,
            StageKind::Mlp {
                kind: MlpKind::Dense,
            } => StageKindTag::DenseMlp,
            StageKind::Mlp {
                kind: MlpKind::Gated,
            } => StageKindTag::GatedMlp,
            StageKind::MoeRouter { .. } => StageKindTag::MoeRouter,
            StageKind::MoeExpert { .. } => StageKindTag::MoeExpert,
            StageKind::Add => StageKindTag::Add,
            StageKind::Readout => StageKindTag::Readout,
            StageKind::Custom { .. } => StageKindTag::Custom,
        }
    }
}

/// A validated dotted-path stage identifier.
///
/// A valid id is non-empty and composed of `.`-separated segments, each of
/// which is non-empty and matches `[a-z0-9_]+` (lowercase ASCII letters,
/// digits, and underscores). Construct with [`StageId::new`]; the type is a
/// transparent newtype over the validated `String` and orders/hashes by it so
/// it can key a [`std::collections::BTreeMap`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StageId(String);

impl StageId {
    /// Validate and construct a [`StageId`].
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidStageId`] when the id is empty, has an
    /// empty segment (leading, trailing, or doubled `.`), or contains any
    /// character outside `[a-z0-9_.]`.
    pub fn new(id: impl Into<String>) -> Result<Self> {
        let id = id.into();
        if id.is_empty() {
            return Err(CortexError::InvalidStageId {
                id,
                reason: "stage id must not be empty".to_string(),
            });
        }
        for segment in id.split('.') {
            if segment.is_empty() {
                return Err(CortexError::InvalidStageId {
                    id: id.clone(),
                    reason: "stage id has an empty '.'-separated segment".to_string(),
                });
            }
            if !segment
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            {
                return Err(CortexError::InvalidStageId {
                    id: id.clone(),
                    reason: format!(
                        "segment '{segment}' must match [a-z0-9_]+ (lowercase ascii, digits, underscore)"
                    ),
                });
            }
        }
        Ok(StageId(id))
    }

    /// Borrow the validated dotted path.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where a stage input comes from.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageSource {
    /// A named external ingress supplied at run time (e.g. `"tokens"`).
    External(String),
    /// The output of an earlier stage in the topology.
    Stage(StageId),
}

/// A single stage: identity, kind, and input wiring.
///
/// Carries no serde derives — it describes the graph, not a persisted format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnStage {
    /// Unique identity of this stage within its topology.
    pub id: StageId,
    /// What the stage computes.
    pub kind: StageKind,
    /// Ordered inputs; each resolves to an external ingress or an earlier stage.
    pub inputs: Vec<StageSource>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_id_accepts_valid_dotted_paths() {
        assert_eq!(StageId::new("tokens").unwrap().as_str(), "tokens");
        assert_eq!(
            StageId::new("block_0.attn.q_proj").unwrap().as_str(),
            "block_0.attn.q_proj"
        );
    }

    #[test]
    fn stage_id_rejects_invalid_paths() {
        for bad in ["", "Block0", "a..b", ".lead", "trail.", "has space", "a.B"] {
            let err = StageId::new(bad).unwrap_err();
            assert!(
                matches!(err, CortexError::InvalidStageId { .. }),
                "expected InvalidStageId for {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn stage_kind_tag_separates_dense_and_gated_mlp() {
        let dense = StageKind::Mlp {
            kind: MlpKind::Dense,
        };
        let gated = StageKind::Mlp {
            kind: MlpKind::Gated,
        };
        assert_eq!(dense.tag(), StageKindTag::DenseMlp);
        assert_eq!(gated.tag(), StageKindTag::GatedMlp);
        assert_ne!(dense.tag(), gated.tag());

        // Normalization flavours are likewise distinct.
        assert_ne!(
            StageKind::Normalization {
                kind: NormKind::Layer
            }
            .tag(),
            StageKind::Normalization {
                kind: NormKind::Rms
            }
            .tag()
        );
    }
}
