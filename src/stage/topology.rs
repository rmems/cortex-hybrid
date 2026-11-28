// SPDX-License-Identifier: MIT OR Apache-2.0

use super::descriptor::{AnnStage, StageId, StageSource};
use crate::error::{CortexError, Result};
use std::collections::BTreeSet;

/// A validated, ordered graph of [`AnnStage`]s.
///
/// Construct with [`AnnTopology::new`], which enforces that the graph is a
/// well-formed DAG in topological order (every [`StageSource::Stage`] refers to
/// an *earlier* stage), or with [`AnnTopology::with_external_stage_deps`] /
/// [`SubSpan::to_topology`] to additionally admit references to stage outputs
/// computed outside the topology and supplied at run time through
/// [`ExternalBindings::with_stage_output`](super::ExternalBindings::with_stage_output). Carries no serde derives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnTopology {
    stages: Vec<AnnStage>,
    external_stage_deps: Vec<StageId>,
}

/// A contiguous sub-span of a topology plus the inputs it depends on from
/// outside the span.
///
/// Returned by [`AnnTopology::sub_span`]. `external_stage_deps` lists, in
/// first-seen order, the stage ids produced *before* the span that the span
/// references; `external_inputs` lists the [`StageSource::External`] ingress
/// names it references. [`SubSpan::to_topology`] turns the span into a
/// runnable [`AnnTopology`] whose `external_stage_deps` must be satisfied with
/// [`ExternalBindings::with_stage_output`](super::ExternalBindings::with_stage_output).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubSpan<'a> {
    /// The contiguous slice of stages, order preserved.
    pub stages: &'a [AnnStage],
    /// Earlier-stage outputs the span consumes but does not itself produce.
    pub external_stage_deps: Vec<StageId>,
    /// Named external ingress the span consumes.
    pub external_inputs: Vec<String>,
}

impl SubSpan<'_> {
    /// Build a runnable topology from this span.
    ///
    /// The stages are kept exactly as they are (same ids, kinds, and wiring);
    /// the span's `external_stage_deps` become the topology's declared
    /// external stage dependencies, so [`super::run_topology`] resolves each of them
    /// from [`ExternalBindings::with_stage_output`](super::ExternalBindings::with_stage_output) instead of from an earlier
    /// stage.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidTopology`] under the same conditions as
    /// [`AnnTopology::with_external_stage_deps`]; not expected for a span taken
    /// from a validated topology.
    pub fn to_topology(&self) -> Result<AnnTopology> {
        AnnTopology::with_external_stage_deps(
            self.stages.to_vec(),
            self.external_stage_deps.clone(),
        )
    }
}

impl AnnTopology {
    /// Validate and construct a topology from an ordered list of stages.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidTopology`] when the list is empty, when
    /// two stages share an id, or when any [`StageSource::Stage`] references a
    /// stage that is not strictly earlier in the list (a forward or self
    /// reference). [`StageSource::External`] inputs are always allowed.
    pub fn new(stages: Vec<AnnStage>) -> Result<Self> {
        Self::with_external_stage_deps(stages, Vec::new())
    }

    /// Validate and construct a topology whose stages may also reference the
    /// stage outputs listed in `external_stage_deps`, which are computed
    /// outside this topology and bound at run time with
    /// [`ExternalBindings::with_stage_output`](super::ExternalBindings::with_stage_output).
    ///
    /// This is how a [`SubSpan`] becomes runnable (see
    /// [`SubSpan::to_topology`]). With an empty `external_stage_deps` it is
    /// exactly [`AnnTopology::new`].
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidTopology`] when the list is empty, when
    /// two stages share an id, when any [`StageSource::Stage`] references a
    /// stage that is neither strictly earlier in the list nor declared in
    /// `external_stage_deps`, when a declared dependency is listed twice, is
    /// also produced by a stage in the list, or is not referenced by any stage.
    pub fn with_external_stage_deps(
        stages: Vec<AnnStage>,
        external_stage_deps: Vec<StageId>,
    ) -> Result<Self> {
        if stages.is_empty() {
            return Err(CortexError::InvalidTopology {
                reason: "topology must contain at least one stage".to_string(),
            });
        }
        let declared = Self::validate_declared_deps(&external_stage_deps)?;
        let mut referenced: BTreeSet<&StageId> = BTreeSet::new();
        let mut seen: BTreeSet<&StageId> = BTreeSet::new();
        for stage in &stages {
            // Validate inputs against *earlier* stages before admitting this
            // stage's own id, so a self-reference is rejected like any other
            // forward reference.
            Self::check_stage_refs(stage, &seen, &declared, &mut referenced)?;
            Self::check_dep_not_produced(stage, &declared)?;
            Self::admit_stage_id(stage, &mut seen)?;
        }
        Self::check_declared_deps_referenced(&external_stage_deps, &referenced)?;
        Ok(Self {
            stages,
            external_stage_deps,
        })
    }

    /// The declared external dependencies as a set, rejecting a duplicate.
    fn validate_declared_deps(deps: &[StageId]) -> Result<BTreeSet<&StageId>> {
        let mut declared = BTreeSet::new();
        for dep in deps {
            if !declared.insert(dep) {
                return Err(CortexError::InvalidTopology {
                    reason: format!("external stage dependency '{dep}' is declared twice"),
                });
            }
        }
        Ok(declared)
    }

    /// Every [`StageSource::Stage`] input of `stage` must name an already-seen
    /// stage or a declared external dependency; declared ones are recorded in
    /// `referenced`.
    fn check_stage_refs<'s>(
        stage: &'s AnnStage,
        seen: &BTreeSet<&StageId>,
        declared: &BTreeSet<&StageId>,
        referenced: &mut BTreeSet<&'s StageId>,
    ) -> Result<()> {
        for source in &stage.inputs {
            let StageSource::Stage(dep) = source else {
                continue;
            };
            if declared.contains(dep) {
                referenced.insert(dep);
            } else if !seen.contains(dep) {
                return Err(CortexError::InvalidTopology {
                    reason: format!(
                        "stage '{}' references stage '{dep}', which is not an earlier stage",
                        stage.id
                    ),
                });
            }
        }
        Ok(())
    }

    /// A declared external dependency may not also be produced by `stage`.
    fn check_dep_not_produced(stage: &AnnStage, declared: &BTreeSet<&StageId>) -> Result<()> {
        if declared.contains(&stage.id) {
            return Err(CortexError::InvalidTopology {
                reason: format!(
                    "external stage dependency '{}' is also produced by the topology",
                    stage.id
                ),
            });
        }
        Ok(())
    }

    /// Record `stage`'s id as produced, rejecting a duplicate.
    fn admit_stage_id<'s>(stage: &'s AnnStage, seen: &mut BTreeSet<&'s StageId>) -> Result<()> {
        if !seen.insert(&stage.id) {
            return Err(CortexError::InvalidTopology {
                reason: format!("duplicate stage id '{}'", stage.id),
            });
        }
        Ok(())
    }

    /// Every declared external dependency must be referenced by some stage.
    fn check_declared_deps_referenced(
        deps: &[StageId],
        referenced: &BTreeSet<&StageId>,
    ) -> Result<()> {
        if let Some(unused) = deps.iter().find(|dep| !referenced.contains(dep)) {
            return Err(CortexError::InvalidTopology {
                reason: format!(
                    "external stage dependency '{unused}' is not referenced by any stage"
                ),
            });
        }
        Ok(())
    }

    /// The stages in topological order.
    pub fn stages(&self) -> &[AnnStage] {
        &self.stages
    }

    /// Stage outputs this topology consumes but does not produce, in
    /// declaration order. Empty for a topology built with [`AnnTopology::new`];
    /// each entry must be bound with [`ExternalBindings::with_stage_output`](super::ExternalBindings::with_stage_output)
    /// before [`super::run_topology`] can run it.
    pub fn external_stage_deps(&self) -> &[StageId] {
        &self.external_stage_deps
    }

    /// Index of the stage with `id`, if present.
    pub fn position_by_id(&self, id: &StageId) -> Option<usize> {
        self.stages.iter().position(|stage| &stage.id == id)
    }

    /// The stage with `id`, if present.
    pub fn stage_by_id(&self, id: &StageId) -> Option<&AnnStage> {
        self.stages.iter().find(|stage| &stage.id == id)
    }

    /// Extract the contiguous span from `start` to `end` (inclusive) and the
    /// inputs it depends on from outside the span.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidTopology`] when either id is absent or
    /// `end` precedes `start`.
    pub fn sub_span(&self, start: &StageId, end: &StageId) -> Result<SubSpan<'_>> {
        let start_idx = self
            .position_by_id(start)
            .ok_or_else(|| CortexError::InvalidTopology {
                reason: format!("sub-span start '{start}' is not in the topology"),
            })?;
        let end_idx = self
            .position_by_id(end)
            .ok_or_else(|| CortexError::InvalidTopology {
                reason: format!("sub-span end '{end}' is not in the topology"),
            })?;
        if end_idx < start_idx {
            return Err(CortexError::InvalidTopology {
                reason: format!("sub-span end '{end}' precedes start '{start}'"),
            });
        }

        let span = &self.stages[start_idx..=end_idx];
        let in_span: BTreeSet<&StageId> = span.iter().map(|stage| &stage.id).collect();

        let mut external_stage_deps: Vec<StageId> = Vec::new();
        let mut external_inputs: Vec<String> = Vec::new();
        for stage in span {
            for source in &stage.inputs {
                match source {
                    StageSource::Stage(dep) => {
                        if !in_span.contains(dep) && !external_stage_deps.contains(dep) {
                            external_stage_deps.push(dep.clone());
                        }
                    }
                    StageSource::External(name) => {
                        if !external_inputs.contains(name) {
                            external_inputs.push(name.clone());
                        }
                    }
                }
            }
        }

        Ok(SubSpan {
            stages: span,
            external_stage_deps,
            external_inputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::test_support::stage;
    use crate::stage::{NormKind, StageKind};

    #[test]
    fn topology_rejects_empty() {
        let err = AnnTopology::new(Vec::new()).unwrap_err();
        assert!(matches!(err, CortexError::InvalidTopology { .. }));
    }

    #[test]
    fn topology_rejects_duplicate_ids() {
        let stages = vec![
            stage("embed", StageKind::Embedding, vec![]),
            stage("embed", StageKind::Add, vec![]),
        ];
        let err = AnnTopology::new(stages).unwrap_err();
        assert!(matches!(err, CortexError::InvalidTopology { .. }));
    }

    #[test]
    fn topology_rejects_forward_reference() {
        let stages = vec![
            stage(
                "a",
                StageKind::Add,
                vec![StageSource::Stage(StageId::new("b").unwrap())],
            ),
            stage("b", StageKind::Embedding, vec![]),
        ];
        let err = AnnTopology::new(stages).unwrap_err();
        assert!(matches!(err, CortexError::InvalidTopology { .. }));
    }

    #[test]
    fn topology_rejects_self_reference() {
        let stages = vec![stage(
            "a",
            StageKind::Add,
            vec![StageSource::Stage(StageId::new("a").unwrap())],
        )];
        let err = AnnTopology::new(stages).unwrap_err();
        assert!(matches!(err, CortexError::InvalidTopology { .. }));
    }

    #[test]
    fn topology_accepts_backward_and_external_references() {
        let stages = vec![
            stage(
                "embed",
                StageKind::Embedding,
                vec![StageSource::External("tokens".to_string())],
            ),
            stage(
                "add",
                StageKind::Add,
                vec![StageSource::Stage(StageId::new("embed").unwrap())],
            ),
        ];
        let topo = AnnTopology::new(stages).expect("valid topology");
        assert_eq!(topo.stages().len(), 2);
        assert_eq!(topo.position_by_id(&StageId::new("add").unwrap()), Some(1));
        assert!(topo.stage_by_id(&StageId::new("embed").unwrap()).is_some());
        assert!(
            topo.stage_by_id(&StageId::new("missing").unwrap())
                .is_none()
        );
    }

    #[test]
    fn sub_span_lists_external_dependencies_in_order() {
        let stages = vec![
            stage(
                "embed",
                StageKind::Embedding,
                vec![StageSource::External("tokens".to_string())],
            ),
            stage(
                "norm",
                StageKind::Normalization {
                    kind: NormKind::Layer,
                },
                vec![StageSource::Stage(StageId::new("embed").unwrap())],
            ),
            stage(
                "attn",
                StageKind::Attention,
                vec![
                    StageSource::Stage(StageId::new("norm").unwrap()),
                    StageSource::External("mask".to_string()),
                ],
            ),
            stage(
                "add",
                StageKind::Add,
                vec![
                    StageSource::Stage(StageId::new("attn").unwrap()),
                    StageSource::Stage(StageId::new("embed").unwrap()),
                ],
            ),
        ];
        let topo = AnnTopology::new(stages).expect("valid topology");

        // Span [attn, add]: depends on 'norm' output (before span), 'embed'
        // output (before span), and external 'mask'. 'attn' is produced inside.
        let span = topo
            .sub_span(
                &StageId::new("attn").unwrap(),
                &StageId::new("add").unwrap(),
            )
            .expect("valid span");
        assert_eq!(span.stages.len(), 2);
        assert_eq!(
            span.external_stage_deps,
            vec![
                StageId::new("norm").unwrap(),
                StageId::new("embed").unwrap()
            ]
        );
        assert_eq!(span.external_inputs, vec!["mask".to_string()]);
    }

    #[test]
    fn sub_span_to_topology_keeps_external_stage_deps() {
        let stages = vec![
            stage(
                "embed",
                StageKind::Embedding,
                vec![StageSource::External("tokens".to_string())],
            ),
            stage(
                "norm",
                StageKind::Normalization {
                    kind: NormKind::Layer,
                },
                vec![StageSource::Stage(StageId::new("embed").unwrap())],
            ),
            stage(
                "add",
                StageKind::Add,
                vec![
                    StageSource::Stage(StageId::new("norm").unwrap()),
                    StageSource::Stage(StageId::new("embed").unwrap()),
                ],
            ),
        ];
        let topo = AnnTopology::new(stages).expect("valid topology");
        assert!(topo.external_stage_deps().is_empty());

        let span = topo
            .sub_span(&StageId::new("add").unwrap(), &StageId::new("add").unwrap())
            .expect("valid span");
        // As a plain topology the span is invalid: it references earlier stages.
        assert!(matches!(
            AnnTopology::new(span.stages.to_vec()),
            Err(CortexError::InvalidTopology { .. })
        ));
        // As a span topology those references are declared external deps.
        let span_topo = span.to_topology().expect("span topology");
        assert_eq!(span_topo.stages(), span.stages);
        assert_eq!(
            span_topo.external_stage_deps(),
            &[
                StageId::new("norm").unwrap(),
                StageId::new("embed").unwrap()
            ]
        );
    }

    #[test]
    fn topology_with_external_stage_deps_validates_declarations() {
        let add = |deps: &[&str]| {
            stage(
                "add",
                StageKind::Add,
                deps.iter()
                    .map(|d| StageSource::Stage(StageId::new(*d).unwrap()))
                    .collect(),
            )
        };
        let id = |s: &str| StageId::new(s).unwrap();

        // Declared dep that no stage references.
        let err = AnnTopology::with_external_stage_deps(vec![add(&["x"])], vec![id("x"), id("y")])
            .unwrap_err();
        assert!(
            matches!(err, CortexError::InvalidTopology { ref reason } if reason.contains("'y'"))
        );

        // Declared dep that is also produced by the topology.
        let err = AnnTopology::with_external_stage_deps(
            vec![stage("x", StageKind::Embedding, vec![]), add(&["x"])],
            vec![id("x")],
        )
        .unwrap_err();
        assert!(
            matches!(err, CortexError::InvalidTopology { ref reason } if reason.contains("'x'"))
        );

        // Duplicate declaration.
        let err = AnnTopology::with_external_stage_deps(vec![add(&["x"])], vec![id("x"), id("x")])
            .unwrap_err();
        assert!(matches!(err, CortexError::InvalidTopology { .. }));

        // Reference that is neither earlier nor declared is still rejected.
        let err = AnnTopology::with_external_stage_deps(vec![add(&["x", "z"])], vec![id("x")])
            .unwrap_err();
        assert!(
            matches!(err, CortexError::InvalidTopology { ref reason } if reason.contains("'z'"))
        );

        // Well-formed.
        let topo =
            AnnTopology::with_external_stage_deps(vec![add(&["x"])], vec![id("x")]).expect("valid");
        assert_eq!(topo.external_stage_deps(), &[id("x")]);
    }

    #[test]
    fn sub_span_rejects_reversed_or_missing_bounds() {
        let stages = vec![
            stage("a", StageKind::Embedding, vec![]),
            stage("b", StageKind::Add, vec![]),
        ];
        let topo = AnnTopology::new(stages).expect("valid topology");
        let reversed = topo.sub_span(&StageId::new("b").unwrap(), &StageId::new("a").unwrap());
        assert!(matches!(reversed, Err(CortexError::InvalidTopology { .. })));
        let missing = topo.sub_span(&StageId::new("a").unwrap(), &StageId::new("z").unwrap());
        assert!(matches!(missing, Err(CortexError::InvalidTopology { .. })));
    }
}
