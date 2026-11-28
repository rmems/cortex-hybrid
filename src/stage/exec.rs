// SPDX-License-Identifier: MIT OR Apache-2.0

use super::descriptor::{
    AnnStage, DType, StageId, StageKind, StageKindTag, StageSource, StageTensor,
};
use super::topology::AnnTopology;
use crate::error::{CortexError, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// What a backend can execute, mirroring the `snn` module's capabilities style.
///
/// `#[non_exhaustive]` so additional fields (a device field is deferred to
/// RM-1827) can be added without a second breaking release. Construct with
/// [`AnnCapabilities::new`]; fields stay readable.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnCapabilities {
    /// Human-readable backend identity, e.g. `"cortex::ReferenceExecutor"`.
    pub backend_name: &'static str,
    /// Stage kinds (by tag) the backend can execute.
    pub supported_tags: BTreeSet<StageKindTag>,
    /// Element types the backend accepts and produces.
    pub supported_dtypes: BTreeSet<DType>,
    /// Whether `execute` mutates backend state across calls.
    pub stateful: bool,
}

impl AnnCapabilities {
    /// A stateless backend report: `stateful` starts `false`; set it with
    /// [`AnnCapabilities::with_stateful`].
    pub fn new(
        backend_name: &'static str,
        supported_tags: BTreeSet<StageKindTag>,
        supported_dtypes: BTreeSet<DType>,
    ) -> Self {
        Self {
            backend_name,
            supported_tags,
            supported_dtypes,
            stateful: false,
        }
    }

    /// Mark whether `execute` mutates backend state across calls. Returns
    /// `self` for chaining.
    pub fn with_stateful(mut self, stateful: bool) -> Self {
        self.stateful = stateful;
        self
    }

    /// Whether this backend can execute `kind` (by its [`StageKind::tag`]).
    pub fn supports(&self, kind: &StageKind) -> bool {
        self.supported_tags.contains(&kind.tag())
    }
}

/// One input handed to [`AnnExecutor::execute`].
///
/// Generic over the executor's tensor type so the contract never names a
/// concrete tensor.
#[non_exhaustive]
#[derive(Debug)]
pub enum StageInput<'a, T> {
    /// Token ids for an embedding ingress.
    Tokens(&'a [u32]),
    /// A tensor produced by an earlier stage or supplied as ingress.
    Tensor(&'a T),
}

/// Executes stages against backend-owned tensors.
///
/// The contract is deliberately generic: it names only the associated
/// [`AnnExecutor::Tensor`] type, never [`crate::Tensor`] or `Vec<f32>`.
///
/// Implementations must, mirroring `snn::SnnBackend`:
///
/// * **Be failure-atomic:** a rejected input (wrong arity, kind, dtype, or an
///   internal error) must leave executor state unmodified, so a caller can fail
///   closed without a half-applied stage.
/// * **Reject unsupported stages structurally:** a stage whose kind is not in
///   [`AnnCapabilities::supported_tags`] returns
///   [`CortexError::UnsupportedStage`] rather than panicking.
/// * **Never panic at the boundary:** all error conditions are reported as
///   [`CortexError`].
pub trait AnnExecutor {
    /// Backend-owned tensor type crossing the stage boundary.
    type Tensor: StageTensor;

    /// Report what this backend can execute.
    fn capabilities(&self) -> AnnCapabilities;

    /// Execute one stage against its resolved inputs, returning its output.
    ///
    /// # Errors
    ///
    /// Returns a [`CortexError`] on unsupported kinds, input arity/kind/dtype
    /// mismatches, or an internal failure. On error, executor state is
    /// unchanged. [`run_topology`] returns an error that already carries stage
    /// context (a `stage_id`-bearing variant or [`CortexError::StageFailed`])
    /// unchanged and wraps any other error in [`CortexError::StageFailed`]
    /// naming the stage.
    fn execute(
        &mut self,
        stage: &AnnStage,
        inputs: &[StageInput<Self::Tensor>],
    ) -> Result<Self::Tensor>;
}

/// One external ingress value: token ids or a tensor.
///
/// Tokens and tensors share a single namespace so a name can only ever resolve
/// to one of them; see [`ExternalBindings::with_tokens`].
#[derive(Debug)]
enum Binding<'a, T> {
    Tokens(&'a [u32]),
    Tensor(&'a T),
}

/// Supplies external ingress for [`run_topology`].
///
/// Keys are [`StageSource::External`] names. Each binding is either token ids
/// (for an embedding ingress) or a tensor (for a hidden-state ingress), and is
/// resolved into the matching [`StageInput`] when a stage references it.
///
/// Names live in **one** namespace: binding the same name twice, whether as
/// tokens and a tensor or as two tensors, is rejected with
/// [`CortexError::ConflictingStageBinding`] at bind time rather than letting
/// one binding silently shadow the other.
///
/// Separately, [`ExternalBindings::with_stage_output`] supplies the output of
/// a stage computed *outside* the topology being run, keyed by [`StageId`], for
/// the [`StageSource::Stage`] references a [`SubSpan::to_topology`](super::SubSpan::to_topology) topology
/// declares in [`AnnTopology::external_stage_deps`].
pub struct ExternalBindings<'a, T> {
    externals: BTreeMap<String, Binding<'a, T>>,
    stage_outputs: BTreeMap<StageId, &'a T>,
}

impl<'a, T> Default for ExternalBindings<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Reports bound names and their kinds (`tokens` / `tensor`), never values, so
/// `T` need not implement `Debug`.
impl<T> fmt::Debug for ExternalBindings<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for (name, binding) in &self.externals {
            map.entry(name, &binding_kind(binding));
        }
        for id in self.stage_outputs.keys() {
            map.entry(id, &"stage output");
        }
        map.finish()
    }
}

impl<'a, T> ExternalBindings<'a, T> {
    /// An empty binding set.
    pub fn new() -> Self {
        Self {
            externals: BTreeMap::new(),
            stage_outputs: BTreeMap::new(),
        }
    }

    /// Bind `name` to token ids. Returns `self` for chaining.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::ConflictingStageBinding`] when `name` is already
    /// bound (as tokens or as a tensor).
    pub fn with_tokens(self, name: impl Into<String>, tokens: &'a [u32]) -> Result<Self> {
        self.insert_external(name.into(), Binding::Tokens(tokens))
    }

    /// Bind `name` to a tensor. Returns `self` for chaining.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::ConflictingStageBinding`] when `name` is already
    /// bound (as tokens or as a tensor).
    pub fn with_tensor(self, name: impl Into<String>, tensor: &'a T) -> Result<Self> {
        self.insert_external(name.into(), Binding::Tensor(tensor))
    }

    /// Bind the externally computed output of stage `id` to a tensor, for a
    /// topology that declares `id` in [`AnnTopology::external_stage_deps`].
    /// Returns `self` for chaining.
    ///
    /// Stage outputs are keyed by [`StageId`] and are independent of the
    /// [`StageSource::External`] names bound with
    /// [`ExternalBindings::with_tensor`]. [`run_topology`] rejects a bound
    /// stage output whose id the topology itself produces.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::ConflictingStageBinding`] when `id` already has
    /// a bound output.
    pub fn with_stage_output(mut self, id: StageId, tensor: &'a T) -> Result<Self> {
        if self.stage_outputs.contains_key(&id) {
            return Err(CortexError::ConflictingStageBinding {
                name: id.to_string(),
                reason: "stage output is already bound".to_string(),
            });
        }
        self.stage_outputs.insert(id, tensor);
        Ok(self)
    }

    /// The bound output of stage `id`, if any.
    fn stage_output(&self, id: &StageId) -> Option<&'a T> {
        self.stage_outputs.get(id).copied()
    }

    fn insert_external(mut self, name: String, binding: Binding<'a, T>) -> Result<Self> {
        if let Some(existing) = self.externals.get(&name) {
            let (existing_kind, new_kind) = (binding_kind(existing), binding_kind(&binding));
            return Err(CortexError::ConflictingStageBinding {
                name,
                reason: format!(
                    "already bound as {existing_kind}; cannot also bind it as {new_kind}"
                ),
            });
        }
        self.externals.insert(name, binding);
        Ok(self)
    }

    /// Resolve `name` into a [`StageInput`], if bound.
    fn resolve(&self, name: &str) -> Option<StageInput<'a, T>> {
        self.externals.get(name).map(|binding| match *binding {
            Binding::Tokens(tokens) => StageInput::Tokens(tokens),
            Binding::Tensor(tensor) => StageInput::Tensor(tensor),
        })
    }
}

fn binding_kind<T>(binding: &Binding<'_, T>) -> &'static str {
    match binding {
        Binding::Tokens(_) => "tokens",
        Binding::Tensor(_) => "tensor",
    }
}

/// Run `executor` over `topology`, resolving ingress from `bindings`.
///
/// Stages run in topological order. For each stage, every [`StageSource`] is
/// resolved — [`StageSource::External`] from `bindings`, [`StageSource::Stage`]
/// from the outputs produced so far or, for an id in
/// [`AnnTopology::external_stage_deps`], from
/// [`ExternalBindings::with_stage_output`] — and passed to
/// [`AnnExecutor::execute`]. Each produced output's [`DType`] is checked
/// against the executor's [`AnnCapabilities::supported_dtypes`] and the output
/// is stored under its stage id. Bound stage outputs are inputs, not outputs,
/// and do not appear in the returned map. The first error encountered stops
/// the run.
///
/// Preflight fails closed **before the first [`AnnExecutor::execute`]**, so a
/// topology rejected in the checks below never leaves a stateful executor
/// half-advanced. After preflight succeeds, a later `execute` failure stops
/// the run at that stage; earlier stages have already been applied and are
/// not rolled back.
///
/// * every stage kind must be advertised in
///   [`AnnCapabilities::supported_tags`] (this uses [`StageKind::tag`]; every
///   [`StageKind::Custom`] name shares [`StageKindTag::Custom`]);
/// * every [`StageSource::External`] name and declared external stage
///   dependency must be bound; and
/// * every bound ingress tensor must report a [`DType`] in
///   [`AnnCapabilities::supported_dtypes`].
///
/// # Errors
///
/// Preflight (no stage executes when any of these fail):
///
/// * [`CortexError::UnsupportedStage`] for the first stage whose kind is not
///   in [`AnnCapabilities::supported_tags`].
/// * [`CortexError::MissingStageBinding`] when a referenced external ingress,
///   or a declared external stage dependency, has no binding; `stage_id` names
///   the stage that needed it.
/// * [`CortexError::ConflictingStageBinding`] when a bound stage output names
///   a stage the topology itself produces (it would shadow that stage).
/// * [`CortexError::StageDTypeMismatch`] when a bound ingress tensor reports a
///   dtype outside [`AnnCapabilities::supported_dtypes`]; `stage_id` names the
///   consuming stage.
///
/// During the run:
///
/// * [`CortexError::InvalidTopology`] when a stage input references an output
///   that is unexpectedly absent (should not occur for a validated topology).
/// * [`CortexError::StageDTypeMismatch`] when a produced output's dtype is
///   unsupported.
/// * [`CortexError::StageFailed`] wrapping any [`AnnExecutor::execute`] error
///   that does not already carry stage context; stage-scoped executor errors
///   pass through unchanged.
pub fn run_topology<E: AnnExecutor>(
    executor: &mut E,
    topology: &AnnTopology,
    bindings: &ExternalBindings<E::Tensor>,
) -> Result<BTreeMap<StageId, E::Tensor>> {
    run_topology_driver(executor, topology, bindings, None)
}

/// Like [`run_topology`], but the returned map holds only the outputs of the
/// stages listed in `retain`.
///
/// Every stage still executes — intermediates feed downstream stages whether
/// or not they are retained — so `retain` selects the egress set a caller gets
/// back rather than skipping work. Intermediates remain in memory until the
/// run finishes; `retain` only filters the returned map. Each id in `retain` must name a stage the
/// topology produces; a bound external stage dependency is an input, not a
/// produced output, and cannot be retained.
///
/// # Errors
///
/// Same as [`run_topology`], plus [`CortexError::InvalidTopology`] — reported
/// before anything executes — when `retain` names an id the topology does not
/// produce.
pub fn run_topology_with_egress<E: AnnExecutor>(
    executor: &mut E,
    topology: &AnnTopology,
    bindings: &ExternalBindings<E::Tensor>,
    retain: &[StageId],
) -> Result<BTreeMap<StageId, E::Tensor>> {
    run_topology_driver(executor, topology, bindings, Some(retain))
}

/// Shared driver for [`run_topology`] and [`run_topology_with_egress`].
fn run_topology_driver<E: AnnExecutor>(
    executor: &mut E,
    topology: &AnnTopology,
    bindings: &ExternalBindings<E::Tensor>,
    retain: Option<&[StageId]>,
) -> Result<BTreeMap<StageId, E::Tensor>> {
    let caps = executor.capabilities();

    // Capability pre-check: an unsupported stage fails the run before any
    // earlier stage executes.
    if let Some(stage) = topology
        .stages()
        .iter()
        .find(|stage| !caps.supports(&stage.kind))
    {
        return Err(CortexError::UnsupportedStage {
            backend: caps.backend_name,
            stage_id: stage.id.to_string(),
            kind: format!("{:?}", stage.kind),
        });
    }

    check_bindings(topology, bindings, &caps.supported_dtypes)?;

    if let Some(retain) = retain
        && let Some(unproduced) = retain.iter().find(|id| topology.stage_by_id(id).is_none())
    {
        return Err(CortexError::InvalidTopology {
            reason: format!("retained egress '{unproduced}' is not a stage in the topology"),
        });
    }

    let mut outputs: BTreeMap<StageId, E::Tensor> = BTreeMap::new();

    for stage in topology.stages() {
        let mut inputs: Vec<StageInput<E::Tensor>> = Vec::with_capacity(stage.inputs.len());
        for source in &stage.inputs {
            match source {
                StageSource::External(name) => {
                    let resolved =
                        bindings
                            .resolve(name)
                            .ok_or_else(|| CortexError::MissingStageBinding {
                                stage_id: stage.id.to_string(),
                                name: name.clone(),
                            })?;
                    inputs.push(resolved);
                }
                StageSource::Stage(dep) => {
                    let tensor = match outputs.get(dep) {
                        Some(tensor) => tensor,
                        None if topology.external_stage_deps().contains(dep) => bindings
                            .stage_output(dep)
                            .ok_or_else(|| CortexError::MissingStageBinding {
                                stage_id: stage.id.to_string(),
                                name: dep.to_string(),
                            })?,
                        None => {
                            return Err(CortexError::InvalidTopology {
                                reason: format!(
                                    "stage '{}' references output of '{dep}', which has not been produced",
                                    stage.id
                                ),
                            });
                        }
                    };
                    inputs.push(StageInput::Tensor(tensor));
                }
            }
        }

        let output = executor
            .execute(stage, &inputs)
            .map_err(|err| scope_to_stage(&stage.id, err))?;
        let dtype = output.meta().dtype;
        if !caps.supported_dtypes.contains(&dtype) {
            return Err(CortexError::StageDTypeMismatch {
                stage_id: stage.id.to_string(),
                expected: supported_dtype_list(&caps.supported_dtypes),
                got: dtype.to_string(),
            });
        }
        outputs.insert(stage.id.clone(), output);
    }

    if let Some(retain) = retain {
        let keep: BTreeSet<&StageId> = retain.iter().collect();
        outputs.retain(|id, _| keep.contains(id));
    }

    Ok(outputs)
}

/// Whether `err` already names the stage that produced it. Errors without
/// stage context are wrapped by the composition driver so a third-party
/// executor failure still reports which stage failed.
fn is_stage_scoped(err: &CortexError) -> bool {
    match err {
        CortexError::UnsupportedStage { .. }
        | CortexError::MissingStageParams { .. }
        | CortexError::StageParamsMismatch { .. }
        | CortexError::InvalidStageParams { .. }
        | CortexError::StageInputArity { .. }
        | CortexError::StageInputKind { .. }
        | CortexError::StageDTypeMismatch { .. }
        | CortexError::MissingStageBinding { .. }
        | CortexError::StageFailed { .. } => true,
        CortexError::UnsupportedOperation { stage_id, .. } => stage_id.is_some(),
        _ => false,
    }
}

/// Tag an [`AnnExecutor::execute`] error with the stage that produced it:
/// errors that already carry stage context pass through unchanged, anything
/// else is wrapped in [`CortexError::StageFailed`].
fn scope_to_stage(stage_id: &StageId, err: CortexError) -> CortexError {
    if is_stage_scoped(&err) {
        return err;
    }
    CortexError::StageFailed {
        stage_id: stage_id.to_string(),
        source: Box::new(err),
    }
}

/// The `expected` list for a [`CortexError::StageDTypeMismatch`].
fn supported_dtype_list(supported_dtypes: &BTreeSet<DType>) -> String {
    supported_dtypes
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Reject an ingress tensor whose dtype the executor does not support, naming
/// the stage that consumes it.
fn check_ingress_dtype<T: StageTensor>(
    stage_id: &StageId,
    tensor: &T,
    supported_dtypes: &BTreeSet<DType>,
) -> Result<()> {
    let dtype = tensor.meta().dtype;
    if supported_dtypes.contains(&dtype) {
        return Ok(());
    }
    Err(CortexError::StageDTypeMismatch {
        stage_id: stage_id.to_string(),
        expected: supported_dtype_list(supported_dtypes),
        got: dtype.to_string(),
    })
}

/// The stage that consumes external dependency `dep`. A validated topology
/// always has at least one (`dep` is declared only when referenced); the
/// fallback is defensive.
fn needing_stage_id<'t>(topology: &'t AnnTopology, dep: &'t StageId) -> &'t StageId {
    topology
        .stages()
        .iter()
        .find(|stage| {
            stage
                .inputs
                .iter()
                .any(|source| matches!(source, StageSource::Stage(id) if id == dep))
        })
        .map(|stage| &stage.id)
        .unwrap_or(dep)
}

/// Fail closed on bindings before anything runs: every declared external stage
/// dependency and every referenced [`StageSource::External`] name needs a
/// binding, bound tensors must report a dtype the executor supports, and no
/// bound stage output may shadow a stage the topology produces.
///
/// External names are checked in topology order, so the reported missing name
/// is the first one the run would have needed.
fn check_bindings<T: StageTensor>(
    topology: &AnnTopology,
    bindings: &ExternalBindings<T>,
    supported_dtypes: &BTreeSet<DType>,
) -> Result<()> {
    for dep in topology.external_stage_deps() {
        let stage_id = needing_stage_id(topology, dep);
        let tensor =
            bindings
                .stage_output(dep)
                .ok_or_else(|| CortexError::MissingStageBinding {
                    stage_id: stage_id.to_string(),
                    name: dep.to_string(),
                })?;
        check_ingress_dtype(stage_id, tensor, supported_dtypes)?;
    }
    for stage in topology.stages() {
        if bindings.stage_output(&stage.id).is_some() {
            return Err(CortexError::ConflictingStageBinding {
                name: stage.id.to_string(),
                reason: "a stage output is bound for a stage the topology itself produces"
                    .to_string(),
            });
        }
        for source in &stage.inputs {
            let StageSource::External(name) = source else {
                continue;
            };
            let resolved =
                bindings
                    .resolve(name)
                    .ok_or_else(|| CortexError::MissingStageBinding {
                        stage_id: stage.id.to_string(),
                        name: name.clone(),
                    })?;
            if let StageInput::Tensor(tensor) = resolved {
                check_ingress_dtype(&stage.id, tensor, supported_dtypes)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::test_support::{micro_model, stage};
    use crate::stage::{MlpKind, ReferenceExecutor, TensorMeta};
    use crate::tensor::Tensor;

    /// Wraps an `f32` reference tensor but reports another dtype in [`StageTensor::meta`].
    #[derive(Debug, Clone)]
    struct MisreportedDTypeTensor {
        inner: Tensor,
        reported: DType,
    }

    impl StageTensor for MisreportedDTypeTensor {
        fn meta(&self) -> TensorMeta {
            TensorMeta::new(self.inner.shape().to_vec(), self.reported)
        }
    }

    struct MisreportingExecutor;

    impl AnnExecutor for MisreportingExecutor {
        type Tensor = MisreportedDTypeTensor;

        fn capabilities(&self) -> AnnCapabilities {
            AnnCapabilities::new(
                "misreport",
                BTreeSet::from([StageKindTag::Add]),
                BTreeSet::from([DType::F32]),
            )
        }

        fn execute(
            &mut self,
            _stage: &AnnStage,
            inputs: &[StageInput<Self::Tensor>],
        ) -> Result<Self::Tensor> {
            let StageInput::Tensor(first) =
                inputs.first().ok_or_else(|| CortexError::StageInputArity {
                    stage_id: _stage.id.to_string(),
                    expected: 1,
                    got: inputs.len(),
                })?
            else {
                return Err(CortexError::StageInputKind {
                    stage_id: _stage.id.to_string(),
                    index: 0,
                });
            };
            Ok(MisreportedDTypeTensor {
                inner: first.inner.clone(),
                reported: DType::F16,
            })
        }
    }

    /// Counts `execute` calls so a test can prove nothing ran.
    struct CountingExecutor<'e, E: AnnExecutor> {
        inner: &'e mut E,
        executes: usize,
    }

    impl<E: AnnExecutor> AnnExecutor for CountingExecutor<'_, E> {
        type Tensor = E::Tensor;

        fn capabilities(&self) -> AnnCapabilities {
            self.inner.capabilities()
        }

        fn execute(
            &mut self,
            stage: &AnnStage,
            inputs: &[StageInput<Self::Tensor>],
        ) -> Result<Self::Tensor> {
            self.executes += 1;
            self.inner.execute(stage, inputs)
        }
    }

    /// Executor that fails with a bare kernel error carrying no stage context.
    struct BareErrorExecutor;

    impl AnnExecutor for BareErrorExecutor {
        type Tensor = Tensor;

        fn capabilities(&self) -> AnnCapabilities {
            AnnCapabilities {
                backend_name: "bare",
                supported_tags: BTreeSet::from([StageKindTag::Add]),
                supported_dtypes: BTreeSet::from([DType::F32]),
                stateful: false,
            }
        }

        fn execute(
            &mut self,
            _stage: &AnnStage,
            _inputs: &[StageInput<Self::Tensor>],
        ) -> Result<Self::Tensor> {
            Err(CortexError::ShapeMismatch {
                expected: vec![1, 4],
                got: vec![1, 2],
            })
        }
    }

    #[test]
    fn capabilities_supports_reports_expected_booleans() {
        let caps = AnnCapabilities::new(
            "test",
            BTreeSet::from([StageKindTag::Embedding, StageKindTag::DenseMlp]),
            BTreeSet::from([DType::F32]),
        );
        assert!(caps.supports(&StageKind::Embedding));
        assert!(caps.supports(&StageKind::Mlp {
            kind: MlpKind::Dense
        }));
        // Gated MLP has a distinct tag and is not advertised.
        assert!(!caps.supports(&StageKind::Mlp {
            kind: MlpKind::Gated
        }));
        assert!(!caps.supports(&StageKind::Attention));
    }

    #[test]
    fn stage_output_bindings_reject_duplicates() {
        let hidden = Tensor::try_from_vec(vec![0.0; 8], &[1, 8]).unwrap();
        let id = StageId::new("blocks.0.residual2").unwrap();
        let err = ExternalBindings::new()
            .with_stage_output(id.clone(), &hidden)
            .unwrap()
            .with_stage_output(id, &hidden)
            .unwrap_err();
        assert!(matches!(
            err,
            CortexError::ConflictingStageBinding { ref name, .. } if name == "blocks.0.residual2"
        ));
    }

    #[test]
    fn run_topology_rejects_unsupported_output_dtype() {
        let topo = AnnTopology::new(vec![
            stage(
                "pass",
                StageKind::Add,
                vec![StageSource::External("hidden".to_string())],
            ),
            stage(
                "next",
                StageKind::Add,
                vec![StageSource::Stage(StageId::new("pass").unwrap())],
            ),
        ])
        .unwrap();
        let hidden = Tensor::try_from_vec(vec![1.0; 4], &[1, 4]).unwrap();
        let ingress = MisreportedDTypeTensor {
            inner: hidden,
            reported: DType::F32,
        };
        let bindings = ExternalBindings::new()
            .with_tensor("hidden", &ingress)
            .unwrap();
        let mut exec = MisreportingExecutor;
        let err = run_topology(&mut exec, &topo, &bindings).unwrap_err();
        match err {
            CortexError::StageDTypeMismatch {
                stage_id,
                expected,
                got,
            } => {
                assert_eq!(stage_id, "pass");
                assert_eq!(expected, "f32");
                assert_eq!(got, "f16");
            }
            other => panic!("expected StageDTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn missing_external_binding_surfaces_through_driver() {
        let model = micro_model();
        let (mut exec, topo) = ReferenceExecutor::from_transformer(&model).unwrap();
        // Bind nothing: the "tokens" external ingress is unresolved.
        let bindings: ExternalBindings<Tensor> = ExternalBindings::new();
        let err = run_topology(&mut exec, &topo, &bindings).unwrap_err();
        assert!(matches!(
            err,
            CortexError::MissingStageBinding { stage_id, name }
                if stage_id == "embedding" && name == "tokens"
        ));
    }

    #[test]
    fn bindings_reject_same_name_bound_as_tokens_and_tensor() {
        let hidden = Tensor::try_from_vec(vec![0.0; 8], &[1, 8]).unwrap();
        let ids = [0u32, 1];

        // tensor first, then tokens
        let err = ExternalBindings::new()
            .with_tensor("x", &hidden)
            .unwrap()
            .with_tokens("x", &ids)
            .unwrap_err();
        assert!(
            matches!(err, CortexError::ConflictingStageBinding { ref name, .. } if name == "x"),
            "expected ConflictingStageBinding, got {err:?}"
        );

        // tokens first, then tensor
        let err = ExternalBindings::new()
            .with_tokens("x", &ids)
            .unwrap()
            .with_tensor("x", &hidden)
            .unwrap_err();
        assert!(matches!(err, CortexError::ConflictingStageBinding { .. }));

        // same kind twice is a collision too
        let err = ExternalBindings::<Tensor>::new()
            .with_tokens("x", &ids)
            .unwrap()
            .with_tokens("x", &ids)
            .unwrap_err();
        assert!(matches!(err, CortexError::ConflictingStageBinding { .. }));

        // Distinct names resolve to their own kind, whichever was bound first.
        let bindings = ExternalBindings::new()
            .with_tokens("tokens", &ids)
            .unwrap()
            .with_tensor("hidden", &hidden)
            .unwrap();
        assert!(matches!(
            bindings.resolve("tokens"),
            Some(StageInput::Tokens(_))
        ));
        assert!(matches!(
            bindings.resolve("hidden"),
            Some(StageInput::Tensor(_))
        ));
        assert!(bindings.resolve("absent").is_none());
    }

    #[test]
    fn run_topology_requires_and_guards_stage_output_bindings() {
        let model = micro_model();
        let (mut exec, topo) = ReferenceExecutor::from_transformer(&model).unwrap();
        let span = topo
            .sub_span(
                &StageId::new("final_norm").unwrap(),
                &StageId::new("readout").unwrap(),
            )
            .unwrap()
            .to_topology()
            .unwrap();
        let dep = StageId::new("blocks.1.residual2").unwrap();
        assert_eq!(span.external_stage_deps(), std::slice::from_ref(&dep));

        // Nothing bound for the declared dep: MissingStageBinding names it and
        // the stage that needed it.
        let err = run_topology(&mut exec, &span, &ExternalBindings::new()).unwrap_err();
        assert!(
            matches!(err, CortexError::MissingStageBinding { ref stage_id, ref name }
                if stage_id == "final_norm" && name == "blocks.1.residual2"),
            "got {err:?}"
        );

        // A stage output bound under an id the span itself produces would
        // shadow that stage; it is rejected before anything runs.
        let hidden = Tensor::randn(&[2, model.config.dim], 0.0, 0.1);
        let bindings = ExternalBindings::new()
            .with_stage_output(dep, &hidden)
            .unwrap()
            .with_stage_output(StageId::new("final_norm").unwrap(), &hidden)
            .unwrap();
        let err = run_topology(&mut exec, &span, &bindings).unwrap_err();
        assert!(
            matches!(err, CortexError::ConflictingStageBinding { ref name, .. } if name == "final_norm"),
            "got {err:?}"
        );
    }

    #[test]
    fn run_topology_rejects_missing_external_ingress_before_any_execute() {
        // Stage 1 only needs `hidden`; stage 2 also needs `mask`. With `mask`
        // unbound the run must fail before stage 1 executes, so a stateful
        // executor is never left half-advanced.
        let hidden = Tensor::try_from_vec(vec![1.0; 8], &[1, 8]).unwrap();
        let mask = Tensor::try_from_vec(vec![0.5; 8], &[1, 8]).unwrap();
        let topo = AnnTopology::new(vec![
            stage(
                "scale",
                StageKind::Add,
                vec![
                    StageSource::External("hidden".to_string()),
                    StageSource::External("hidden".to_string()),
                ],
            ),
            stage(
                "masked",
                StageKind::Add,
                vec![
                    StageSource::Stage(StageId::new("scale").unwrap()),
                    StageSource::External("mask".to_string()),
                ],
            ),
        ])
        .unwrap();

        let mut reference = ReferenceExecutor::new();
        let mut exec = CountingExecutor {
            inner: &mut reference,
            executes: 0,
        };
        let partial = ExternalBindings::new()
            .with_tensor("hidden", &hidden)
            .unwrap();
        let err = run_topology(&mut exec, &topo, &partial).unwrap_err();
        assert!(
            matches!(err, CortexError::MissingStageBinding { ref stage_id, ref name }
                if stage_id == "masked" && name == "mask"),
            "got {err:?}"
        );
        assert_eq!(
            exec.executes, 0,
            "no stage may execute before preflight passes"
        );

        // Fully bound, both stages run.
        let full = ExternalBindings::new()
            .with_tensor("hidden", &hidden)
            .unwrap()
            .with_tensor("mask", &mask)
            .unwrap();
        let out = run_topology(&mut exec, &topo, &full).unwrap();
        assert_eq!(exec.executes, 2);
        // scale = hidden + hidden = 2.0; masked = scale + mask = 2.5.
        assert_eq!(out[&StageId::new("masked").unwrap()].data(), &[2.5; 8]);
    }

    #[test]
    fn run_topology_rejects_unsupported_stage_before_any_execute() {
        // Stage 2's kind is not advertised by the executor's capabilities; the
        // run must fail before stage 1 executes, not after earlier stages ran.
        let hidden = Tensor::try_from_vec(vec![1.0; 8], &[1, 8]).unwrap();
        let topo = AnnTopology::new(vec![
            stage(
                "supported",
                StageKind::Add,
                vec![
                    StageSource::External("hidden".to_string()),
                    StageSource::External("hidden".to_string()),
                ],
            ),
            stage(
                "router",
                StageKind::MoeRouter {
                    num_experts: 2,
                    top_k: 1,
                },
                vec![StageSource::Stage(StageId::new("supported").unwrap())],
            ),
        ])
        .unwrap();
        let bindings = ExternalBindings::new()
            .with_tensor("hidden", &hidden)
            .unwrap();
        let mut reference = ReferenceExecutor::new();
        let mut exec = CountingExecutor {
            inner: &mut reference,
            executes: 0,
        };
        let err = run_topology(&mut exec, &topo, &bindings).unwrap_err();
        match err {
            CortexError::UnsupportedStage {
                backend,
                stage_id,
                kind,
            } => {
                assert_eq!(backend, "reference");
                assert_eq!(stage_id, "router");
                assert!(kind.contains("MoeRouter"), "kind carried: {kind}");
            }
            other => panic!("expected UnsupportedStage, got {other:?}"),
        }
        assert_eq!(
            exec.executes, 0,
            "no stage may execute before the capability pre-check passes"
        );
    }

    #[test]
    fn run_topology_rejects_unsupported_ingress_dtype_before_any_execute() {
        // A bound tensor reporting a dtype outside `supported_dtypes` is
        // rejected in preflight, naming the consuming stage.
        let hidden = MisreportedDTypeTensor {
            inner: Tensor::try_from_vec(vec![1.0; 4], &[1, 4]).unwrap(),
            reported: DType::BF16,
        };
        let right = MisreportedDTypeTensor {
            inner: Tensor::try_from_vec(vec![2.0; 4], &[1, 4]).unwrap(),
            reported: DType::F32,
        };
        let topo = AnnTopology::new(vec![stage(
            "sum",
            StageKind::Add,
            vec![
                StageSource::External("hidden".to_string()),
                StageSource::External("right".to_string()),
            ],
        )])
        .unwrap();
        let bindings = ExternalBindings::new()
            .with_tensor("hidden", &hidden)
            .unwrap()
            .with_tensor("right", &right)
            .unwrap();
        let mut misreporting = MisreportingExecutor;
        let mut exec = CountingExecutor {
            inner: &mut misreporting,
            executes: 0,
        };
        let err = run_topology(&mut exec, &topo, &bindings).unwrap_err();
        match err {
            CortexError::StageDTypeMismatch {
                stage_id,
                expected,
                got,
            } => {
                assert_eq!(stage_id, "sum");
                assert_eq!(expected, "f32");
                assert_eq!(got, "bf16");
            }
            other => panic!("expected StageDTypeMismatch, got {other:?}"),
        }
        assert_eq!(
            exec.executes, 0,
            "no stage may execute before ingress dtypes are checked"
        );
    }

    #[test]
    fn run_topology_rejects_unsupported_bound_stage_output_dtype() {
        // The same ingress dtype check applies to tensors bound for declared
        // external stage dependencies.
        let dep = StageId::new("prior").unwrap();
        let injected = MisreportedDTypeTensor {
            inner: Tensor::try_from_vec(vec![1.0; 4], &[1, 4]).unwrap(),
            reported: DType::F16,
        };
        let right = MisreportedDTypeTensor {
            inner: Tensor::try_from_vec(vec![2.0; 4], &[1, 4]).unwrap(),
            reported: DType::F32,
        };
        let topo = AnnTopology::with_external_stage_deps(
            vec![stage(
                "sum",
                StageKind::Add,
                vec![
                    StageSource::Stage(dep.clone()),
                    StageSource::External("right".to_string()),
                ],
            )],
            vec![dep.clone()],
        )
        .unwrap();
        let bindings = ExternalBindings::new()
            .with_stage_output(dep, &injected)
            .unwrap()
            .with_tensor("right", &right)
            .unwrap();
        let mut exec = MisreportingExecutor;
        let err = run_topology(&mut exec, &topo, &bindings).unwrap_err();
        match err {
            CortexError::StageDTypeMismatch {
                stage_id,
                expected,
                got,
            } => {
                assert_eq!(stage_id, "sum");
                assert_eq!(expected, "f32");
                assert_eq!(got, "f16");
            }
            other => panic!("expected StageDTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn run_topology_wraps_executor_errors_without_stage_context() {
        // A third-party executor returning a bare kernel error must surface it
        // as StageFailed naming the stage, like the reference executor's own
        // wrapping.
        let a = Tensor::try_from_vec(vec![1.0; 4], &[1, 4]).unwrap();
        let b = Tensor::try_from_vec(vec![2.0; 4], &[1, 4]).unwrap();
        let topo = AnnTopology::new(vec![stage(
            "sum",
            StageKind::Add,
            vec![
                StageSource::External("a".to_string()),
                StageSource::External("b".to_string()),
            ],
        )])
        .unwrap();
        let bindings = ExternalBindings::new()
            .with_tensor("a", &a)
            .unwrap()
            .with_tensor("b", &b)
            .unwrap();
        let mut exec = BareErrorExecutor;
        let err = run_topology(&mut exec, &topo, &bindings).unwrap_err();
        match err {
            CortexError::StageFailed { stage_id, source } => {
                assert_eq!(stage_id, "sum");
                assert!(matches!(*source, CortexError::ShapeMismatch { .. }));
            }
            other => panic!("expected StageFailed wrapping ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn run_topology_passes_stage_scoped_executor_errors_through() {
        // An executor error that already carries stage context is returned
        // unchanged rather than re-wrapped in StageFailed.
        let topo = AnnTopology::new(vec![stage("sum", StageKind::Add, vec![])]).unwrap();
        let bindings: ExternalBindings<MisreportedDTypeTensor> = ExternalBindings::new();
        let mut exec = MisreportingExecutor;
        // MisreportingExecutor answers StageInputArity, which already names
        // the stage; the driver must not wrap it again.
        let err = run_topology(&mut exec, &topo, &bindings).unwrap_err();
        match err {
            CortexError::StageInputArity { stage_id, .. } => {
                assert_eq!(stage_id, "sum");
            }
            other => panic!("expected StageInputArity, got {other:?}"),
        }
    }

    #[test]
    fn run_topology_with_egress_retains_only_requested_outputs() {
        let model = micro_model();
        let (mut exec, topo) = ReferenceExecutor::from_transformer(&model).unwrap();
        let ids = [1u32, 3, 0];
        let bindings = ExternalBindings::new().with_tokens("tokens", &ids).unwrap();
        let retain = [
            StageId::new("blocks.0.residual2").unwrap(),
            StageId::new("readout").unwrap(),
        ];
        let outputs = run_topology_with_egress(&mut exec, &topo, &bindings, &retain).unwrap();
        assert_eq!(
            outputs.keys().collect::<Vec<_>>(),
            retain.iter().collect::<Vec<_>>(),
            "only the retained egress set is returned"
        );
        let expected = model.try_forward(&ids).unwrap();
        assert_eq!(
            outputs[&StageId::new("readout").unwrap()].data(),
            expected.data()
        );
    }

    #[test]
    fn run_topology_with_egress_rejects_unproduced_ids_before_any_execute() {
        let model = micro_model();
        let (mut inner, topo) = ReferenceExecutor::from_transformer(&model).unwrap();
        let ids = [1u32, 3, 0];
        let bindings = ExternalBindings::new().with_tokens("tokens", &ids).unwrap();
        let retain = [
            StageId::new("readout").unwrap(),
            StageId::new("absent").unwrap(),
        ];
        let mut exec = CountingExecutor {
            inner: &mut inner,
            executes: 0,
        };
        let err = run_topology_with_egress(&mut exec, &topo, &bindings, &retain).unwrap_err();
        assert!(
            matches!(err, CortexError::InvalidTopology { ref reason } if reason.contains("'absent'")),
            "got {err:?}"
        );
        assert_eq!(exec.executes, 0);
    }
}
