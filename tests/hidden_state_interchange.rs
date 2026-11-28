// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public CPU hidden-state copy contract, independent of backend storage.

use cortex_hybrid::adapter::{DeviceClass, HiddenStateBuffer, HiddenStateLayout};
use cortex_hybrid::error::ReportedDType;
use cortex_hybrid::stage::DType;
use cortex_hybrid::{CortexError, ReferenceExecutor, Tensor};

#[test]
fn reference_buffer_exposes_validated_host_metadata() {
    let values = [1.0, -0.0, 0.5, -2.0, 3.0, 4.0];
    let tensor = Tensor::try_from_vec(values.to_vec(), &[2, 3]).unwrap();
    let buffer = ReferenceExecutor::export_hidden(&tensor).unwrap();
    let meta = buffer.cpu_f32_meta().unwrap();

    assert_eq!(meta.shape, [2, 3]);
    assert_eq!(meta.dtype, DType::F32);
    assert_eq!(meta.device, DeviceClass::Cpu);
    assert_eq!(meta.layout, HiddenStateLayout::RowMajor);
    assert!(std::ptr::eq(meta.shape, buffer.shape.as_slice()));
    let restored = ReferenceExecutor::import_hidden(&buffer).unwrap();
    assert_eq!(restored.shape(), meta.shape);
    assert_eq!(
        restored
            .data()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        values.map(f32::to_bits)
    );
}

#[test]
fn portable_metadata_rejects_invalid_dtype_length_and_overflow() {
    fn metadata_failure(buffer: &HiddenStateBuffer) -> CortexError {
        buffer.cpu_f32_meta().unwrap_err()
    }

    let mut buffer = HiddenStateBuffer {
        data: vec![1.0],
        shape: vec![1],
        dtype: DType::F16,
    };
    assert!(matches!(
        metadata_failure(&buffer),
        CortexError::StageDTypeMismatch {
            stage_id,
            expected,
            got: ReportedDType::Stage(DType::F16),
        } if stage_id == "hidden_state" && expected == [DType::F32]
    ));

    buffer.dtype = DType::F32;
    buffer.shape = vec![2, 3];
    assert!(matches!(
        metadata_failure(&buffer),
        CortexError::StageFailed { stage_id, source }
            if stage_id == "hidden_state"
                && matches!(*source, CortexError::ShapeMismatch { .. })
    ));

    buffer.data.clear();
    buffer.shape = vec![usize::MAX, 2];
    assert!(matches!(
        metadata_failure(&buffer),
        CortexError::StageFailed { stage_id, source }
            if stage_id == "hidden_state"
                && matches!(*source, CortexError::SizeOverflow { .. })
    ));

    buffer.shape.push(0);
    assert_eq!(buffer.cpu_f32_meta().unwrap().shape, [usize::MAX, 2, 0]);

    buffer.shape.clear();
    buffer.data.push(2.0);
    assert!(buffer.cpu_f32_meta().unwrap().shape.is_empty());
}

#[test]
#[cfg(feature = "candle")]
fn reference_and_candle_exchange_stage_outputs_through_the_cpu_buffer() {
    use cortex_hybrid::adapter::{CandleExecutor, export_hidden_buffer, import_hidden_buffer};
    use cortex_hybrid::stage::{
        AnnStage, AnnTopology, ExternalBindings, StageId, StageKind, StageSource, run_topology,
    };

    let tensor = Tensor::try_from_vec(vec![1.0, -0.0, 0.5, -2.0, 3.0, 4.0], &[2, 3]).unwrap();
    let id = StageId::new("add").unwrap();
    let topology = AnnTopology::new(vec![AnnStage {
        id: id.clone(),
        kind: StageKind::Add,
        inputs: vec![
            StageSource::External("left".into()),
            StageSource::External("right".into()),
        ],
    }])
    .unwrap();

    let reference = run_topology(
        &mut ReferenceExecutor::new(),
        &topology,
        &ExternalBindings::new()
            .with_tensor("left", &tensor)
            .unwrap()
            .with_tensor("right", &tensor)
            .unwrap(),
    )
    .unwrap();
    let egress = ReferenceExecutor::export_hidden(&reference[&id]).unwrap();
    let candle_ingress = import_hidden_buffer(&egress).unwrap();
    let candle = run_topology(
        &mut CandleExecutor::new(),
        &topology,
        &ExternalBindings::new()
            .with_tensor("left", &candle_ingress)
            .unwrap()
            .with_tensor("right", &candle_ingress)
            .unwrap(),
    )
    .unwrap();
    let exported = export_hidden_buffer(&candle[&id]).unwrap();
    assert_eq!(
        exported.cpu_f32_meta().unwrap(),
        egress.cpu_f32_meta().unwrap()
    );

    let restored = ReferenceExecutor::import_hidden(&exported).unwrap();
    assert_eq!(restored.shape(), [2, 3]);
    assert_eq!(
        restored
            .data()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        [4.0f32, -0.0, 2.0, -8.0, 12.0, 16.0].map(f32::to_bits)
    );
}

#[test]
#[cfg(feature = "candle")]
fn candle_bridge_rejects_nonportable_or_native_overflowing_shapes() {
    use cortex_hybrid::adapter::import_hidden_buffer;

    let mut buffer = HiddenStateBuffer {
        data: Vec::new(),
        shape: vec![usize::MAX, 2, 0],
        dtype: DType::BF16,
    };
    assert!(matches!(
        import_hidden_buffer(&buffer),
        Err(CortexError::StageDTypeMismatch {
            got: ReportedDType::Stage(DType::BF16),
            ..
        })
    ));

    buffer.dtype = DType::F32;
    assert!(buffer.cpu_f32_meta().is_ok());
    assert!(matches!(
        import_hidden_buffer(&buffer),
        Err(CortexError::SizeOverflow { shape }) if shape == buffer.shape
    ));

    buffer.shape = vec![2, 3];
    assert!(matches!(
        import_hidden_buffer(&buffer),
        Err(CortexError::StageFailed { source, .. })
            if matches!(*source, CortexError::ShapeMismatch { .. })
    ));
}

#[test]
#[cfg(feature = "burn")]
fn valid_metadata_does_not_imply_burn_transfer_support() {
    use cortex_hybrid::adapter::{BurnAdapter, ExternalAdapter};

    let burn = BurnAdapter::unavailable();
    let buffer = HiddenStateBuffer {
        data: vec![1.0],
        shape: vec![1, 1],
        dtype: DType::F32,
    };
    assert!(buffer.cpu_f32_meta().is_ok());
    assert!(!ExternalAdapter::capabilities(&burn).hidden_state_io);
    assert!(matches!(
        ExternalAdapter::import_hidden(&burn, &buffer),
        Err(CortexError::UnsupportedOperation {
            category: "policy",
            ..
        })
    ));
}
