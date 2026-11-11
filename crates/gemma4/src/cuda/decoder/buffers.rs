//! Prepared, charged activation and full-history continuation storage.
use super::DEVICE_ONLY_STAGING;
use crate::{AttentionKind, Gemma4Config, Gemma4Error};
use cudarc::driver::CudaStream;
use engine_nvidia::{CudaBf16Error, CudaBf16Matrix};
use ribn_foundation::BytePool;
use std::sync::Arc;

/// Device-resident keys and values for one KV owner layer.
pub(in crate::cuda) struct DeviceCache {
    pub(in crate::cuda) keys: CudaBf16Matrix,
    pub(in crate::cuda) values: CudaBf16Matrix,
}

// Keep field names, shapes and extraction order together. The generated types
// remain concrete CUDA matrices; this is only the workspace's fixed layout.
macro_rules! workspace_layout {
    (
        shared($config:ident, $rows:ident) {
            $($shared:ident: $shape:expr),+ $(,)?
        }
        geometry($gc:ident, $gr:ident, $width:ident, $q:ident, $kv:ident) {
            $($field:ident: $columns:expr),+ $(,)?
        }
    ) => {
        /// Model-wide buffers shared by every layer.
        pub(in crate::cuda) struct Buffers {
            $(pub(in crate::cuda) $shared: CudaBf16Matrix,)+
            pub(in crate::cuda) local: GeometryBuffers,
            pub(in crate::cuda) global: GeometryBuffers,
        }

        /// Per-geometry working set; local and global layers differ in head width.
        pub(in crate::cuda) struct GeometryBuffers {
            pub(in crate::cuda) width: usize,
            $(pub(in crate::cuda) $field: CudaBf16Matrix,)+
        }

        impl Buffers {
            /// Retryable, idempotent retirement without dismantling the workspace.
            pub(in crate::cuda) fn close(&mut self) -> Result<(), CudaBf16Error> {
                $(self.$shared.close()?;)+
                self.local.close()?;
                self.global.close()
            }

            const MATRIX_COUNT: usize = [$(stringify!($shared)),+].len();

            fn layout(
                $config: &Gemma4Config,
                $rows: usize,
            ) -> Result<[MatrixLayout; Self::MATRIX_COUNT], CudaBf16Error> {
                Ok([$($shape?,)+])
            }

            fn take(
                matrices: &mut impl Iterator<Item = CudaBf16Matrix>,
                local_width: usize,
                global_width: usize,
            ) -> Self {
                Self {
                    $($shared: matrices.next().expect("complete workspace layout"),)+
                    local: GeometryBuffers::take(matrices, local_width),
                    global: GeometryBuffers::take(matrices, global_width),
                }
            }
        }

        impl GeometryBuffers {
            /// A failed close leaves every field reachable for retirement retry.
            pub(in crate::cuda) fn close(&mut self) -> Result<(), CudaBf16Error> {
                $(self.$field.close()?;)+
                Ok(())
            }

            const MATRIX_COUNT: usize = [$(stringify!($field)),+].len();

            fn layout(
                $gc: &Gemma4Config,
                $gr: usize,
                $width: usize,
            ) -> Result<[MatrixLayout; Self::MATRIX_COUNT], CudaBf16Error> {
                let $q = $gc.num_attention_heads().checked_mul($width)
                    .ok_or(CudaBf16Error::Invalid("query width overflow"))?;
                let $kv = $gc.num_key_value_heads().checked_mul($width)
                    .ok_or(CudaBf16Error::Invalid("KV width overflow"))?;
                Ok([$(MatrixLayout::device($gr, $columns)?,)+])
            }

            fn take(
                matrices: &mut impl Iterator<Item = CudaBf16Matrix>,
                width: usize,
            ) -> Self {
                Self {
                    width,
                    $($field: matrices.next().expect("complete geometry layout"),)+
                }
            }
        }
    };
}

workspace_layout! {
    shared(config, rows) {
        hidden: MatrixLayout::staged(rows, config.hidden_size()),
        hidden_next: MatrixLayout::device(rows, config.hidden_size()),
        final_input: MatrixLayout::device(1, config.hidden_size()),
        final_norm: MatrixLayout::device(1, config.hidden_size()),
        logits: MatrixLayout::staged(1, config.vocab_size()),
        ple_projection: MatrixLayout::device(rows, config.packed_ple_dim()),
        ple_projected_scaled: MatrixLayout::device(rows, config.packed_ple_dim()),
        ple_normed: MatrixLayout::staged(rows, config.packed_ple_dim()),
        ple_context: MatrixLayout::staged(rows, config.ple_dim()),
    }
    geometry(config, rows, width, q, kv) {
        cos: width,
        sin: width,
        normed: config.hidden_size(),
        queries: q,
        query_norm: q,
        query_rotary: q,
        keys: kv,
        key_norm: kv,
        key_rotary: kv,
        values: kv,
        value_norm: kv,
        attended: q,
        projected: config.hidden_size(),
        post_attention: config.hidden_size(),
        ffn_norm: config.hidden_size(),
        gate: config.intermediate_size(),
        up: config.intermediate_size(),
        activated: config.intermediate_size(),
        product: config.intermediate_size(),
        down: config.hidden_size(),
        post_feedforward: config.hidden_size(),
        ple_gate: config.ple_dim(),
        ple_activated: config.ple_dim(),
        ple_product: config.ple_dim(),
        ple_projected: config.hidden_size(),
        post_ple: config.hidden_size(),
    }
}

struct MatrixLayout {
    rows: usize,
    columns: usize,
    staging: usize,
}

impl MatrixLayout {
    fn device(rows: usize, columns: usize) -> Result<Self, CudaBf16Error> {
        // Match the matrix factory's dimension/alignment contract before the
        // first allocation, including geometry used only by the final layer kind.
        if rows == 0 || columns == 0 {
            return Err(CudaBf16Error::Invalid("matrix dimensions must be positive"));
        }
        i32::try_from(rows).map_err(|_| CudaBf16Error::Invalid("matrix rows exceed i32"))?;
        i32::try_from(columns).map_err(|_| CudaBf16Error::Invalid("matrix columns exceed i32"))?;
        Self::bytes(rows, columns)?
            .checked_add(15)
            .ok_or(CudaBf16Error::Invalid("matrix alignment overflow"))?;
        Ok(Self {
            rows,
            columns,
            staging: DEVICE_ONLY_STAGING,
        })
    }

    fn staged(rows: usize, columns: usize) -> Result<Self, CudaBf16Error> {
        let mut layout = Self::device(rows, columns)?;
        layout.staging = Self::bytes(rows, columns)?;
        Ok(layout)
    }

    fn bytes(rows: usize, columns: usize) -> Result<usize, CudaBf16Error> {
        rows.checked_mul(columns)
            .and_then(|n| n.checked_mul(2))
            .ok_or(CudaBf16Error::Invalid("matrix byte extent overflow"))
    }
}

struct BufferLayout {
    shared: [MatrixLayout; Buffers::MATRIX_COUNT],
    local: [MatrixLayout; GeometryBuffers::MATRIX_COUNT],
    global: [MatrixLayout; GeometryBuffers::MATRIX_COUNT],
    local_width: usize,
    global_width: usize,
}

impl BufferLayout {
    fn new(config: &Gemma4Config, rows: usize) -> Result<Self, CudaBf16Error> {
        let local_width = AttentionWidth::Local.width(config);
        let global_width = AttentionWidth::Global.width(config);
        Ok(Self {
            shared: Buffers::layout(config, rows)?,
            local: GeometryBuffers::layout(config, rows, local_width)?,
            global: GeometryBuffers::layout(config, rows, global_width)?,
            local_width,
            global_width,
        })
    }

    fn matrices(&self) -> impl Iterator<Item = &MatrixLayout> {
        self.shared.iter().chain(&self.local).chain(&self.global)
    }
}

/// Install in the execution owner before calling `build`. Every completed matrix
/// stays here on allocation failure, until explicit cleanup succeeds. The factory
/// may use either a raw-stream allocator or an opaque domain without exposing it.
pub(in crate::cuda) struct BufferConstruction {
    matrices: Vec<CudaBf16Matrix>,
}

impl BufferConstruction {
    pub(in crate::cuda) fn new() -> Self {
        Self {
            matrices: Vec::new(),
        }
    }

    /// A failed build must be closed before starting another; it is not resumed.
    /// The complete layout and host metadata are prepared before device allocation.
    pub(in crate::cuda) fn build(
        &mut self,
        config: &Gemma4Config,
        rows: usize,
        mut allocate: impl FnMut(usize, usize, usize) -> Result<CudaBf16Matrix, CudaBf16Error>,
    ) -> Result<Buffers, CudaBf16Error> {
        if !self.matrices.is_empty() {
            return Err(CudaBf16Error::InUse);
        }
        let layout = BufferLayout::new(config, rows)?;
        self.matrices
            .try_reserve_exact(layout.matrices().count())
            .map_err(CudaBf16Error::Staging)?;
        for shape in layout.matrices() {
            // Reserved capacity makes this ownership handoff infallible. No
            // partial typed struct can drop earlier matrices on the next error.
            self.matrices
                .push(allocate(shape.rows, shape.columns, shape.staging)?);
        }
        let mut matrices = self.matrices.drain(..);
        Ok(Buffers::take(
            &mut matrices,
            layout.local_width,
            layout.global_width,
        ))
    }

    /// Idempotent; failed cleanup retains all matrix objects for another attempt.
    pub(in crate::cuda) fn close(&mut self) -> Result<(), CudaBf16Error> {
        for matrix in &mut self.matrices {
            matrix.close()?;
        }
        self.matrices.clear();
        Ok(())
    }
}

impl Buffers {
    pub(super) fn new(
        config: &Gemma4Config,
        rows: usize,
        stream: &Arc<CudaStream>,
        pool: &Arc<BytePool>,
    ) -> Result<Self, Gemma4Error> {
        let mut construction = BufferConstruction::new();
        match construction.build(config, rows, |rows, columns, staging| {
            CudaBf16Matrix::allocate(rows, columns, Arc::clone(stream), pool, staging)
        }) {
            Ok(buffers) => Ok(buffers),
            Err(error) => {
                let detail = match construction.close() {
                    Ok(()) => error.to_string(),
                    Err(cleanup) => format!(
                        "{error}; workspace cleanup remains unresolved: {cleanup}; \
                         legacy allocation Drop will retry or quarantine"
                    ),
                };
                Err(Gemma4Error::new("CUDA decoder", detail))
            }
        }
    }
}

#[derive(Clone, Copy)]
enum AttentionWidth {
    Local,
    Global,
}

impl AttentionWidth {
    fn width(self, config: &Gemma4Config) -> usize {
        let kind = match self {
            Self::Local => AttentionKind::Sliding,
            Self::Global => AttentionKind::Full,
        };
        config
            .layers()
            .iter()
            .find(|layer| layer.kind == kind)
            .map_or(0, |layer| layer.head_dim)
    }
}

impl DeviceCache {
    pub(in crate::cuda) fn close(&mut self) -> Result<(), CudaBf16Error> {
        self.keys.close()?;
        self.values.close()
    }
}

/// Installed in the model before any sequence allocation. The unfinished key
/// remains owned here if allocating its value fails; completed layers also stay
/// reachable until cleanup succeeds or the bundle transfers to a decoder.
pub(in crate::cuda) struct CacheConstruction {
    pub(in crate::cuda) caches: Vec<Option<DeviceCache>>,
    key: Option<CudaBf16Matrix>,
}
impl CacheConstruction {
    pub(in crate::cuda) fn new(layers: usize) -> Self {
        Self {
            caches: Vec::with_capacity(layers),
            key: None,
        }
    }

    fn build(
        &mut self,
        config: &Gemma4Config,
        capacity: usize,
        stream: &Arc<CudaStream>,
        pool: &Arc<BytePool>,
    ) -> Result<(), Gemma4Error> {
        self.build_with(config, capacity, |rows, columns, staging| {
            CudaBf16Matrix::allocate(rows, columns, Arc::clone(stream), pool, staging)
        })
        .map_err(|error| Gemma4Error::new("CUDA decoder", error.to_string()))
    }

    fn layout(
        config: &Gemma4Config,
        capacity: usize,
    ) -> Result<Vec<Option<MatrixLayout>>, CudaBf16Error> {
        let mut layout = Vec::new();
        layout
            .try_reserve_exact(config.layers().len())
            .map_err(CudaBf16Error::Staging)?;
        for (layer, geometry) in config.layers().iter().enumerate() {
            let shape = if geometry.kv_owner == layer {
                let columns = config
                    .num_key_value_heads()
                    .checked_mul(geometry.head_dim)
                    .ok_or(CudaBf16Error::Invalid("cache KV width overflow"))?;
                Some(MatrixLayout::device(capacity, columns)?)
            } else {
                None
            };
            layout.push(shape);
        }
        Ok(layout)
    }

    /// Build once; partial keys and completed layers remain owned on error.
    /// Cleanup does not reset this owner for a second build.
    pub(in crate::cuda) fn build_with(
        &mut self,
        config: &Gemma4Config,
        capacity: usize,
        mut allocate: impl FnMut(usize, usize, usize) -> Result<CudaBf16Matrix, CudaBf16Error>,
    ) -> Result<(), CudaBf16Error> {
        if self.key.is_some() || !self.caches.is_empty() {
            return Err(CudaBf16Error::InUse);
        }
        let layout = Self::layout(config, capacity)?;
        self.caches
            .try_reserve_exact(layout.len())
            .map_err(CudaBf16Error::Staging)?;
        for shape in layout {
            let Some(shape) = shape else {
                self.caches.push(None);
                continue;
            };
            self.key = Some(allocate(shape.rows, shape.columns, shape.staging)?);
            let values = allocate(shape.rows, shape.columns, shape.staging)?;
            let keys = self
                .key
                .take()
                .expect("allocated key stays owned until value exists");
            // Capacity was reserved before device allocation; this handoff does
            // not allocate or run fallible cleanup between key/value ownership.
            self.caches.push(Some(DeviceCache { keys, values }));
        }
        Ok(())
    }

    pub(in crate::cuda) fn close(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(key) = self.key.as_mut() {
            key.close()?;
        }
        for cache in self.caches.iter_mut().flatten() {
            cache.close()?;
        }
        Ok(())
    }
}

impl super::Gemma4CudaModel {
    pub(crate) fn has_unsettled_failure(&self) -> bool {
        self.failure.is_some() || self.cache_retirement.is_some()
    }

    pub(super) fn allocate_caches(&mut self) -> Result<Vec<Option<DeviceCache>>, Gemma4Error> {
        if self.failure.is_some() {
            return Err(Gemma4Error::new(
                "CUDA cache construction",
                "model is faulted",
            ));
        }
        if self.cache_retirement.is_some() {
            self.retire_failure()
                .map_err(|error| Gemma4Error::new("CUDA cache retirement", error.to_string()))?;
        }
        self.cache_retirement = Some(CacheConstruction::new(self.config().layers().len()));
        let result = {
            let Self {
                host,
                stream,
                pool,
                max_positions,
                cache_retirement,
                ..
            } = self;
            cache_retirement
                .as_mut()
                .expect("construction owner installed before allocation")
                .build(&host.config, *max_positions, stream, pool)
        };
        if let Err(error) = result {
            // Do not discard partial keys/layers even if their cleanup also fails.
            return match self.retire_failure() {
                Ok(()) => Err(error),
                Err(cleanup) => Err(Gemma4Error::new(
                    "CUDA cache construction",
                    format!("{error}; cleanup remains pending: {cleanup}"),
                )),
            };
        }
        Ok(self
            .cache_retirement
            .take()
            .expect("successful cache construction")
            .caches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Gemma4Config {
        Gemma4Config::from_json(
            &serde_json::from_str(include_str!(
                "../../../../../benchmarks/gemma4-reference/config.json"
            ))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn qualified_workspaces_preserve_every_shape_and_staging_extent() {
        let config = config();
        for rows in [1, 8, 16] {
            let layout = BufferLayout::new(&config, rows).unwrap();
            assert_eq!((layout.local_width, layout.global_width), (256, 512));
            // Independent scalar/8/16 layout, in the original field order.
            let mut expected = vec![
                (rows, 2560, rows * 2560 * 2),   // hidden
                (rows, 2560, 1),                 // hidden_next
                (1, 2560, 1),                    // final_input
                (1, 2560, 1),                    // final_norm
                (1, 262_144, 262_144 * 2),       // logits
                (rows, 10752, 1),                // ple_projection
                (rows, 10752, 1),                // ple_projected_scaled
                (rows, 10752, rows * 10752 * 2), // ple_normed
                (rows, 256, rows * 256 * 2),     // ple_context
            ];
            for (width, q, kv) in [(256, 2048, 512), (512, 4096, 1024)] {
                expected.extend(
                    [
                        width, width, // cos, sin
                        2560,  // normed
                        q, q, q, // queries, query_norm, query_rotary
                        kv, kv, kv, kv, kv, // keys, key_norm, key_rotary, values, value_norm
                        q,  // attended
                        2560, 2560, 2560, // projected, post_attention, ffn_norm
                        10240, 10240, 10240, 10240, // gate, up, activated, product
                        2560, 2560, // down, post_feedforward
                        256, 256, 256, // ple_gate, ple_activated, ple_product
                        2560, 2560, // ple_projected, post_ple
                    ]
                    .map(|columns| (rows, columns, 1)),
                );
            }
            let actual: Vec<_> = layout
                .matrices()
                .map(|shape| (shape.rows, shape.columns, shape.staging))
                .collect();
            assert_eq!(actual.len(), 61);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn invalid_late_geometry_and_extents_never_reach_the_factory() {
        let valid = config();
        let mut cases = vec![
            (valid.clone(), 0),
            (valid.clone(), i32::MAX as usize + 1),
            (valid.clone(), usize::MAX),
        ];
        let mut invalid = valid.clone();
        invalid.intermediate = i32::MAX as usize + 1;
        cases.push((invalid, 16));
        let mut invalid = valid.clone();
        invalid.heads = usize::MAX;
        cases.push((invalid, 8));
        let mut invalid = valid.clone();
        invalid.kv_heads = usize::MAX;
        cases.push((invalid, 8));
        let mut invalid = valid.clone();
        invalid.packed_ple = usize::MAX;
        cases.push((invalid, 16));
        for width in [0, i32::MAX as usize + 1] {
            let mut invalid = valid.clone();
            for layer in &mut invalid.layers {
                if layer.kind == AttentionKind::Full {
                    layer.head_dim = width;
                }
            }
            cases.push((invalid, 16));
        }
        // Mutations deliberately bypass config validation to test this allocation
        // boundary, especially the global layout visited after all local fields.
        for (config, rows) in cases {
            let mut construction = BufferConstruction::new();
            let mut calls = 0;
            assert!(matches!(
                construction.build(&config, rows, |_, _, _| {
                    calls += 1;
                    Err(CudaBf16Error::Invalid("unexpected allocation"))
                }),
                Err(CudaBf16Error::Invalid(_))
            ));
            assert_eq!(calls, 0);
            assert!(construction.matrices.is_empty());
            assert_eq!(construction.matrices.capacity(), 0);
        }
    }

    #[test]
    fn cache_layout_preserves_owner_slots_and_device_only_staging() {
        let config = config();
        for capacity in [1, 8, 32] {
            let layout = CacheConstruction::layout(&config, capacity).unwrap();
            let actual: Vec<_> = layout
                .iter()
                .map(|shape| {
                    shape
                        .as_ref()
                        .map(|shape| (shape.rows, shape.columns, shape.staging))
                })
                .collect();
            let expected: Vec<_> = (0..42)
                .map(|layer| {
                    (layer < 24).then_some((capacity, if layer % 6 == 5 { 1024 } else { 512 }, 1))
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn cache_preflight_checks_later_owners_before_any_allocation() {
        let valid = config();
        let mut cases = vec![
            (valid.clone(), 0),
            (valid.clone(), i32::MAX as usize + 1),
            (valid.clone(), usize::MAX),
        ];
        for width in [0, i32::MAX as usize, usize::MAX] {
            let mut invalid = valid.clone();
            invalid.layers[23].head_dim = width;
            cases.push((invalid, 32));
        }
        for (config, capacity) in cases {
            let mut construction = CacheConstruction::new(config.layers().len());
            let mut calls = 0;
            assert!(matches!(
                construction.build_with(&config, capacity, |_, _, _| {
                    calls += 1;
                    Err(CudaBf16Error::Invalid("unexpected allocation"))
                }),
                Err(CudaBf16Error::Invalid(_))
            ));
            assert_eq!(calls, 0);
            assert!(construction.key.is_none());
            assert!(construction.caches.is_empty());
        }
    }

    #[test]
    fn cache_factory_keeps_typed_reservation_refusal_and_reserves_all_slots() {
        use ribn_foundation::ReserveError;

        let config = config();
        // The constructor's capacity hint cannot permit a fallible Vec growth
        // between installing a key and handing off a completed key/value pair.
        let mut construction = CacheConstruction::new(0);
        let mut calls = Vec::new();
        let result = construction.build_with(&config, 32, |rows, columns, staging| {
            calls.push((rows, columns, staging));
            Err(CudaBf16Error::Reserve(ReserveError::TooLarge {
                requested: 32768,
                capacity: 0,
            }))
        });
        assert!(matches!(
            result,
            Err(CudaBf16Error::Reserve(ReserveError::TooLarge {
                requested: 32768,
                capacity: 0,
            }))
        ));
        assert_eq!(calls, [(32, 512, 1)]);
        assert!(construction.caches.capacity() >= config.layers().len());
        assert!(construction.caches.is_empty());
        assert!(construction.key.is_none());
        construction.close().unwrap();
        construction.close().unwrap();
    }

    #[test]
    fn factory_refusal_preserves_error_and_allows_cleanup_then_retry() {
        let config = config();
        let mut construction = BufferConstruction::new();
        for _ in 0..2 {
            let mut calls = Vec::new();
            assert!(matches!(
                construction.build(&config, 8, |rows, columns, staging| {
                    calls.push((rows, columns, staging));
                    Err(CudaBf16Error::Invalid("injected allocation refusal"))
                }),
                Err(CudaBf16Error::Invalid("injected allocation refusal"))
            ));
            assert_eq!(calls, [(8, 2560, 8 * 2560 * 2)]);
            assert!(construction.matrices.is_empty());
            assert!(construction.matrices.capacity() >= 61);
            construction.close().unwrap();
            construction.close().unwrap();
        }
    }
}
