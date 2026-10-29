//! Host-only lookup materialization for the CUDA decoder.
//!
//! Projections and norms consume device weights. Retaining a complete CPU
//! inventory here would duplicate every one of those unused host parameters.

use crate::{CpuParameter, Gemma4Config, Gemma4Error, Gemma4Source, LayerParameter, ParameterId};

pub(super) struct HostLookups {
    pub(super) config: Gemma4Config,
    pub(super) token_embedding: CpuParameter,
    pub(super) ple_embedding: CpuParameter,
    pub(super) layer_scalars: Vec<u16>,
}

impl HostLookups {
    pub(super) fn load(source: &Gemma4Source) -> Result<Self, Gemma4Error> {
        let token_embedding = read(source, ParameterId::TokenEmbedding)?;
        let ple_embedding = read(source, ParameterId::PleEmbedding)?;
        let layer_scalars = (0..source.config().layers().len())
            .map(|layer| {
                read(
                    source,
                    ParameterId::Layer {
                        layer,
                        parameter: LayerParameter::LayerScalar,
                    },
                )
                .map(|scalar| scalar.values()[0])
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            config: source.config().clone(),
            token_embedding,
            ple_embedding,
            layer_scalars,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        (self.token_embedding.values().len()
            + self.ple_embedding.values().len()
            + self.layer_scalars.len())
            * 2
    }
}

fn read(source: &Gemma4Source, id: ParameterId) -> Result<CpuParameter, Gemma4Error> {
    // Whole-inventory preflight has already validated every role, shape and dtype.
    // Read the retained source handle, never a newly resolved package path.
    let parameter = source
        .parameter(id)
        .ok_or_else(|| Gemma4Error::new("CUDA host lookup", format!("missing parameter {id:?}")))?;
    CpuParameter::read(&parameter.spec().shape, parameter.reader())
}

#[cfg(test)]
mod tests {
    use super::HostLookups;
    use crate::{
        Gemma4CpuWeights, Gemma4Source, LayerParameter, ParameterId, weight_fixtures as fixtures,
    };

    #[test]
    fn only_lookup_payload_is_retained_and_matches_the_full_cpu_inventory() {
        for sharded in [false, true] {
            let dir = fixtures::Dir::new();
            let package = fixtures::package(&dir, &fixtures::inventory(), sharded);
            let source = Gemma4Source::open(&package).unwrap();
            let full = Gemma4CpuWeights::load(&source).unwrap();
            let names = if sharded {
                vec!["part-0.safetensors", "part-1.safetensors"]
            } else {
                vec!["model.safetensors"]
            };
            for name in names {
                std::fs::rename(dir.0.join(name), dir.0.join(format!("selected-{name}"))).unwrap();
                std::fs::write(dir.0.join(name), b"not the selected source").unwrap();
            }
            let lookups = HostLookups::load(&source).unwrap();
            assert_eq!(&lookups.config, full.config());
            for (actual, id) in [
                (&lookups.token_embedding, ParameterId::TokenEmbedding),
                (&lookups.ple_embedding, ParameterId::PleEmbedding),
            ] {
                let expected = full.parameter(id).unwrap();
                assert_eq!(actual.shape(), expected.shape());
                assert_eq!(actual.values(), expected.values());
            }
            assert_eq!(lookups.layer_scalars.len(), 4);
            for (layer, &bits) in lookups.layer_scalars.iter().enumerate() {
                assert_eq!(
                    bits,
                    full.layer(layer, LayerParameter::LayerScalar)
                        .unwrap()
                        .values()[0]
                );
            }
            // Independent fixture dimensions: 7x7 token and 9x8 PLE tables,
            // plus four BF16 scalars. No projection/norm parameter is retained.
            assert_eq!(lookups.bytes(), 250);
            assert!(full.bytes() > u64::try_from(lookups.bytes()).unwrap());
        }
    }
}
