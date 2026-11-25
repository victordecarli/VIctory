//! The bytes of a [`HostShader`]: what `victory-gpu`'s shader cache stores.
//!
//! Little-endian words and length-prefixed lists, behind a magic. The
//! format is ours and changes freely: the cache keys every entry by the
//! build that wrote it. Writing takes every struct apart without `..`, so a
//! field added to [`HostShader`], [`UserData`] or [`StaticSampler`] does not
//! compile until it is written here, and reading builds the value whole.

use super::{HostShader, StaticSampler, UserData};
use crate::native::{HwStage, RootType};

const MAGIC: &[u8; 4] = b"VSH2";

fn stage_code(stage: HwStage) -> u32 {
    match stage {
        HwStage::Ls => 0,
        HwStage::Hs => 1,
        HwStage::Es => 2,
        HwStage::Gs => 3,
        HwStage::Vs => 4,
        HwStage::Ps => 5,
        HwStage::Ms => 6,
        HwStage::Cs => 7,
    }
}

fn stage(code: u32) -> Option<HwStage> {
    Some(match code {
        0 => HwStage::Ls,
        1 => HwStage::Hs,
        2 => HwStage::Es,
        3 => HwStage::Gs,
        4 => HwStage::Vs,
        5 => HwStage::Ps,
        6 => HwStage::Ms,
        7 => HwStage::Cs,
        _ => return None,
    })
}

struct Writer(Vec<u8>);

impl Writer {
    fn word(&mut self, v: u32) {
        self.0.extend(v.to_le_bytes());
    }
    fn list(&mut self, bytes: &[u8]) {
        self.word(bytes.len() as u32);
        self.0.extend(bytes);
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }
    fn word(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn list(&mut self) -> Option<Vec<u8>> {
        let n = self.word()? as usize;
        Some(self.take(n)?.to_vec())
    }
}

fn write_user_data(w: &mut Writer, u: &UserData) {
    let UserData { index, slot, root, dword } = *u;
    for v in [index, u32::from(slot), root.to_raw(), dword] {
        w.word(v);
    }
}

fn read_user_data(r: &mut Reader) -> Option<UserData> {
    Some(UserData {
        index: r.word()?,
        slot: u8::try_from(r.word()?).ok()?,
        root: RootType::from_raw(r.word()?),
        dword: r.word()?,
    })
}

fn write_sampler(w: &mut Writer, s: &StaticSampler) {
    let StaticSampler {
        filter,
        address: [u, v, a],
        mip_lod_bias,
        max_anisotropy,
        comparison,
        border,
        min_lod,
        max_lod,
    } = *s;
    for x in [
        filter,
        u,
        v,
        a,
        mip_lod_bias.to_bits(),
        max_anisotropy,
        comparison,
        border,
        min_lod.to_bits(),
        max_lod.to_bits(),
    ] {
        w.word(x);
    }
}

fn read_sampler(r: &mut Reader) -> Option<StaticSampler> {
    Some(StaticSampler {
        filter: r.word()?,
        address: [r.word()?, r.word()?, r.word()?],
        mip_lod_bias: f32::from_bits(r.word()?),
        max_anisotropy: r.word()?,
        comparison: r.word()?,
        border: r.word()?,
        min_lod: f32::from_bits(r.word()?),
        max_lod: f32::from_bits(r.word()?),
    })
}

/// `n` items read by `item`, with no allocation the input cannot back.
fn read_items<T>(r: &mut Reader, item: fn(&mut Reader) -> Option<T>) -> Option<Vec<T>> {
    let n = r.word()? as usize;
    let mut items = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        items.push(item(r)?);
    }
    Some(items)
}

impl HostShader {
    /// The shader as bytes that [`HostShader::from_bytes`] reads back.
    pub fn to_bytes(&self) -> Vec<u8> {
        let HostShader {
            stage,
            dxil,
            user_data,
            samplers,
            static_samplers,
            root_cbvs,
            root_srvs,
            wave_size,
        } = self;
        let mut w = Writer(MAGIC.to_vec());
        w.word(stage_code(*stage));
        w.list(dxil);
        w.word(user_data.len() as u32);
        for u in user_data {
            write_user_data(&mut w, u);
        }
        w.word(samplers.len() as u32);
        for &words in samplers {
            for v in words {
                w.word(v);
            }
        }
        w.word(static_samplers.len() as u32);
        for s in static_samplers {
            write_sampler(&mut w, s);
        }
        w.list(root_cbvs);
        w.list(root_srvs);
        // 0: no wave size required (`WaveSize` is never 0).
        w.word(wave_size.unwrap_or(0));
        w.0
    }

    /// The shader [`HostShader::to_bytes`] wrote, or `None` for bytes of
    /// another format, cut short or with anything left over.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader(bytes);
        if r.take(4)? != MAGIC {
            return None;
        }
        let stage = stage(r.word()?)?;
        let dxil = r.list()?;
        let user_data = read_items(&mut r, read_user_data)?;
        let samplers = read_items(&mut r, |r| Some([r.word()?, r.word()?, r.word()?, r.word()?]))?;
        let static_samplers = read_items(&mut r, read_sampler)?;
        let root_cbvs = r.list()?;
        let root_srvs = r.list()?;
        let wave_size = match r.word()? {
            0 => None,
            n => Some(n),
        };
        r.0.is_empty().then_some(HostShader {
            stage,
            dxil,
            user_data,
            samplers,
            static_samplers,
            root_cbvs,
            root_srvs,
            wave_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::emit::{HostShader, StaticSampler, UserData};
    use crate::native::{HwStage, RootType};

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData { index: 8, slot: 1, root: RootType::Table, dword: 0 }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([0x400, 0xc0_0000, 0x18f0_0000, 0])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            wave_size: Some(64),
        }
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = shader().to_bytes();
        assert_eq!(HostShader::from_bytes(&bytes), Some(shader()));
        assert_eq!(HostShader::from_bytes(&bytes[..bytes.len() - 1]), None);
        assert_eq!(HostShader::from_bytes(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(HostShader::from_bytes(b"VSH1"), None);
        let older = [b"VSH1".as_slice(), &bytes[4..]].concat();
        assert_eq!(HostShader::from_bytes(&older), None, "another format's entry");
    }

    /// The wave size a shader requires comes back, absent or not; an entry
    /// without it reads as damaged.
    #[test]
    fn entries_keep_the_wave_size() {
        let none = HostShader { wave_size: None, ..shader() };
        assert_eq!(HostShader::from_bytes(&none.to_bytes()), Some(none));
        let bytes = shader().to_bytes();
        assert_eq!(HostShader::from_bytes(&bytes[..bytes.len() - 4]), None);
    }

    #[test]
    fn every_field_comes_back() {
        let sampler = StaticSampler {
            filter: 0x55,
            address: [1, 2, 3],
            mip_lod_bias: -1.5,
            max_anisotropy: 16,
            comparison: 4,
            border: 2,
            min_lod: 0.25,
            max_lod: 13.75,
        };
        let s = HostShader {
            stage: HwStage::Cs,
            dxil: (0..=255).collect(),
            user_data: vec![
                UserData { index: 0, slot: 0, root: RootType::Constants, dword: 3 },
                UserData { index: 70, slot: 255, root: RootType::Table, dword: 1 },
            ],
            samplers: vec![[1, 2, 3, 4], [u32::MAX, 0, 7, 9]],
            static_samplers: vec![sampler, StaticSampler { min_lod: 1.0, ..sampler }],
            root_cbvs: vec![0, 7, 63],
            root_srvs: vec![5],
            wave_size: Some(32),
        };
        assert_eq!(HostShader::from_bytes(&s.to_bytes()), Some(s));
    }

    #[test]
    fn every_root_type_comes_back() {
        for root in [
            RootType::Constants,
            RootType::Descriptor,
            RootType::Table,
            RootType::ImmediateSampler,
            RootType::ResourceHeap,
            RootType::SamplerHeap,
            RootType::Other(2),
            RootType::Other(u32::MAX),
        ] {
            let s = HostShader {
                user_data: vec![UserData { index: 1, slot: 2, root, dword: 0 }],
                ..shader()
            };
            assert_eq!(HostShader::from_bytes(&s.to_bytes()), Some(s), "{root:?}");
        }
    }

    #[test]
    fn every_stage_comes_back() {
        for stage in [
            HwStage::Ls,
            HwStage::Hs,
            HwStage::Es,
            HwStage::Gs,
            HwStage::Vs,
            HwStage::Ps,
            HwStage::Ms,
            HwStage::Cs,
        ] {
            let s = HostShader { stage, ..shader() };
            assert_eq!(HostShader::from_bytes(&s.to_bytes()), Some(s), "{stage:?}");
        }
    }
}
