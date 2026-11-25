//! A [`HostRootLayout`] as a D3D12 root signature: the only translation and
//! the only creation. Every choice (types, registers, visibility,
//! volatility, flags) is the layout's; this copies them into D3D12 structures
//! and makes no decision of its own.

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;

use crate::emit::{
    HostRootLayout, HostRootParam, HostSampler, ParamKind, RootFlags, RootView, Visibility,
};

/// The D3D12 description of a layout, owning the arrays it points into.
struct RootDesc {
    params: Vec<D3D12_ROOT_PARAMETER1>,
    samplers: Vec<D3D12_STATIC_SAMPLER_DESC>,
    flags: D3D12_ROOT_SIGNATURE_FLAGS,
}

fn visibility(v: Visibility) -> D3D12_SHADER_VISIBILITY {
    match v {
        Visibility::All => D3D12_SHADER_VISIBILITY_ALL,
        Visibility::Vertex => D3D12_SHADER_VISIBILITY_VERTEX,
        Visibility::Pixel => D3D12_SHADER_VISIBILITY_PIXEL,
        Visibility::Mesh => D3D12_SHADER_VISIBILITY_MESH,
    }
}

fn param(p: &HostRootParam) -> D3D12_ROOT_PARAMETER1 {
    let (kind, anonymous) = match p.kind {
        ParamKind::Constants { dwords } => (
            D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: p.register,
                    RegisterSpace: p.space,
                    Num32BitValues: dwords,
                },
            },
        ),
        ParamKind::View { view, volatile } => (
            match view {
                RootView::Cbv => D3D12_ROOT_PARAMETER_TYPE_CBV,
                RootView::Srv => D3D12_ROOT_PARAMETER_TYPE_SRV,
            },
            D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: p.register,
                    RegisterSpace: p.space,
                    Flags: if volatile {
                        D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE
                    } else {
                        D3D12_ROOT_DESCRIPTOR_FLAG_NONE
                    },
                },
            },
        ),
    };
    D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: anonymous,
        ShaderVisibility: visibility(p.visibility),
    }
}

fn sampler(s: &HostSampler) -> D3D12_STATIC_SAMPLER_DESC {
    let d = &s.sampler;
    let [u, v, w] = d.address.map(|a| D3D12_TEXTURE_ADDRESS_MODE(a.cast_signed()));
    D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER(d.filter.cast_signed()),
        AddressU: u,
        AddressV: v,
        AddressW: w,
        MipLODBias: d.mip_lod_bias,
        MaxAnisotropy: d.max_anisotropy,
        ComparisonFunc: D3D12_COMPARISON_FUNC(d.comparison.cast_signed()),
        BorderColor: D3D12_STATIC_BORDER_COLOR(d.border.cast_signed()),
        MinLOD: d.min_lod,
        MaxLOD: d.max_lod,
        ShaderRegister: s.register,
        RegisterSpace: s.space,
        ShaderVisibility: visibility(s.visibility),
    }
}

fn flags(f: RootFlags) -> D3D12_ROOT_SIGNATURE_FLAGS {
    let RootFlags { input_assembler, resource_heap_indexed, sampler_heap_indexed } = f;
    let mut flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    if input_assembler {
        flags |= D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    }
    if resource_heap_indexed {
        flags |= D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
    }
    if sampler_heap_indexed {
        flags |= D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    }
    flags
}

/// The single translation, one to one.
fn describe(layout: &HostRootLayout) -> RootDesc {
    RootDesc {
        params: layout.params().iter().map(param).collect(),
        samplers: layout.static_samplers().iter().map(sampler).collect(),
        flags: flags(layout.flags()),
    }
}

/// The serialized (version 1.1) root signature of a layout; `Err` carries
/// D3D12's own text.
fn serialize(layout: &HostRootLayout) -> Result<Vec<u8>, String> {
    let root = describe(layout);
    let counts = u32::try_from(root.params.len())
        .and_then(|p| u32::try_from(root.samplers.len()).map(|s| (p, s)))
        .map_err(|e| e.to_string())?;
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: counts.0,
                pParameters: root.params.as_ptr(),
                NumStaticSamplers: counts.1,
                pStaticSamplers: root.samplers.as_ptr(),
                Flags: root.flags,
            },
        },
    };
    let mut blob: Option<ID3DBlob> = None;
    let mut error: Option<ID3DBlob> = None;
    // SAFETY: the description points at arrays `root` owns for the whole
    // call; each blob is read in full before it is released.
    unsafe {
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            return Err(match error {
                Some(error) => String::from_utf8_lossy(blob_bytes(&error)).into_owned(),
                None => e.message().to_string(),
            });
        }
        let blob = blob.ok_or("no root signature blob")?;
        Ok(blob_bytes(&blob).to_vec())
    }
}

/// The bytes of a blob.
///
/// # Safety
/// The blob is not released or written while the slice is in use.
unsafe fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the pointer and size describe the blob's buffer, which lives
    // as long as the borrow.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

/// The root signature of `layout` and its serialized bytes (the pipeline
/// library keys on them). `Err` carries D3D12's text.
pub fn root_signature(
    device: &ID3D12Device,
    layout: &HostRootLayout,
) -> Result<(ID3D12RootSignature, Vec<u8>), String> {
    let bytes = serialize(layout)?;
    // SAFETY: `bytes` is a serialized root signature.
    let root =
        unsafe { device.CreateRootSignature(0, &bytes) }.map_err(|e| e.message().to_string())?;
    Ok((root, bytes))
}

#[cfg(test)]
#[cfg(windows)]
mod tests {
    use super::*;
    use crate::emit::{GraphicsStage, HostShader, StaticSampler};
    use crate::native::HwStage;
    use windows::core::Interface;

    fn shader(stage: HwStage) -> HostShader {
        HostShader {
            stage,
            dxil: Vec::new(),
            user_data: vec![crate::emit::UserData {
                index: 5,
                slot: 1,
                root: crate::native::RootType::Table,
                dword: 0,
            }],
            samplers: Vec::new(),
            static_samplers: vec![StaticSampler::from_words([
                0x1_2345,
                0x00ff_f000,
                0,
                0xc000_0000,
            ])],
            root_cbvs: vec![7, 2],
            root_srvs: vec![4],
            used: Vec::new(),
            wave_size: None,
        }
    }

    /// What the deserializer reads back from `bytes`: the 1.1 description.
    fn read_back(bytes: &[u8], check: impl FnOnce(&D3D12_ROOT_SIGNATURE_DESC1)) {
        // SAFETY: `bytes` is a serialized root signature; the description
        // belongs to the deserializer, which outlives the check.
        unsafe {
            let mut raw = std::ptr::null_mut();
            D3D12CreateVersionedRootSignatureDeserializer(
                bytes.as_ptr().cast(),
                bytes.len(),
                &ID3D12VersionedRootSignatureDeserializer::IID,
                &mut raw,
            )
            .unwrap();
            let de = ID3D12VersionedRootSignatureDeserializer::from_raw(raw);
            let versioned =
                de.GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1).unwrap();
            let versioned = &*versioned;
            assert_eq!(versioned.Version, D3D_ROOT_SIGNATURE_VERSION_1_1);
            check(&versioned.Anonymous.Desc_1_1);
        }
    }

    fn check_params(
        desc: &D3D12_ROOT_SIGNATURE_DESC1,
        expect: &[(D3D12_ROOT_PARAMETER_TYPE, D3D12_SHADER_VISIBILITY, u32)],
    ) {
        assert_eq!(desc.NumParameters as usize, expect.len());
        // SAFETY: the deserializer holds `NumParameters` parameters.
        let params = unsafe { std::slice::from_raw_parts(desc.pParameters, expect.len()) };
        for (p, &(kind, vis, slot)) in params.iter().zip(expect) {
            assert_eq!(p.ParameterType, kind);
            assert_eq!(p.ShaderVisibility, vis);
            // SAFETY: the union member read is the one `ParameterType` names.
            unsafe {
                if kind == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS {
                    let c = p.Anonymous.Constants;
                    assert_eq!((c.ShaderRegister, c.RegisterSpace, c.Num32BitValues), (0, 0, slot));
                } else {
                    let d = p.Anonymous.Descriptor;
                    assert_eq!(
                        (d.ShaderRegister, d.RegisterSpace, d.Flags),
                        (slot, 1, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE)
                    );
                }
            }
        }
    }

    /// The expectation is what the game's own signature held before the
    /// layout: constants `(0, 0, n)`, views `(slot, 1, DATA_VOLATILE)`,
    /// samplers in space 0 per stage, both heaps indexed directly.
    #[test]
    fn layouts_serialize_as_production_root_signatures() {
        let (vs, ps, cs) = (shader(HwStage::Vs), shader(HwStage::Ps), shader(HwStage::Cs));
        let graphics =
            HostRootLayout::graphics(&[(&vs, GraphicsStage::Vertex), (&ps, GraphicsStage::Pixel)]);
        let bytes = serialize(&graphics).unwrap();
        read_back(&bytes, |desc| {
            let per_stage = |vis| {
                [
                    (D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, vis, 8),
                    (D3D12_ROOT_PARAMETER_TYPE_CBV, vis, 7),
                    (D3D12_ROOT_PARAMETER_TYPE_CBV, vis, 2),
                    (D3D12_ROOT_PARAMETER_TYPE_SRV, vis, 4),
                ]
            };
            let mut expect = per_stage(D3D12_SHADER_VISIBILITY_VERTEX).to_vec();
            expect.extend(per_stage(D3D12_SHADER_VISIBILITY_PIXEL));
            check_params(desc, &expect);
            assert_eq!(desc.NumStaticSamplers, 2);
            // SAFETY: two static samplers.
            let samplers = unsafe { std::slice::from_raw_parts(desc.pStaticSamplers, 2) };
            let want = StaticSampler::from_words([0x1_2345, 0x00ff_f000, 0, 0xc000_0000]);
            for (s, vis) in
                samplers.iter().zip([D3D12_SHADER_VISIBILITY_VERTEX, D3D12_SHADER_VISIBILITY_PIXEL])
            {
                assert_eq!(s.Filter.0, want.filter.cast_signed());
                assert_eq!(
                    [s.AddressU.0, s.AddressV.0, s.AddressW.0],
                    want.address.map(u32::cast_signed)
                );
                assert_eq!(s.MipLODBias, want.mip_lod_bias);
                assert_eq!(s.MaxAnisotropy, want.max_anisotropy);
                assert_eq!(s.ComparisonFunc.0, want.comparison.cast_signed());
                assert_eq!(s.BorderColor.0, want.border.cast_signed());
                assert_eq!((s.MinLOD, s.MaxLOD), (want.min_lod, want.max_lod));
                assert_eq!((s.ShaderRegister, s.RegisterSpace), (0, 0));
                assert_eq!(s.ShaderVisibility, vis);
            }
            assert_eq!(
                desc.Flags,
                D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED
            );
        });

        let compute = serialize(&HostRootLayout::compute(&cs)).unwrap();
        read_back(&compute, |desc| {
            let all = D3D12_SHADER_VISIBILITY_ALL;
            check_params(
                desc,
                &[
                    (D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, all, 8),
                    (D3D12_ROOT_PARAMETER_TYPE_CBV, all, 7),
                    (D3D12_ROOT_PARAMETER_TYPE_CBV, all, 2),
                    (D3D12_ROOT_PARAMETER_TYPE_SRV, all, 4),
                ],
            );
            assert_eq!(desc.NumStaticSamplers, 1);
            assert_eq!(
                desc.Flags,
                D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED
            );
        });
    }
}
