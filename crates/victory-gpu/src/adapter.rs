//! The host adapter the device runs on. DXGI's default adapter can be the
//! integrated GPU on a machine that also has a discrete one, so the device
//! goes on the first hardware adapter in DXGI's high-performance order
//! (external, then discrete, then integrated GPUs) that makes a feature
//! level 12_0 device, and the adapter and its wave lanes are logged once at
//! start-up.

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1, D3D12CreateDevice,
    ID3D12Device,
};
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

/// A device on the preferred hardware adapter; on DXGI's default adapter
/// when no hardware adapter makes one or the factory cannot order them
/// (before Windows 10 1803, or a host with only the software adapter,
/// such as a CI runner), as before this choice existed.
pub(crate) fn create_device() -> windows::core::Result<ID3D12Device> {
    let device = match preferred() {
        Ok(Some(device)) => device,
        Ok(None) => {
            tracing::warn!("no hardware adapter makes a 12_0 device; using DXGI's default adapter");
            default_device()?
        }
        Err(e) => {
            tracing::warn!(error = %e, "adapters not ordered by preference; using DXGI's default adapter");
            default_device()?
        }
    };
    log_adapter(&device);
    log_wave_lanes(&device);
    Ok(device)
}

/// The first adapter, in `adapters`' order, that is not software and
/// that `create` makes a device on.
fn first_hardware<A, D>(
    adapters: impl IntoIterator<Item = (A, bool)>,
    mut create: impl FnMut(&A) -> Option<D>,
) -> Option<(A, D)> {
    adapters
        .into_iter()
        .filter(|(_, software)| !software)
        .find_map(|(adapter, _)| create(&adapter).map(|device| (adapter, device)))
}

/// A device on the first hardware adapter in high-performance order.
fn preferred() -> windows::core::Result<Option<ID3D12Device>> {
    // SAFETY: plain DXGI factory creation.
    let factory: IDXGIFactory6 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))? };
    let adapters = (0..).map_while(|i| {
        // SAFETY: a plain enumeration; DXGI_ERROR_NOT_FOUND ends the list.
        match unsafe {
            factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                i,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            )
        } {
            Ok(adapter) => Some(adapter),
            Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => None,
            Err(e) => {
                tracing::warn!(index = i, error = %e, "adapter enumeration stopped");
                None
            }
        }
    });
    let described = adapters.filter_map(|adapter| {
        // SAFETY: a plain query on a live adapter.
        match unsafe { adapter.GetDesc1() } {
            Ok(desc) => {
                let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0.cast_unsigned() != 0;
                Some((adapter, software))
            }
            Err(e) => {
                tracing::warn!(error = %e, "adapter without a description skipped");
                None
            }
        }
    });
    let chosen = first_hardware(described, |adapter| {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: device creation on a live hardware adapter; the runtime
        // writes `device`.
        match unsafe { D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, &mut device) } {
            Ok(()) => device,
            Err(e) => {
                tracing::info!(error = %e, "adapter makes no 12_0 device; trying the next");
                None
            }
        }
    });
    Ok(chosen.map(|(_, device)| device))
}

fn default_device() -> windows::core::Result<ID3D12Device> {
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: device creation on DXGI's default adapter; the runtime writes
    // `device`.
    unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)? };
    device.ok_or_else(windows::core::Error::empty)
}

/// Logs the adapter `device` runs on, found by its LUID so the default
/// adapter is named too.
fn log_adapter(device: &ID3D12Device) {
    match describe(device) {
        Ok((desc, driver)) => {
            let len =
                desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..len]);
            let driver = match driver {
                Ok(v) => driver_version(v),
                Err(e) => format!("unknown ({e})"),
            };
            tracing::info!(
                adapter = %name,
                vendor = format_args!("{:#06x}", desc.VendorId),
                dedicated_mb = desc.DedicatedVideoMemory >> 20,
                driver = %driver,
                "host device adapter"
            );
        }
        Err(e) => tracing::warn!(error = %e, "the host device's adapter could not be described"),
    }
}

/// Logs the lanes a wave of `device` has. A console wave64 program that
/// needs its 64 lanes declares `WaveSize(64)`, which a host whose maximum is
/// 32 (NVIDIA) refuses at pipeline creation.
fn log_wave_lanes(device: &ID3D12Device) {
    let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the query writes the structure of the given size.
    let queried = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut o1).cast(),
            size_of_val(&o1) as u32,
        )
    };
    match queried {
        Ok(()) => tracing::info!(
            min = o1.WaveLaneCountMin,
            max = o1.WaveLaneCountMax,
            wave_ops = o1.WaveOps.as_bool(),
            "host wave lanes"
        ),
        Err(e) => tracing::warn!(error = %e, "the host's wave lanes could not be queried"),
    }
}

type Described = (DXGI_ADAPTER_DESC1, windows::core::Result<i64>);

fn describe(device: &ID3D12Device) -> windows::core::Result<Described> {
    // SAFETY: plain DXGI queries on live objects.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(device.GetAdapterLuid())?;
        let desc = adapter.GetDesc1()?;
        // The user-mode driver version. Checked on one host only: an RTX
        // 5070 reported 32.0.16.2124, the driver version Windows lists for
        // it (Win32_VideoController.DriverVersion).
        let driver = adapter.CheckInterfaceSupport(&IDXGIDevice::IID);
        Ok((desc, driver))
    }
}

/// A user-mode driver version as Windows shows it: four 16-bit parts, the
/// most significant first.
fn driver_version(v: i64) -> String {
    let v = v.cast_unsigned();
    format!("{}.{}.{}.{}", v >> 48, (v >> 32) & 0xffff, (v >> 16) & 0xffff, v & 0xffff)
}

#[cfg(test)]
mod tests;
