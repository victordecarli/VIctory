use super::*;

/// Synthetic adapters in DXGI's preference order: (name, software).
const ADAPTERS: [(&str, bool); 4] =
    [("software", true), ("refuses", false), ("discrete", false), ("integrated", false)];

#[test]
fn the_first_hardware_adapter_that_makes_a_device_is_chosen() {
    let mut tried = Vec::new();
    let chosen = first_hardware(ADAPTERS, |name| {
        tried.push(*name);
        (*name != "refuses").then_some("device")
    });
    assert_eq!(chosen, Some(("discrete", "device")));
    // The software adapter is never tried; nothing after the choice is.
    assert_eq!(tried, ["refuses", "discrete"]);
}

#[test]
fn no_adapter_is_chosen_when_none_makes_a_device() {
    assert_eq!(first_hardware(ADAPTERS, |_| None::<()>), None);
    let software_only = [("software", true)];
    assert_eq!(first_hardware(software_only, |_| Some(())), None);
    assert_eq!(first_hardware([] as [(&str, bool); 0], |_| Some(())), None);
}

/// The user-mode driver version packs four 16-bit parts, the most
/// significant first, as Windows shows it (e.g. 32.0.16.2124).
#[test]
fn the_driver_version_reads_as_four_parts() {
    let v = (32i64 << 48) | (16 << 16) | 2124;
    assert_eq!(driver_version(v), "32.0.16.2124");
    // A first part past 0x7fff makes the i64 negative: still four parts.
    assert_eq!(driver_version(-1), "65535.65535.65535.65535");
    assert_eq!(driver_version(0), "0.0.0.0");
}
