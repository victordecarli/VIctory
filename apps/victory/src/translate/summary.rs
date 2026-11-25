//! `translate --json`: a run's counts per kind of program, in the form
//! `cargo xtask progress` reads. Aggregate counts only, never a shader's
//! name, hash or an example, so the file may be committed
//! (`docs/asset-policy.md`).

use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use super::tally::Tally;

/// The version of the [`Summary`] format; `cargo xtask progress` refuses
/// any other. Raise it when a field changes meaning or goes away.
pub(super) const SCHEMA: u32 = 1;

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct Summary {
    pub(super) schema: u32,
    /// Whether every file given was measured: with an infrastructure
    /// failure (a file not read, a panic) every count is a lower bound.
    pub(super) complete: bool,
    /// `--host-lanes`: the widest wave the host was assumed to run. Below
    /// 64, console wave64 compute that cannot run as two waves keeps its
    /// 64-lane waves, which such a host refuses.
    pub(super) host_lanes: u32,
    /// Vertex, pixel and the other graphics stages.
    pub(super) vertex_pixel: Kind,
    pub(super) compute: Kind,
    /// Of `compute.validated`, the programs that need 64-lane waves: a host
    /// with fewer lanes refuses them.
    pub(super) compute_wave64: usize,
    pub(super) mesh: Kind,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct Kind {
    /// Programs of this kind found, duplicates skipped.
    pub(super) found: usize,
    /// Of those, the ones emitted as DXIL that the validator accepted:
    /// the project counts a shader as translated only then.
    pub(super) validated: usize,
}

impl Summary {
    pub(super) fn of(t: &Tally, host_lanes: u32) -> Self {
        Self {
            schema: SCHEMA,
            complete: t.infra.is_empty(),
            host_lanes,
            vertex_pixel: Kind { found: t.found, validated: t.validated },
            compute: Kind { found: t.cs_found, validated: t.cs_validated },
            compute_wave64: t.cs_validated_wave64,
            mesh: Kind { found: t.ms_found, validated: t.ms_validated },
        }
    }
}

/// Writes `t`'s summary to `path` as pretty JSON.
pub(super) fn write(path: &Path, t: &Tally, host_lanes: u32) -> Result<()> {
    let text = serde_json::to_string_pretty(&Summary::of(t, host_lanes))?;
    std::fs::write(path, text + "\n")
        .with_context(|| format!("writing the summary to {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_kind_counts_what_the_validator_accepted() {
        let t = Tally {
            found: 10,
            validated: 6,
            cs_found: 5,
            cs_validated: 3,
            cs_validated_wave64: 1,
            ms_found: 3,
            ms_validated: 2,
            // Emitted but refused by the validator: not translated.
            emitted: 7,
            cs_emitted: 4,
            cs_wave64: 2,
            ms_emitted: 3,
            // Stage-internal counts are not part of the summary.
            emit_programs: 8,
            ..Tally::default()
        };
        assert_eq!(
            Summary::of(&t, 32),
            Summary {
                schema: SCHEMA,
                complete: true,
                host_lanes: 32,
                vertex_pixel: Kind { found: 10, validated: 6 },
                compute: Kind { found: 5, validated: 3 },
                compute_wave64: 1,
                mesh: Kind { found: 3, validated: 2 },
            }
        );
    }

    #[test]
    fn an_infrastructure_failure_marks_it_incomplete() {
        let mut t = Tally::default();
        t.infra("file: unreadable: NotFound".into(), || "a".into());
        assert!(!Summary::of(&t, 64).complete);
    }

    #[test]
    fn the_file_holds_counts_and_no_examples() {
        let mut t = Tally { found: 2, validated: 1, ..Tally::default() };
        t.fail("emit: x".into(), || "shaders.pak@0x10 Ps".into());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        write(&path, &t, 64).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["vertex_pixel"]["found"], 2);
        assert_eq!(json["schema"], SCHEMA);
        assert!(!text.contains("shaders.pak"), "{text}");
    }
}
