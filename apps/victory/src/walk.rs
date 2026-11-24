//! The files under the commands' paths, each read whole or with the reason
//! it was not.

use std::io;
use std::path::{Path, PathBuf};

/// Files above this size are skipped (read whole into memory for now);
/// `victory translate` counts them and does not fail on them.
pub(crate) const MAX_FILE: u64 = 2 << 30;

/// Why a file under the roots was not read.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Skip {
    /// Larger than the limit.
    TooLarge,
    Unreadable(io::ErrorKind),
    /// The walk failed here: nothing under this path was listed.
    Walk(Option<io::ErrorKind>),
}

impl Skip {
    /// A reason that groups: never the path, which goes with the example.
    pub(crate) fn reason(&self) -> String {
        match self {
            Skip::TooLarge => "file: too large".into(),
            Skip::Unreadable(kind) => format!("file: unreadable: {kind:?}"),
            Skip::Walk(Some(kind)) => format!("walk: {kind:?}"),
            // walkdir's only error without an I/O cause.
            Skip::Walk(None) => "walk: link loop".into(),
        }
    }
}

/// Every file under `roots` (symbolic links are not followed), read whole
/// if at most `max` bytes.
pub(crate) fn files(
    roots: &[PathBuf],
    max: u64,
) -> impl Iterator<Item = (PathBuf, Result<Vec<u8>, Skip>)> + '_ {
    roots.iter().flat_map(move |root| {
        walkdir::WalkDir::new(root).into_iter().filter_map(move |entry| match entry {
            Err(e) => {
                let at = e.path().unwrap_or(root).to_owned();
                Some((at, Err(Skip::Walk(e.io_error().map(io::Error::kind)))))
            }
            Ok(entry) if !entry.file_type().is_file() => None,
            Ok(entry) => {
                // A size that cannot be learnt is no reason to skip:
                // the read then reports the real error.
                let len = entry.metadata().map_or(0, |m| m.len());
                let data = read_whole(entry.path(), len, max);
                Some((entry.into_path(), data))
            }
        })
    })
}

fn read_whole(path: &Path, len: u64, max: u64) -> Result<Vec<u8>, Skip> {
    if len > max {
        return Err(Skip::TooLarge);
    }
    std::fs::read(path).map_err(|e| Skip::Unreadable(e.kind()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(roots: &[PathBuf], max: u64) -> Vec<(PathBuf, Result<Vec<u8>, Skip>)> {
        files(roots, max).collect()
    }

    #[test]
    fn files_under_the_limit_are_read() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a"), b"abcd").unwrap();
        let got = collect(&[dir.path().to_owned()], 4);
        assert_eq!(got, [(dir.path().join("a"), Ok(b"abcd".to_vec()))]);
    }

    #[test]
    fn files_over_the_limit_are_skipped_with_a_reason() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a"), b"abcd").unwrap();
        let got = collect(&[dir.path().to_owned()], 3);
        assert_eq!(got, [(dir.path().join("a"), Err(Skip::TooLarge))]);
    }

    #[test]
    fn directories_are_walked_not_reported() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("a"), b"x").unwrap();
        let got = collect(&[dir.path().to_owned()], 4);
        assert_eq!(got, [(dir.path().join("sub").join("a"), Ok(b"x".to_vec()))]);
    }

    #[test]
    fn a_root_that_cannot_be_listed_is_a_walk_failure() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing");
        let got = collect(std::slice::from_ref(&missing), 4);
        assert_eq!(got, [(missing, Err(Skip::Walk(Some(io::ErrorKind::NotFound))))]);
    }

    // Permissions cannot make a listed file unreadable for root or on every
    // platform, so the read is checked on a path that is gone.
    #[test]
    fn a_failed_read_keeps_the_io_kind() {
        let dir = tempfile::tempdir().unwrap();
        let got = read_whole(&dir.path().join("gone"), 0, 4);
        assert_eq!(got, Err(Skip::Unreadable(io::ErrorKind::NotFound)));
    }

    #[test]
    fn reasons_name_the_kind_not_the_path() {
        assert_eq!(Skip::TooLarge.reason(), "file: too large");
        assert_eq!(
            Skip::Unreadable(io::ErrorKind::PermissionDenied).reason(),
            "file: unreadable: PermissionDenied"
        );
        assert_eq!(Skip::Walk(Some(io::ErrorKind::NotFound)).reason(), "walk: NotFound");
        assert_eq!(Skip::Walk(None).reason(), "walk: link loop");
    }
}
