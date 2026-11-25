//! How a shader container is named in logs and in `VICTORY_SKIP_PIPELINES`:
//! the first 8 bytes of its BLAKE3, written as 16 lowercase hex digits.

/// The first 8 bytes of a shader container's BLAKE3.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ShaderHash([u8; 8]);

const DIGITS: &[u8; 16] = b"0123456789abcdef";

impl ShaderHash {
    /// The name of the container whose full hash is `full`.
    pub(crate) fn of(full: &blake3::Hash) -> Self {
        let mut first = [0; 8];
        first.copy_from_slice(&full.as_bytes()[..8]);
        ShaderHash(first)
    }

    /// The 16 lowercase hex digits, as logged. A pipeline's key hashes
    /// these, so it stays what earlier runs stored.
    pub(crate) fn hex(self) -> [u8; 16] {
        let mut out = [0; 16];
        for (pair, byte) in out.as_chunks_mut::<2>().0.iter_mut().zip(self.0) {
            *pair = [DIGITS[usize::from(byte >> 4)], DIGITS[usize::from(byte & 0xf)]];
        }
        out
    }

    /// A hash written as 16 hex digits of either case; `None` for anything
    /// else.
    #[cfg(any(test, all(windows, feature = "diagnostics")))]
    pub(crate) fn parse(text: &str) -> Option<Self> {
        let digits = text.as_bytes();
        if digits.len() != 16 {
            return None;
        }
        let mut out = [0; 8];
        for (byte, &[high, low]) in out.iter_mut().zip(digits.as_chunks::<2>().0) {
            let high = char::from(high).to_digit(16)?;
            let low = char::from(low).to_digit(16)?;
            *byte = u8::try_from(high << 4 | low).ok()?;
        }
        Some(ShaderHash(out))
    }
}

impl std::fmt::Display for ShaderHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(std::str::from_utf8(&self.hex()).map_err(|_| std::fmt::Error)?)
    }
}

impl std::fmt::Debug for ShaderHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::ShaderHash;

    #[test]
    fn a_hash_is_written_as_the_start_of_the_full_hex() {
        let full = blake3::hash(b"a shader container");
        let hash = ShaderHash::of(&full);
        assert_eq!(hash.to_string(), full.to_hex()[..16]);
        assert_eq!(hash.hex(), full.to_hex().as_bytes()[..16]);
    }

    #[test]
    fn written_hashes_read_back_in_either_case() {
        let hash = ShaderHash::of(&blake3::hash(b"x"));
        let text = hash.to_string();
        assert_eq!(ShaderHash::parse(&text), Some(hash));
        assert_eq!(ShaderHash::parse(&text.to_ascii_uppercase()), Some(hash));
    }

    #[test]
    fn anything_but_16_hex_digits_is_refused() {
        for text in
            ["", "0123456789abcde", "0123456789abcdef0", "0123456789abcdeg", "+123456789abcdef"]
        {
            assert_eq!(ShaderHash::parse(text), None, "{text:?}");
        }
    }
}
