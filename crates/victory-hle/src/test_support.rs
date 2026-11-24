//! Helpers for tests that go through a pointer the code under test wrote
//! out (`create` fills a local that starts null).

use std::ptr::NonNull;

/// `p`, which the test fails on if it is null, so the pointers it goes
/// through afterwards are known to be set.
pub(crate) fn valid<T>(p: *const T) -> *const T {
    NonNull::new(p.cast_mut()).expect("the code under test wrote a pointer").as_ptr().cast_const()
}

/// As [`valid`], for a pointer the test writes through.
pub(crate) fn valid_mut<T>(p: *mut T) -> *mut T {
    NonNull::new(p).expect("the code under test wrote a pointer").as_ptr()
}
