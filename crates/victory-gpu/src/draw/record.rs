//! Where a draw, mesh dispatch, dispatch or indirect execution the host did
//! not run is counted: each public method of [`State`](super::State) ends
//! here with the `Result` of its work, so a drop cannot go uncounted.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::object::This;
use crate::pipeline::drops::{self, NotRun};

/// What did not run, for the total it is added to.
#[derive(Clone, Copy)]
pub(super) enum Work {
    /// A draw or mesh dispatch (the last traced draw is marked).
    Draw,
    Dispatch,
    /// An indirect execution: up to `max` draws or dispatches, so counted
    /// apart from them.
    Indirect,
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, Ordering::Relaxed)
}
static INDIRECT: AtomicU64 = AtomicU64::new(0);

/// Counts `work` of `pipeline` that did not run, on its total and where its
/// cause lives; true if it ran.
///
/// # Safety
///
/// `pipeline` is null or a live pipeline `crate::pipeline` created.
pub(super) unsafe fn record(pipeline: usize, work: Work, result: Result<(), NotRun>) -> bool {
    let Err(why) = result else {
        return true;
    };
    match work {
        Work::Draw | Work::Dispatch => super::DROPPED.fetch_add(1, Ordering::Relaxed),
        Work::Indirect => INDIRECT.fetch_add(1, Ordering::Relaxed),
    };
    match why {
        // SAFETY: as this function's contract gives it.
        NotRun::NoHostPipeline => unsafe { crate::pipeline::count_drop(pipeline as This, None) },
        NotRun::General(cause) => drops::general(cause),
        // SAFETY: as this function's contract gives it.
        NotRun::Unprepared(reason) => unsafe {
            crate::pipeline::count_drop(pipeline as This, Some(reason));
        },
    }
    #[cfg(feature = "diagnostics")]
    if matches!(work, Work::Draw)
        && let Some(last) = super::FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut()
    {
        last.push_str(" DROPPED");
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::drops::Cause;

    // One test: both totals are process-wide.
    #[test]
    fn a_dropped_indirect_execution_counts_on_the_indirect_total_only() {
        // From zero.
        take_indirect();
        super::super::take_dropped();
        // SAFETY: a null pipeline, which the contract allows.
        assert!(unsafe { record(0, Work::Indirect, Ok(())) }, "it ran");
        let cause = NotRun::General(Cause::IndirectLayout);
        // SAFETY: as above.
        assert!(!unsafe { record(0, Work::Indirect, Err(cause)) });
        assert_eq!(take_indirect(), 1);
        assert_eq!(super::super::take_dropped(), 0);
    }
}
