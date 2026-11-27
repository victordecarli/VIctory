//! The frame report: what the last frames cost, logged every few seconds
//! with the `victory::summary` target as one event with a field per number.

use crate::memory::Memory;

/// What the counters added up to since the last report.
pub(super) struct Counts {
    /// Frames presented, at least 1.
    pub(super) frames: u32,
    pub(super) seconds: f64,
    pub(super) draws: u64,
    /// Microseconds preparing submissions, then waiting for the GPU.
    pub(super) submission_us: [u64; 2],
    /// Title memory copies (microseconds, bytes), then texture refreshes
    /// (microseconds, bytes).
    pub(super) preparing: [u64; 4],
    /// Copying shader writes back (microseconds, bytes).
    pub(super) copy_back: [u64; 2],
    pub(super) dropped: u64,
    pub(super) indirect: u64,
}

/// Per frame, over every frame since the last report: times in ms, sizes
/// in MB.
pub(super) struct FrameReport {
    pub(super) fps: f64,
    pub(super) ms_per_frame: f64,
    /// Preparing submissions.
    pub(super) cpu_ms: f64,
    pub(super) copies_ms: f64,
    pub(super) copies_mb: f64,
    pub(super) textures_ms: f64,
    pub(super) textures_mb: f64,
    /// Waiting for the GPU.
    pub(super) gpu_ms: f64,
    pub(super) copy_back_ms: f64,
    pub(super) copy_back_mb: f64,
    pub(super) draws_per_frame: f64,
    /// Draws dropped without a host pipeline.
    pub(super) dropped_per_frame: f64,
    /// Host pipelines being made when logged.
    pub(super) pipelines_making: u64,
    /// Indirect executions dropped (each also counted on its cause).
    pub(super) indirect_dropped_per_frame: f64,
    /// The frame being recorded, as the drop events name it.
    pub(super) frame: u64,
    /// Memory when logged, not per frame.
    pub(super) memory: Memory,
}

impl FrameReport {
    /// The report of `counts`, with the pipelines being made, the frame
    /// being recorded and the memory now.
    pub(super) fn per_frame(
        counts: &Counts,
        pipelines_making: u64,
        frame: u64,
        memory: Memory,
    ) -> Self {
        let Counts {
            frames,
            seconds,
            draws,
            submission_us: [cpu_us, gpu_us],
            preparing: [copies_us, copies_bytes, textures_us, textures_bytes],
            copy_back: [back_us, back_bytes],
            dropped,
            indirect,
        } = *counts;
        let frames = f64::from(frames.max(1));
        let ms = |us: u64| us as f64 / 1e3 / frames;
        let mb = |bytes: u64| bytes as f64 / 1e6 / frames;
        let each = |n: u64| n as f64 / frames;
        FrameReport {
            fps: frames / seconds,
            ms_per_frame: seconds * 1000.0 / frames,
            cpu_ms: ms(cpu_us),
            copies_ms: ms(copies_us),
            copies_mb: mb(copies_bytes),
            textures_ms: ms(textures_us),
            textures_mb: mb(textures_bytes),
            gpu_ms: ms(gpu_us),
            copy_back_ms: ms(back_us),
            copy_back_mb: mb(back_bytes),
            draws_per_frame: each(draws),
            dropped_per_frame: each(dropped),
            pipelines_making,
            indirect_dropped_per_frame: each(indirect),
            frame,
            memory,
        }
    }

    /// Logs the report: times, sizes and the frame rate to a tenth, counts
    /// per frame and memory (MB) whole.
    pub(super) fn log(&self) {
        let FrameReport {
            fps,
            ms_per_frame,
            cpu_ms,
            copies_ms,
            copies_mb,
            textures_ms,
            textures_mb,
            gpu_ms,
            copy_back_ms,
            copy_back_mb,
            draws_per_frame,
            dropped_per_frame,
            pipelines_making,
            indirect_dropped_per_frame,
            frame,
            memory:
                Memory {
                    commit,
                    system_commit,
                    commit_limit,
                    copies,
                    snapshots,
                    staging,
                    title_private,
                    title_mapped,
                },
        } = *self;
        let tenth = |x: f64| (x * 10.0).round() / 10.0;
        let whole = |x: f64| x.round() as u64;
        let mb = |bytes: u64| whole(bytes as f64 / 1e6);
        tracing::info!(
            target: crate::SUMMARY,
            fps = tenth(fps),
            ms_per_frame = tenth(ms_per_frame),
            cpu_ms = tenth(cpu_ms),
            copies_ms = tenth(copies_ms),
            copies_mb = tenth(copies_mb),
            textures_ms = tenth(textures_ms),
            textures_mb = tenth(textures_mb),
            gpu_ms = tenth(gpu_ms),
            copy_back_ms = tenth(copy_back_ms),
            copy_back_mb = tenth(copy_back_mb),
            draws_per_frame = whole(draws_per_frame),
            dropped_per_frame = whole(dropped_per_frame),
            pipelines_making,
            indirect_dropped_per_frame = whole(indirect_dropped_per_frame),
            frame,
            commit_mb = mb(commit),
            system_commit_mb = mb(system_commit),
            commit_limit_mb = mb(commit_limit),
            copies_held_mb = mb(copies),
            snapshots_held_mb = mb(snapshots),
            staging_held_mb = mb(staging),
            title_private_mb = mb(title_private),
            title_mapped_mb = mb(title_mapped),
            "frame report"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Metadata, Subscriber};

    use super::{Counts, FrameReport};
    use crate::memory::Memory;

    type Fields = BTreeMap<String, String>;

    /// Keeps each event's target and fields, the values as `Debug` text.
    struct Recorder(Arc<Mutex<Vec<(String, Fields)>>>);

    struct Visitor<'a>(&'a mut Fields);

    impl Visit for Visitor<'_> {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0.insert(field.name().to_owned(), format!("{value:?}"));
        }
    }

    impl Subscriber for Recorder {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }

        fn record(&self, _: &Id, _: &Record<'_>) {}

        fn record_follows_from(&self, _: &Id, _: &Id) {}

        fn event(&self, event: &Event<'_>) {
            let mut fields = Fields::new();
            event.record(&mut Visitor(&mut fields));
            self.0.lock().unwrap().push((event.metadata().target().to_owned(), fields));
        }

        fn enter(&self, _: &Id) {}

        fn exit(&self, _: &Id) {}
    }

    #[test]
    fn each_number_is_a_field_rounded_as_the_text_was() {
        let report = FrameReport {
            fps: 59.94,
            ms_per_frame: 16.683,
            cpu_ms: 3.24,
            copies_ms: 1.04,
            copies_mb: 12.36,
            textures_ms: 0.57,
            textures_mb: 7.0,
            gpu_ms: 9.96,
            copy_back_ms: 0.04,
            copy_back_mb: 0.26,
            draws_per_frame: 1234.4,
            dropped_per_frame: 2.6,
            pipelines_making: 3,
            indirect_dropped_per_frame: 0.4,
            frame: 812,
            memory: Memory {
                commit: 22_302_400_000,
                system_commit: 46_309_600_000,
                commit_limit: 47_632_000_000,
                copies: 1_200_600_000,
                snapshots: 174_400_000,
                staging: 2_400_000_000,
                title_private: 12_100_000_000,
                title_mapped: 6_549_700_000,
            },
        };
        let events = Arc::new(Mutex::new(Vec::new()));
        tracing::subscriber::with_default(Recorder(events.clone()), || report.log());
        let events = events.lock().unwrap();
        let [(target, fields)] = &events[..] else {
            panic!("one event, got {events:?}");
        };
        assert_eq!(target, "victory::summary");
        let want: Fields = [
            ("message", "frame report"),
            ("fps", "59.9"),
            ("ms_per_frame", "16.7"),
            ("cpu_ms", "3.2"),
            ("copies_ms", "1.0"),
            ("copies_mb", "12.4"),
            ("textures_ms", "0.6"),
            ("textures_mb", "7.0"),
            ("gpu_ms", "10.0"),
            ("copy_back_ms", "0.0"),
            ("copy_back_mb", "0.3"),
            ("draws_per_frame", "1234"),
            ("dropped_per_frame", "3"),
            ("pipelines_making", "3"),
            ("indirect_dropped_per_frame", "0"),
            ("frame", "812"),
            ("commit_mb", "22302"),
            ("system_commit_mb", "46310"),
            ("commit_limit_mb", "47632"),
            ("copies_held_mb", "1201"),
            ("snapshots_held_mb", "174"),
            ("staging_held_mb", "2400"),
            ("title_private_mb", "12100"),
            ("title_mapped_mb", "6550"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
        assert_eq!(*fields, want);
    }

    #[test]
    fn the_report_averages_every_frame_since_the_last_one() {
        // 5 s and 4 frames since the last report, whatever the caption
        // measured in between.
        let counts = Counts {
            frames: 4,
            seconds: 5.0,
            draws: 2000,
            submission_us: [8000, 20_000],
            preparing: [4000, 8_000_000, 2000, 1_000_000],
            copy_back: [400, 2_000_000],
            dropped: 12,
            indirect: 6,
        };
        let now = Memory {
            commit: 1,
            system_commit: 2,
            commit_limit: 3,
            copies: 4,
            snapshots: 5,
            staging: 6,
            title_private: 7,
            title_mapped: 8,
        };
        let r = FrameReport::per_frame(&counts, 7, 900, now);
        let FrameReport {
            fps,
            ms_per_frame,
            cpu_ms,
            copies_ms,
            copies_mb,
            textures_ms,
            textures_mb,
            gpu_ms,
            copy_back_ms,
            copy_back_mb,
            draws_per_frame,
            dropped_per_frame,
            pipelines_making,
            indirect_dropped_per_frame,
            frame,
            memory,
        } = r;
        assert_eq!(
            [fps, ms_per_frame, cpu_ms, copies_ms, copies_mb, textures_ms, textures_mb, gpu_ms],
            [0.8, 1250.0, 2.0, 1.0, 2.0, 0.5, 0.25, 5.0]
        );
        assert_eq!(
            [
                copy_back_ms,
                copy_back_mb,
                draws_per_frame,
                dropped_per_frame,
                indirect_dropped_per_frame
            ],
            [0.1, 0.5, 500.0, 3.0, 1.5]
        );
        assert_eq!((pipelines_making, frame), (7, 900));
        // Memory is what it is now, not averaged.
        let Memory {
            commit,
            system_commit,
            commit_limit,
            copies,
            snapshots,
            staging,
            title_private,
            title_mapped,
        } = memory;
        assert_eq!(
            [commit, system_commit, commit_limit, copies, snapshots, staging],
            [1, 2, 3, 4, 5, 6]
        );
        assert_eq!([title_private, title_mapped], [7, 8]);
    }
}
