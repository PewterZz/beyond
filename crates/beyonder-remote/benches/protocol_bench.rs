use beyonder_remote::protocol::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn cell(g: &str, bold: bool) -> PtyCell {
    PtyCell {
        g: g.into(),
        fg: [255, 255, 255],
        bg: None,
        bold,
    }
}

fn grid(rows: usize, cols: usize, fill: &str) -> Vec<Vec<PtyCell>> {
    vec![vec![cell(fill, false); cols]; rows]
}

fn realistic_grid() -> Vec<Vec<PtyCell>> {
    let mut g = grid(24, 80, " ");
    for (i, ch) in "$ cargo build --release".chars().enumerate() {
        g[0][i] = cell(&ch.to_string(), false);
    }
    for (i, ch) in "   Compiling beyonder v0.1.0 (/Users/user/Projects/Beyonder)"
        .chars()
        .enumerate()
    {
        if i < 80 {
            g[1][i] = cell(&ch.to_string(), false);
        }
    }
    for (i, ch) in "    Finished `release` profile [optimized] target(s) in 42.3s"
        .chars()
        .enumerate()
    {
        if i < 80 {
            g[2][i] = cell(&ch.to_string(), false);
        }
    }
    g
}

fn bench_frame_diff(c: &mut Criterion) {
    let prev = realistic_grid();
    let mut curr = prev.clone();
    // Change 10 cells to simulate a cursor move + small output.
    for slot in curr[3].iter_mut().take(10) {
        *slot = cell("X", false);
    }

    c.bench_function("frame_diff_10_changed", |b| {
        b.iter(|| compute_frame_diff(black_box(&prev), black_box(&curr)))
    });

    let curr_full = grid(24, 80, "X");
    c.bench_function("frame_diff_all_changed", |b| {
        b.iter(|| compute_frame_diff(black_box(&prev), black_box(&curr_full)))
    });

    c.bench_function("frame_diff_identical", |b| {
        b.iter(|| compute_frame_diff(black_box(&prev), black_box(&prev)))
    });
}

fn bench_pack_cells(c: &mut Criterion) {
    let cells = realistic_grid();

    c.bench_function("pack_cells_24x80", |b| {
        b.iter(|| pack_cells(black_box(&cells)))
    });

    let packed = pack_cells(&cells);
    c.bench_function("unpack_cells_24x80", |b| {
        b.iter(|| unpack_cells(black_box(&packed), 24, 80))
    });
}

fn bench_pack_diff(c: &mut Criterion) {
    let prev = realistic_grid();
    let mut curr = prev.clone();
    for i in 0..50 {
        curr[i / 80][i % 80] = cell("Y", true);
    }
    let changes = compute_frame_diff(&prev, &curr).unwrap();

    c.bench_function("pack_diff_50_changes", |b| {
        b.iter(|| pack_diff_changes(black_box(&changes)))
    });

    let packed = pack_diff_changes(&changes);
    c.bench_function("unpack_diff_50_changes", |b| {
        b.iter(|| unpack_diff_changes(black_box(&packed)))
    });
}

fn bench_zstd_compression(c: &mut Criterion) {
    let cells = realistic_grid();
    let msg = ServerMsg::PtyFramePacked(PtyFramePacked {
        cols: 80,
        rows: 24,
        cursor_col: 0,
        cursor_row: 3,
        packed: pack_cells(&cells),
    });
    let mut cbor = Vec::new();
    ciborium::into_writer(&msg, &mut cbor).unwrap();

    c.bench_function("zstd_compress_frame", |b| {
        b.iter(|| compress_cbor(black_box(&cbor), 1))
    });

    let compressed = compress_cbor(&cbor, 1).unwrap();
    c.bench_function("zstd_decompress_frame", |b| {
        b.iter(|| decompress_frame(black_box(&compressed)))
    });
}

fn bench_cbor_vs_packed(c: &mut Criterion) {
    let cells = realistic_grid();

    // Old path: CBOR-encode full PtyFrame.
    let cbor_msg = ServerMsg::PtyFrame(PtyFrame {
        cols: 80,
        rows: 24,
        cursor_col: 0,
        cursor_row: 0,
        cells: cells.clone(),
    });
    c.bench_function("encode_cbor_full_frame", |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(32768);
            ciborium::into_writer(black_box(&cbor_msg), &mut buf).unwrap();
            buf
        })
    });

    // New path: pack cells + CBOR-encode packed frame.
    c.bench_function("encode_packed_frame", |b| {
        b.iter(|| {
            let packed = pack_cells(black_box(&cells));
            let msg = ServerMsg::PtyFramePacked(PtyFramePacked {
                cols: 80,
                rows: 24,
                cursor_col: 0,
                cursor_row: 0,
                packed,
            });
            let mut buf = Vec::with_capacity(16384);
            ciborium::into_writer(&msg, &mut buf).unwrap();
            buf
        })
    });

    // Full pipeline: pack + CBOR + zstd.
    c.bench_function("encode_packed_zstd_frame", |b| {
        b.iter(|| {
            let packed = pack_cells(black_box(&cells));
            let msg = ServerMsg::PtyFramePacked(PtyFramePacked {
                cols: 80,
                rows: 24,
                cursor_col: 0,
                cursor_row: 0,
                packed,
            });
            let mut cbor = Vec::with_capacity(16384);
            ciborium::into_writer(&msg, &mut cbor).unwrap();
            compress_cbor(&cbor, 1)
        })
    });
}

fn bench_adaptive_throttle(c: &mut Criterion) {
    c.bench_function("throttle_active_report", |b| {
        let mut t = AdaptiveThrottle::default();
        b.iter(|| {
            t.report_activity(black_box(true));
            black_box(t.interval_ms);
        })
    });

    c.bench_function("throttle_idle_rampdown", |b| {
        b.iter(|| {
            let mut t = AdaptiveThrottle::default();
            for _ in 0..20 {
                t.report_activity(false);
            }
            black_box(t.interval_ms);
        })
    });
}

criterion_group!(
    benches,
    bench_frame_diff,
    bench_pack_cells,
    bench_pack_diff,
    bench_zstd_compression,
    bench_cbor_vs_packed,
    bench_adaptive_throttle,
);
criterion_main!(benches);
