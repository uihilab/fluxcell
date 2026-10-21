// Index of the cell (x, y), clamped to the grid; rows run south to north.
fn cellAt(x: i32, y: i32) -> u32 {
    return u32(clamp(y, 0, i32(view.height) - 1)) * view.width + u32(clamp(x, 0, i32(view.width) - 1));
}
