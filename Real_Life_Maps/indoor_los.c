/* Pixel-exact, symmetric supercover line of sight for indoor_map_generation.py.
 * A ray that passes precisely between two pixels requires both to be open.
 */
#include <stdint.h>
#include <stddef.h>

static int open_pixel(const uint8_t *open, int height, int width, int row, int col) {
    return row >= 0 && row < height && col >= 0 && col < width
        && open[(size_t)row * (size_t)width + (size_t)col] != 0;
}

int indoor_los(const uint8_t *open, int height, int width,
               int row0, int col0, int row1, int col1) {
    int dx = col1 > col0 ? col1 - col0 : col0 - col1;
    int dy = row1 > row0 ? row1 - row0 : row0 - row1;
    int sx = col1 > col0 ? 1 : -1;
    int sy = row1 > row0 ? 1 : -1;
    int x = col0, y = row0, ix = 0, iy = 0;
    if (!open_pixel(open, height, width, y, x)
        || !open_pixel(open, height, width, row1, col1)) return 0;
    while (ix < dx || iy < dy) {
        int64_t comparison = (int64_t)(1 + 2 * ix) * dy
                           - (int64_t)(1 + 2 * iy) * dx;
        if (comparison == 0) {
            if (!open_pixel(open, height, width, y, x + sx)
                || !open_pixel(open, height, width, y + sy, x)) return 0;
            x += sx; y += sy; ++ix; ++iy;
        } else if (comparison < 0) {
            x += sx; ++ix;
        } else {
            y += sy; ++iy;
        }
        if (!open_pixel(open, height, width, y, x)) return 0;
    }
    return 1;
}

static void set_bit(uint8_t *packed, int row_bytes, int row, int col) {
    packed[(size_t)row * (size_t)row_bytes + (size_t)(col >> 3)]
        |= (uint8_t)(1u << (col & 7));
}

uint64_t indoor_visibility_rows(const uint8_t *open, int height, int width,
                                const int32_t *anchors, const int32_t *components,
                                int count, uint8_t *packed, int row_bytes,
                                int first, int last) {
    uint64_t visible_pairs = 0;
    for (int i = first; i < last; ++i) {
        set_bit(packed, row_bytes, i, i);
        int row0 = anchors[2 * i], col0 = anchors[2 * i + 1];
        for (int j = i + 1; j < count; ++j) {
            if (components[i] != components[j]) continue;
            if (indoor_los(open, height, width, row0, col0,
                           anchors[2 * j], anchors[2 * j + 1])) {
                set_bit(packed, row_bytes, i, j);
                set_bit(packed, row_bytes, j, i);
                ++visible_pairs;
            }
        }
    }
    return visible_pairs;
}
