#include <inttypes.h>

unsigned char coerce(double pxl);

void apply_spatial_filter(unsigned char **orig_image,
                          double **final_image, uint16_t width,
                          uint16_t height, double **kernel, uint16_t k_rows,
                          uint16_t k_cols);

/**
 * Coerce a pixel value into the range [0, 255]
 */
inline unsigned char coerce(double pxl) {
  if (pxl > 255)
    pxl = 255;
  if (pxl < 0)
    pxl = 0;
  return (unsigned char)pxl;
}

void apply_spatial_filter(unsigned char **orig_image,
                          double **final_image, uint16_t width,
                          uint16_t height, double **kernel, uint16_t k_rows,
                          uint16_t k_cols) {
  uint8_t half_k_rows = (uint8_t)(k_rows / 2);
  uint8_t half_k_cols = (uint8_t)(k_cols / 2);
  for (uint16_t i = 0; i < height; i++) {
    for (uint16_t j = 0; j < width; j++) {
      double pxl = 0;

      for (uint8_t s = 0; s < k_rows; s++) {
        for (uint8_t t = 0; t < k_cols; t++) {
          uint16_t x = i - half_k_rows + s;
          uint16_t y = j - half_k_cols + t;

          unsigned char orig_image_x_y;

          if (x < 0 || x >= height || y < 0 || y >= width)
            orig_image_x_y = 0;
          else
            orig_image_x_y = orig_image[x][y];

          double weighted_pxl = orig_image_x_y * kernel[s][t];

          pxl += weighted_pxl;
        }
      }

      final_image[i][j] = pxl;
    }
  }
}

int main() {}
