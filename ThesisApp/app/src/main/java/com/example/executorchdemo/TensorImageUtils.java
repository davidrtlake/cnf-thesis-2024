package com.example.executorchdemo;

import android.graphics.Bitmap;
import java.nio.FloatBuffer;
import org.pytorch.executorch.Tensor;

public final class TensorImageUtils {

  public static Tensor bitmapToFloat32Tensor(
          final Bitmap bitmap, final float[] normMeanRGB, final float normStdRGB[]) {
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);
    bitmapToFloatBuffer(bitmap, 0, 0, width, height, normMeanRGB, normStdRGB, floatBuffer, 0);
    return Tensor.fromBlob(floatBuffer, new long[]{1, 3, height, width});
  }

  public static void bitmapToFloatBuffer(
          final Bitmap bitmap,
          final int x,
          final int y,
          final int width,
          final int height,
          final float[] normMeanRGB,
          final float[] normStdRGB,
          final FloatBuffer outBuffer,
          final int outBufferOffset) {

    final int pixelsCount = height * width;
    final int[] pixels = new int[pixelsCount];
    bitmap.getPixels(pixels, 0, width, x, y, width, height);
    final int offset_g = pixelsCount;
    final int offset_b = 2 * pixelsCount;
    for (int i = 0; i < pixelsCount; i++) {
      final int c = pixels[i];
      float r = ((c >> 16) & 0xff) / 255.0f;
      float g = ((c >> 8) & 0xff) / 255.0f;
      float b = ((c) & 0xff) / 255.0f;
      outBuffer.put(outBufferOffset + i, (r - normMeanRGB[0]) / normStdRGB[0]);
      outBuffer.put(outBufferOffset + offset_g + i, (g - normMeanRGB[1]) / normStdRGB[1]);
      outBuffer.put(outBufferOffset + offset_b + i, (b - normMeanRGB[2]) / normStdRGB[2]);
    }
  }
}
