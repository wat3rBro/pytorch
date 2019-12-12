package org.pytorch.torchvision;

import android.media.Image;

import java.nio.ByteBuffer;

public class PytorchVision {
  static {
    System.loadLibrary("pytorch_vision_jni");
  }

  public static void putImage(final Image image) {
    Image.Plane Y = image.getPlanes()[0];
    Image.Plane U = image.getPlanes()[1];
    Image.Plane V = image.getPlanes()[2];

    nativePutYuvImage(
        Y.getBuffer(),
        Y.getRowStride(),
        Y.getPixelStride(),
        U.getBuffer(),
        V.getBuffer(),
        U.getRowStride(),
        U.getPixelStride(),
        image.getWidth(),
        image.getHeight());
  }

  private static native void nativePutYuvImage(
      ByteBuffer yBuffer,
      int yRowStride,
      int yPixelStride,
      ByteBuffer uBuffer,
      ByteBuffer vBuffer,
      int uvRowStride,
      int uvPixelStride,
      int imageWidth,
      int imageHeight);
}
