#include <cassert>
#include <vector>

#include <libyuv.h>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>



#if defined(__ANDROID__)

#include <android/log.h>
#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "pytorch-vision-jni", __VA_ARGS__)

#endif

namespace pytorch_vision_jni {
class PytorchVisionJni : public facebook::jni::JavaClass<PytorchVisionJni> {
 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/torchvision/PytorchVision;";

  static void putYuvImage(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> yBuffer,
      const int yRowStride,
      const int yPixelStride,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> uBuffer,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> vBuffer,
      const int uvRowStride,
      const int uvPixelStride,
      const int imageWidth,
      const int imageHeight) {

    int halfImageWidth = (imageWidth + 1) / 2;
    int halfImageHeight = (imageHeight + 1) / 2;

    const uint32_t yuvSize = imageHeight * imageWidth + 2 * halfImageHeight * halfImageWidth;

    std::vector<uint8_t> yuvData;
    std::vector<uint8_t> rgbData;

    if (yuvData.size() != yuvSize) {
      yuvData.resize(yuvSize);
    }

    const uint32_t argbSize = 4 * imageHeight * imageWidth;
    if (rgbData.size() != argbSize) {
      rgbData.resize(argbSize);
    }

    const auto ret = libyuv::Android420ToI420(
        yBuffer->getDirectBytes(),
        yRowStride,
        uBuffer->getDirectBytes(),
        uvRowStride,
        vBuffer->getDirectBytes(),
        uvRowStride,
        uvPixelStride,
        yuvData.data(),
        imageWidth,
        yuvData.data() + imageHeight * imageWidth,
        halfImageWidth,
        yuvData.data() + imageHeight * imageWidth +
            halfImageHeight * halfImageWidth,
        halfImageWidth,
        imageWidth,
        imageHeight);
    assert(ret != 0);

/*
    const auto cvtRet = libyuv::I420ToARGB(
        yuvData.data(),
        imageWidth,
        yuvData.data() + imageHeight * imageWidth,
        halfImageWidth,
        yuvData.data() + imageHeight * imageWidth +
            halfImageHeight * halfImageWidth,
        halfImageWidth,
        rgbData.data(),
        4 * imageWidth,
        imageWidth,
        imageHeight);
    assert(cvtRey != 0);
*/
  }

  static void registerNatives() {
    javaClassStatic()->registerNatives({
        makeNativeMethod("nativePutYuvImage", PytorchVisionJni::putYuvImage),
    });
  }

  static void cameraOutputToAtTensor(facebook::jni::alias_ref<jclass>) {
    ALOGI("XXX cameraOutputToAtTensor()");
  }

};
} // namespace pytorch_vision_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { pytorch_vision_jni::PytorchVisionJni::registerNatives(); });
}