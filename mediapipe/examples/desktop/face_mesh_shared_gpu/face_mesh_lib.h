#ifndef FACE_MESH_LIBRARY_H
#define FACE_MESH_LIBRARY_H

#include <cstdlib>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

class MPFaceMeshDetector {
public:
  MPFaceMeshDetector(int numFaces,
                     bool with_attention,
                     const char *face_detection_model_path,
                     const char *face_landmark_model_path,
                     const char *face_landmark_model_with_attention_path,
                     int window_size_param,
                     float velocity_scale_param);

  void DetectFaces(const cv::Mat &camera_frame,
                   cv::Rect *multi_face_bounding_boxes,
                   int fps,
                   int *numFaces);

  void DetectLandmarks(cv::Point2f **multi_face_landmarks, int *numFaces);
  void DetectLandmarks(cv::Point3f **multi_face_landmarks, int *numFaces);

  static constexpr auto kLandmarksNumWithoutAttention = 468;
  static constexpr auto kLandmarksNumWithAttention = 478;
  static int kLandmarksNum;

private:
  absl::Status InitFaceMeshDetector(int numFaces,
                                    bool with_attention,
                                    const char *face_detection_model_path,
                                    const char *face_landmark_model_path,
                                    const char *face_landmark_model_with_attention_path,
                                    int window_size_param,
                                    float velocity_scale_param);
  absl::Status DetectFacesWithStatus(const cv::Mat &camera_frame,
                                     cv::Rect *multi_face_bounding_boxes,
                                     int fps,
                                     int *numFaces);

  absl::Status DetectLandmarksWithStatus(cv::Point2f **multi_face_landmarks);
  absl::Status DetectLandmarksWithStatus(cv::Point3f **multi_face_landmarks);

  static constexpr auto kInputStream = "input_video";
  static constexpr auto kInputStream_fps = "fps";
  static constexpr auto kOutputStream_landmarks = "filtered_multi_face_landmarks";
  static constexpr auto kOutputStream_faceCount = "face_count";
  static constexpr auto kOutputStream_face_rects_from_landmarks =
      "face_rects_from_landmarks";

  static const std::string graphConfig;

  mediapipe::CalculatorGraph graph;

  std::unique_ptr<mediapipe::OutputStreamPoller> landmarks_poller_ptr;
  std::unique_ptr<mediapipe::OutputStreamPoller> face_count_poller_ptr;
  std::unique_ptr<mediapipe::OutputStreamPoller>
      face_rects_from_landmarks_poller_ptr;

  int face_count;
  int image_width;
  int image_height;
  mediapipe::Packet face_landmarks_packet;
  
  mediapipe::GlCalculatorHelper gpu_helper;
};

#ifdef __cplusplus
extern "C" {
#endif

MPFaceMeshDetector *
MPFaceMeshDetectorConstruct(int numFaces,
    bool with_attention = true,
    const char *face_detection_model_path = "mediapipe/modules/face_detection/face_detection_short_range.tflite",
    const char *face_landmark_model_path = "mediapipe/modules/face_landmark/face_landmark.tflite",
    const char *face_landmark_model_with_attention_path = "mediapipe/modules/face_landmark/face_landmark_with_attention.tflite",
    int window_size_param = 10,
    float velocity_scale_param = 10.0);

void MPFaceMeshDetectorDestruct(MPFaceMeshDetector *detector);

void MPFaceMeshDetectorDetectFaces(
    MPFaceMeshDetector *detector, const cv::Mat &camera_frame,
    cv::Rect *multi_face_bounding_boxes, int fps, int *numFaces);

void
MPFaceMeshDetectorDetect2DLandmarks(MPFaceMeshDetector *detector,
                                    cv::Point2f **multi_face_landmarks,
                                    int *numFaces);
void
MPFaceMeshDetectorDetect3DLandmarks(MPFaceMeshDetector *detector,
                                    cv::Point3f **multi_face_landmarks,
                                    int *numFaces);

extern const int MPFaceMeshDetectorLandmarksNum;

#ifdef __cplusplus
};
#endif
#endif