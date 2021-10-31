#ifndef MEDIAPIPE_MP_FACE_MESH_DETECTOR_H
#define MEDIAPIPE_MP_FACE_MESH_DETECTOR_H

#include "absl/memory/memory.h"
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

class MPFaceMeshDetector {
public:
  MPFaceMeshDetector(int numFaces, const char *face_detection_model_path,
                     const char *face_landmark_model_path);

  void DetectFaces(const cv::Mat &camera_frame,
                   cv::Rect *multi_face_bounding_boxes, int *numFaces);

  void DetectLandmarks(cv::Point2f **multi_face_landmarks, int *numFaces);
  void DetectLandmarks(cv::Point3f **multi_face_landmarks, int *numFaces);

  static constexpr auto kLandmarksNum = 468;

private:
  absl::Status InitFaceMeshDetector(int numFaces,
                                    const char *face_detection_model_path,
                                    const char *face_landmark_model_path);
  absl::Status DetectFacesWithStatus(const cv::Mat &camera_frame,
                                     cv::Rect *multi_face_bounding_boxes,
                                     int *numFaces);
  absl::Status DetectLandmarksWithStatus(cv::Point2f **multi_face_landmarks);
  absl::Status DetectLandmarksWithStatus(cv::Point3f **multi_face_landmarks);

  static constexpr auto kInputStream = "input_video";
  static constexpr auto kOutputStream_landmarks = "multi_face_landmarks";
  static constexpr auto kOutputStream_faceCount = "face_count";
  static constexpr auto kOutputStream_face_rects_from_landmarks = "face_rects_from_landmarks";

  static const std::string graphConfig;

  mediapipe::CalculatorGraph graph;

  std::unique_ptr<mediapipe::OutputStreamPoller> landmarks_poller_ptr;
  std::unique_ptr<mediapipe::OutputStreamPoller> face_count_poller_ptr;
  std::unique_ptr<mediapipe::OutputStreamPoller> face_rects_from_landmarks_poller_ptr;

  int face_count;
  int image_width;
  int image_height;
  mediapipe::Packet face_landmarks_packet;
};

#endif //MEDIAPIPE_MP_FACE_MESH_DETECTOR_H
