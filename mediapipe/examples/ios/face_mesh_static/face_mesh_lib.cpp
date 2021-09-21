#include "face_mesh_lib.h"
#include "face_mesh_graph.h"

MPFaceMeshDetector::MPFaceMeshDetector(int numFaces,
                                       const char *face_detection_model_path,
                                       const char *face_landmark_model_path) {
  face_mesh_graph::staticLinkAllCalculators();

  const auto status = InitFaceMeshDetector(numFaces, face_detection_model_path,
                                           face_landmark_model_path);
  if (!status.ok()) {
    LOG(INFO) << "Failed constructing FaceMeshDetector: " << status.message();
  }
}

absl::Status
MPFaceMeshDetector::InitFaceMeshDetector(int numFaces,
                                         const char *face_detection_model_path,
                                         const char *face_landmark_model_path) {
  if (face_detection_model_path == nullptr) {
    face_detection_model_path =
        "mediapipe/modules/face_detection/face_detection_short_range.tflite";
  }

  if (face_landmark_model_path == nullptr) {
    face_landmark_model_path =
        "mediapipe/modules/face_landmark/face_landmark.tflite";
  }


  LOG(INFO) << "Get calculator graph config contents: " << face_mesh_graph::graphConfig;

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
              face_mesh_graph::graphConfig);
  LOG(INFO) << "Initialize the calculator graph.";

  // Prepare graph config.
  std::map<std::string, mediapipe::Packet> extra_side_packets;
  extra_side_packets.insert({"num_faces", mediapipe::MakePacket<int>(std::max(numFaces, 1))});
  extra_side_packets.insert({"face_detection_model_path", mediapipe::MakePacket<std::string>(face_detection_model_path)});
  extra_side_packets.insert({"face_landmark_model_path", mediapipe::MakePacket<std::string>(face_landmark_model_path)});

  MP_RETURN_IF_ERROR(graph.Initialize(config, extra_side_packets));

  LOG(INFO) << "Start running the calculator graph.";

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarks_poller,
                   graph.AddOutputStreamPoller(kOutputStream_landmarks));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller face_count_poller,
                   graph.AddOutputStreamPoller(kOutputStream_faceCount));
  ASSIGN_OR_RETURN(
      mediapipe::OutputStreamPoller face_rects_from_landmarks_poller,
      graph.AddOutputStreamPoller(kOutputStream_face_rects_from_landmarks));

  landmarks_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
      std::move(landmarks_poller));
  face_count_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
      std::move(face_count_poller));
  face_rects_from_landmarks_poller_ptr =
      std::make_unique<mediapipe::OutputStreamPoller>(
          std::move(face_rects_from_landmarks_poller));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "MPFaceMeshDetector constructed successfully.";

  return absl::OkStatus();
}

absl::Status
MPFaceMeshDetector::DetectFacesWithStatus(const cv::Mat &camera_frame,
                                          cv::Rect *multi_face_bounding_boxes,
                                          int *numFaces) {
  if (!numFaces || !multi_face_bounding_boxes) {
    return absl::InvalidArgumentError(
        "MPFaceMeshDetector::DetectFacesWithStatus requires notnull pointer to "
        "save results data.");
  }

  // Reset face counts.
  *numFaces = 0;
  face_count = 0;

  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  camera_frame.copyTo(input_frame_mat);

  // Send image packet into the graph.
  size_t frame_timestamp_us = static_cast<double>(cv::getTickCount()) /
                              static_cast<double>(cv::getTickFrequency()) * 1e6;
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us))));

  // Get face count.
  mediapipe::Packet face_count_packet;
  if (!face_count_poller_ptr ||
      !face_count_poller_ptr->Next(&face_count_packet)) {
    return absl::CancelledError(
        "Failed during getting next face_count_packet.");
  }

  auto &face_count_val = face_count_packet.Get<int>();

  if (face_count_val <= 0) {
    return absl::OkStatus();
  }

  // Get face bounding boxes.
  mediapipe::Packet face_rects_from_landmarks_packet;
  if (!face_rects_from_landmarks_poller_ptr ||
      !face_rects_from_landmarks_poller_ptr->Next(
          &face_rects_from_landmarks_packet)) {
    return absl::CancelledError(
        "Failed during getting next face_rects_from_landmarks_packet.");
  }

  auto &face_bounding_boxes =
      face_rects_from_landmarks_packet
          .Get<::std::vector<::mediapipe::NormalizedRect>>();

  image_width = camera_frame.cols;
  image_height = camera_frame.rows;
  const auto image_width_f = static_cast<float>(image_width);
  const auto image_height_f = static_cast<float>(image_height);

  // Convert vector<NormalizedRect> (center based Rects) to cv::Rect*
  // (leftTop based Rects).
  for (int i = 0; i < face_count_val; ++i) {
    const auto &normalized_bounding_box = face_bounding_boxes[i];
    auto &bounding_box = multi_face_bounding_boxes[i];

    const auto width =
        static_cast<int>(normalized_bounding_box.width() * image_width_f);
    const auto height =
        static_cast<int>(normalized_bounding_box.height() * image_height_f);

    bounding_box.x =
        static_cast<int>(normalized_bounding_box.x_center() * image_width_f) -
        (width >> 1);
    bounding_box.y =
        static_cast<int>(normalized_bounding_box.y_center() * image_height_f) -
        (height >> 1);
    bounding_box.width = width;
    bounding_box.height = height;
  }

  // Get face landmarks.
  if (!landmarks_poller_ptr ||
      !landmarks_poller_ptr->Next(&face_landmarks_packet)) {
    return absl::CancelledError("Failed during getting next landmarks_packet.");
  }

  *numFaces = face_count_val;
  face_count = face_count_val;

  return absl::OkStatus();
}

void MPFaceMeshDetector::DetectFaces(const cv::Mat &camera_frame,
                                     cv::Rect *multi_face_bounding_boxes,
                                     int *numFaces) {
  const auto status =
      DetectFacesWithStatus(camera_frame, multi_face_bounding_boxes, numFaces);
  if (!status.ok()) {
    LOG(INFO) << "MPFaceMeshDetector::DetectFaces failed: " << status.message();
  }
}
absl::Status MPFaceMeshDetector::DetectLandmarksWithStatus(
    cv::Point2f **multi_face_landmarks) {

  if (face_landmarks_packet.IsEmpty()) {
    return absl::CancelledError("Face landmarks packet is empty.");
  }

  auto &face_landmarks =
      face_landmarks_packet
          .Get<::std::vector<::mediapipe::NormalizedLandmarkList>>();

  const auto image_width_f = static_cast<float>(image_width);
  const auto image_height_f = static_cast<float>(image_height);

  // Convert landmarks to cv::Point2f**.
  for (int i = 0; i < face_count; ++i) {
    const auto &normalizedLandmarkList = face_landmarks[i];
    const auto landmarks_num = normalizedLandmarkList.landmark_size();

    if (landmarks_num != kLandmarksNum) {
      return absl::CancelledError("Detected unexpected landmarks number.");
    }

    auto &face_landmarks = multi_face_landmarks[i];

    for (int j = 0; j < landmarks_num; ++j) {
      const auto &landmark = normalizedLandmarkList.landmark(j);
      face_landmarks[j].x = landmark.x() * image_width_f;
      face_landmarks[j].y = landmark.y() * image_height_f;
    }
  }

  return absl::OkStatus();
}

absl::Status MPFaceMeshDetector::DetectLandmarksWithStatus(
    cv::Point3f **multi_face_landmarks) {

  if (face_landmarks_packet.IsEmpty()) {
    return absl::CancelledError("Face landmarks packet is empty.");
  }

  auto &face_landmarks =
      face_landmarks_packet
          .Get<::std::vector<::mediapipe::NormalizedLandmarkList>>();

  const auto image_width_f = static_cast<float>(image_width);
  const auto image_height_f = static_cast<float>(image_height);

  // Convert landmarks to cv::Point3f**.
  for (int i = 0; i < face_count; ++i) {
    const auto &normalized_landmark_list = face_landmarks[i];
    const auto landmarks_num = normalized_landmark_list.landmark_size();

    if (landmarks_num != kLandmarksNum) {
      return absl::CancelledError("Detected unexpected landmarks number.");
    }

    auto &face_landmarks = multi_face_landmarks[i];

    for (int j = 0; j < landmarks_num; ++j) {
      const auto &landmark = normalized_landmark_list.landmark(j);
      face_landmarks[j].x = landmark.x() * image_width_f;
      face_landmarks[j].y = landmark.y() * image_height_f;
      face_landmarks[j].z = landmark.z();
    }
  }

  return absl::OkStatus();
}

void MPFaceMeshDetector::DetectLandmarks(cv::Point2f **multi_face_landmarks,
                                         int *numFaces) {
  *numFaces = 0;
  const auto status = DetectLandmarksWithStatus(multi_face_landmarks);
  if (!status.ok()) {
    LOG(INFO) << "MPFaceMeshDetector::DetectLandmarks failed: "
              << status.message();
  }
  *numFaces = face_count;
}

void MPFaceMeshDetector::DetectLandmarks(cv::Point3f **multi_face_landmarks,
                                         int *numFaces) {
  *numFaces = 0;
  const auto status = DetectLandmarksWithStatus(multi_face_landmarks);
  if (!status.ok()) {
    LOG(INFO) << "MPFaceMeshDetector::DetectLandmarks failed: "
              << status.message();
  }
  *numFaces = face_count;
}

extern "C" {
MPFaceMeshDetector *
MPFaceMeshDetectorConstruct(int numFaces, const char *face_detection_model_path,
                            const char *face_landmark_model_path) {
  return new MPFaceMeshDetector(numFaces, face_detection_model_path,
                                face_landmark_model_path);
}

void MPFaceMeshDetectorDestruct(MPFaceMeshDetector *detector) {
  delete detector;
}

void MPFaceMeshDetectorDetectFaces(
    MPFaceMeshDetector *detector, const cv::Mat &camera_frame,
    cv::Rect *multi_face_bounding_boxes, int *numFaces) {
  detector->DetectFaces(camera_frame, multi_face_bounding_boxes, numFaces);
}
void
MPFaceMeshDetectorDetect2DLandmarks(MPFaceMeshDetector *detector,
                                    cv::Point2f **multi_face_landmarks,
                                    int *numFaces) {
  detector->DetectLandmarks(multi_face_landmarks, numFaces);
}
void
MPFaceMeshDetectorDetect3DLandmarks(MPFaceMeshDetector *detector,
                                    cv::Point3f **multi_face_landmarks,
                                    int *numFaces) {
  detector->DetectLandmarks(multi_face_landmarks, numFaces);
}

const int MPFaceMeshDetectorLandmarksNum =
    MPFaceMeshDetector::kLandmarksNum;
}
