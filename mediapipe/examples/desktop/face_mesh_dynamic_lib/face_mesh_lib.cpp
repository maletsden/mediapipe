#include "face_mesh_lib.h"

extern "C" {

DLLEXPORT MPFaceMeshDetector * MPFaceMeshDetectorConstruct(int numFaces,
                                                           const char *face_detection_model_path,
                                                           const char *face_landmark_model_path) {
  return new MPFaceMeshDetector(numFaces, face_detection_model_path, face_landmark_model_path);
}

DLLEXPORT void MPFaceMeshDetectorDestruct(MPFaceMeshDetector *detector) {
  delete detector;
}

DLLEXPORT void MPFaceMeshDetectorDetectFaces(MPFaceMeshDetector *detector,
                                             const cv::Mat &camera_frame,
                                             cv::Rect *multi_face_bounding_boxes,
                                             int *numFaces) {
  detector->DetectFaces(camera_frame, multi_face_bounding_boxes, numFaces);
}

DLLEXPORT void MPFaceMeshDetectorDetect2DLandmarks(MPFaceMeshDetector *detector,
                                                   cv::Point2f **multi_face_landmarks,
                                                   int *numFaces) {
  detector->DetectLandmarks(multi_face_landmarks, numFaces);
}

DLLEXPORT void MPFaceMeshDetectorDetect3DLandmarks(MPFaceMeshDetector *detector,
                                                   cv::Point3f **multi_face_landmarks,
                                                   int *numFaces) {
  detector->DetectLandmarks(multi_face_landmarks, numFaces);
}

DLLEXPORT const int MPFaceMeshDetectorLandmarksNum =
    MPFaceMeshDetector::kLandmarksNum;
}

const std::string MPFaceMeshDetector::graphConfig = R"pb(
# MediaPipe graph that performs face mesh with TensorFlow Lite on CPU.

# Input image. (ImageFrame)
input_stream: "input_video"

# Collection of detected/processed faces, each represented as a list of
# landmarks. (std::vector<NormalizedLandmarkList>)
output_stream: "multi_face_landmarks"

# Detected faces count. (int)
output_stream: "face_count"

# Regions of interest calculated based on landmarks.
# (std::vector<NormalizedRect>)
output_stream: "face_rects_from_landmarks"

node {
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:face_count"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}

# Defines side packets for further use in the graph.
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:num_faces"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { int_value: $numFaces }
    }
  }
}

# Defines side packets for further use in the graph.
node {
    calculator: "ConstantSidePacketCalculator"
    output_side_packet: "PACKET:face_detection_model_path"
    options: {
        [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
            packet { string_value: "$faceDetectionModelPath" }
        }
    }
}

# Defines side packets for further use in the graph.
node {
    calculator: "ConstantSidePacketCalculator"
    output_side_packet: "PACKET:face_landmark_model_path"
    node_options: {
        [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
            packet { string_value: "$faceLandmarkModelPath" }
    }
  }
}

node {
    calculator: "LocalFileContentsCalculator"
    input_side_packet: "FILE_PATH:0:face_detection_model_path"
    input_side_packet: "FILE_PATH:1:face_landmark_model_path"
    output_side_packet: "CONTENTS:0:face_detection_model_blob"
    output_side_packet: "CONTENTS:1:face_landmark_model_blob"
}

node {
    calculator: "TfLiteModelCalculator"
    input_side_packet: "MODEL_BLOB:face_detection_model_blob"
    output_side_packet: "MODEL:face_detection_model"
}
node {
    calculator: "TfLiteModelCalculator"
    input_side_packet: "MODEL_BLOB:face_landmark_model_blob"
    output_side_packet: "MODEL:face_landmark_model"
}


# Subgraph that detects faces and corresponding landmarks.
node {
  calculator: "FaceLandmarkFrontSideModelCpuWithFaceCounter"
  input_stream: "IMAGE:throttled_input_video"
  input_side_packet: "NUM_FACES:num_faces"
  input_side_packet: "MODEL:0:face_detection_model"
  input_side_packet: "MODEL:1:face_landmark_model"
  output_stream: "LANDMARKS:multi_face_landmarks"
  output_stream: "ROIS_FROM_LANDMARKS:face_rects_from_landmarks"
  output_stream: "DETECTIONS:face_detections"
  output_stream: "ROIS_FROM_DETECTIONS:face_rects_from_detections"
  output_stream: "FACE_COUNT_FROM_LANDMARKS:face_count"
}

)pb";
