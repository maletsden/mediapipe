#include "face_mesh_lib.h"

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  cv::VideoCapture capture;
  capture.open(0);
  if (!capture.isOpened()) {
    return -1;
  }

  constexpr char kWindowName[] = "MediaPipe";

  cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  capture.set(cv::CAP_PROP_FPS, 30);
#endif

  LOG(INFO) << "VideoCapture initialized.";

  // Maximum number of faces that can be detected
  constexpr int maxNumFaces = 1;
  constexpr char face_detection_model_path[] = "mediapipe/modules/face_detection/face_detection_short_range.tflite";
  constexpr char face_landmark_model_path[] = "mediapipe/modules/face_landmark/face_landmark.tflite";

  MPFaceMeshDetector *faceMeshDetector = MPFaceMeshDetectorConstruct(maxNumFaces,face_landmark_model_path);

  // Allocate memory for face landmarks.
  auto multiFaceLandmarks = new cv::Point2f *[maxNumFaces];
  for (int i = 0; i < maxNumFaces; ++i) {
    multiFaceLandmarks[i] = new cv::Point2f[MPFaceMeshDetectorLandmarksNum];
  }

  std::vector<cv::Rect> multiFaceBoundingBoxes(maxNumFaces);

  LOG(INFO) << "FaceMeshDetector constructed.";

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;

  while (grab_frames) {
    // Capture opencv camera.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      LOG(INFO) << "Ignore empty frames from camera.";
      continue;
    }

    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    int faceCount = 0;

    MPFaceMeshDetectorDetectFaces(faceMeshDetector, camera_frame,
                                  multiFaceBoundingBoxes.data(), &faceCount);

    if (faceCount > 0) {
      int landmarksFaceCount = 0;
      MPFaceMeshDetectorDetect2DLandmarks(faceMeshDetector, multiFaceLandmarks, &landmarksFaceCount);

      if (landmarksFaceCount != faceCount) {
        LOG(ERROR) << "Face count differ between MPFaceMeshDetectorDetectFaces and"
                      "MPFaceMeshDetectorDetect2DLandmarks.";
      }

      for (int faceIdx = 0; faceIdx < faceCount; ++faceIdx) {
        // Draw bounding box.
        cv::rectangle(camera_frame_raw, multiFaceBoundingBoxes[faceIdx], cv::Scalar(0, 255, 0), 3);

        // Draw face landmarks.
        const auto faceLandmarks = multiFaceLandmarks[faceIdx];
        for (int landmarkIdx = 0; landmarkIdx < MPFaceMeshDetectorLandmarksNum; ++landmarkIdx) {
          cv::circle(camera_frame_raw, faceLandmarks[landmarkIdx], 1, cv::Scalar(0, 0, 255), -1);
        }
      }
    }

    const int pressed_key = cv::waitKey(5);
    if (pressed_key >= 0 && pressed_key != 255) {
      grab_frames = false;
    }

    cv::imshow(kWindowName, camera_frame_raw);
  }

  LOG(INFO) << "Shutting down.";

  // Deallocate memory for face landmarks.
  for (int i = 0; i < maxNumFaces; ++i) {
    delete[] multiFaceLandmarks[i];
  }
  delete[] multiFaceLandmarks;

  MPFaceMeshDetectorDestruct(faceMeshDetector);
}