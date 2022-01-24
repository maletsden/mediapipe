#include "face_mesh_lib.h"
#include "canonical_mesh.h"

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  cv::VideoCapture capture;
  capture.open(0);
  if (!capture.isOpened()) {
    return EXIT_FAILURE;
  }

  constexpr char kWindowName[] = "MediaPipe";

  cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  capture.set(cv::CAP_PROP_FPS, 30);
#endif

  LOG(INFO) << "VideoCapture initialized.";
  MPFaceMeshParameterList parameters;
  parameters.numFaces = 1;
  parameters.with_attention = true;
  parameters.face_detection_model_path = "mediapipe/modules/face_detection/face_detection_short_range.tflite";
  parameters.face_landmark_model_path = "mediapipe/modules/face_landmark/face_landmark.tflite";
  parameters.face_landmark_model_with_attention_path = "mediapipe/modules/face_landmark/face_landmark_with_attention.tflite";
  parameters.window_size_param = 10;
  parameters.velocity_scale_param = 10;
  // Maximum number of faces that can be detected
  /*constexpr int maxNumFaces = 1;
  constexpr char face_detection_model_path[] =
      "mediapipe/modules/face_detection/face_detection_short_range.tflite";
  constexpr char face_landmark_model_path[] =
      "mediapipe/modules/face_landmark/face_landmark.tflite";
  constexpr char face_landmark_with_attention_model_path[] =
      "mediapipe/modules/face_landmark/face_landmark_with_attention.tflite";
  constexpr char geometry_pipeline_metadata_landmarks_path[] =
      "mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_landmarks.binarypb";
  constexpr bool with_attention = true;*/

  //double f_x = 640;
  //double f_y = 640;
  //double c_x = 320;
  //double c_y = 240;
  //cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << f_x, 0.0, c_x, 0.0, f_y, c_y, 0.0, 0.0, 1.0);

  MPFaceMeshDetector* faceMeshDetector = MPFaceMeshDetectorConstruct(parameters);

  // Allocate memory for face landmarks.
  auto multiFaceLandmarks = new cv::Point2f *[parameters.numFaces];
  for (int i = 0; i < parameters.numFaces; ++i) {
    multiFaceLandmarks[i] = new cv::Point2f[MPFaceMeshDetectorLandmarksNum];
  }

  std::vector<cv::Rect> multiFaceBoundingBoxes(parameters.numFaces);
  //std::vector<cv::Mat> multiFacePoses(maxNumFaces, cv::Mat::zeros(4, 4, CV_64F));

  LOG(INFO) << "FaceMeshDetector constructed.";

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;

  //cv::Mat transl_y = (cv::Mat_<double>(4, 4) << 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  //cv::Mat transl_z = (cv::Mat_<double>(4, 4) << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  //cv::Mat transl_x = (cv::Mat_<double>(4, 4) << -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  int fps = 30;
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
    	multiFaceBoundingBoxes.data(), fps, &faceCount);

	if (faceCount > 0) {
      auto& face_bounding_box = multiFaceBoundingBoxes[0];

      cv::rectangle(camera_frame_raw, face_bounding_box, cv::Scalar(0, 255, 0), 3);

      int landmarksNum = 0;
      MPFaceMeshDetectorDetect2DLandmarks(faceMeshDetector, multiFaceLandmarks,
    	&landmarksNum);
      auto& face_landmarks = multiFaceLandmarks[0];
      auto& landmark = face_landmarks[0];

	  int numFaces = 0;
	  //MPFaceMeshDetectorDetectFacePoses(faceMeshDetector, multiFacePoses.data(), &numFaces);

	  for (auto i = 0; i < 478; ++i) {
	   	cv::circle(camera_frame_raw, face_landmarks[i], 1.2, cv::Scalar(0, 0, 255));
	  }

	  /*auto projectPoint = [&](auto& p) -> cv::Point2f
	  {
		cv::Mat point = (cv::Mat_<double>(4, 1) << p.x, p.y, p.z, 1);
		cv::Mat imp = transl_x * multiFacePoses[0] * point;
		return cv::Point2f((((imp.at<double>(0, 0) / imp.at<double>(2, 0)) + 1) / 2) * 640, (((imp.at<double>(1, 0) / imp.at<double>(2, 0)) + 0.75) / 1.5) * 480);
	  };

	  if (!multiFacePoses.empty()) {
		for (auto i = 0; i < canonical::mesh.size(); ++i) {
			cv::circle(camera_frame_raw, projectPoint(canonical::mesh[i]), 1.2, cv::Scalar(0, 0, 255));
		}
	  }*/

	  LOG(INFO) << "First landmark: x - " << landmark.x << ", y - "
	   	<< landmark.y;

	  /*if (!multiFacePoses.empty()) {
		cv::Mat projected_point = transl_x * multiFacePoses[0].inv() * point;
		auto new_point = cv::Point2f(projected_point.at<double>(0, 0) / projected_point.at<double>(2, 0), projected_point.at<double>(1, 0) / projected_point.at<double>(2, 0));
		std::cout << "Projected First landmark: x - " << ((new_point.x + 1) / 2) * 640 << ", y - "
	      << ((new_point.y + 1) / 2) * 480 << "\n";
		std::cout << "Projected First landmark(0): x - " << new_point.x << ", y - "
  		  << new_point.y << "\n";
	  }*/
	  /*std::cout << "Second landmark: x - " << face_landmarks[1].x << ", y - "
		<< face_landmarks[1].y;
	  std::cout << "Third landmark: x - " << face_landmarks[2].x << ", y - "
		<< face_landmarks[2].y;
	  auto printRow = [&](int i) -> void
  	  {
		std::cout  << "{ " << multiFacePoses[0].at<double>(i, 0) << ", " << multiFacePoses[0].at<double>(i, 1) << ", "
		  << multiFacePoses[0].at<double>(i, 2) << ", "
		  << multiFacePoses[0].at<double>(i, 3) << " }";
	  };

	  if (!multiFacePoses.empty())
  	  {
		for (int i = 0; i < 4; ++i)
		{
	  	  printRow(i);
		}
		std::cout << "\n";
	  }*/
    }

    const int pressed_key = cv::waitKey(5);
    if (pressed_key >= 0 && pressed_key != 255)
      grab_frames = false;

    cv::imshow(kWindowName, camera_frame_raw);
  }

  LOG(INFO) << "Shutting down.";

  // Deallocate memory for face landmarks.
  for (int i = 0; i < parameters.numFaces; ++i) {
    delete[] multiFaceLandmarks[i];
  }
  delete[] multiFaceLandmarks;

  MPFaceMeshDetectorDestruct(faceMeshDetector);
}