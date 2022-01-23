#include "face_mesh_lib.h"
#include "face_mesh_graph.h"
#include "Eigen/Core"

int MPFaceMeshDetector::kLandmarksNum = 468;

MPFaceMeshDetector::MPFaceMeshDetector(int numFaces,
    bool with_attention,
    /*cv::Mat cameraMatrix,*/
    const char* face_detection_model_path,
    const char* face_landmark_model_path,
    const char* face_landmark_with_attention_model_path
    /*const char* geometry_pipeline_metadata_landmarks_path*/,
    int window_size_param,
    float velocity_scale_param) {
    const auto status = InitFaceMeshDetector(
        numFaces,
        /*cameraMatrix,*/
        with_attention,
        face_detection_model_path,
        face_landmark_model_path,
        face_landmark_with_attention_model_path
        /*geometry_pipeline_metadata_landmarks_path*/);
    if (!status.ok()) {
        LOG(INFO) << "Failed constructing FaceMeshDetector.";
        LOG(INFO) << status.message();
    }
    if (with_attention) {
        kLandmarksNum = kLandmarksNumWithAttention;
    }
}

absl::Status
MPFaceMeshDetector::InitFaceMeshDetector(int numFaces,
    /*cv::Mat cameraMatrix,*/
    bool with_attention,
    const char* face_detection_model_path,
    const char* face_landmark_model_path,
    const char* face_landmark_with_attention_model_path
    /*const char* geometry_pipeline_metadata_landmarks_path*/
    int window_size_param,
    float velocity_scale_param) {
    numFaces = std::max(numFaces, 1);
    /*m_cameraMatrix = cameraMatrix.clone();*/
    const char* SplitTensorVectorCalculator =
    R""""(
    # Splits a vector of tensors into landmark tensors and face flag tensor.
    node {
        calculator: "SplitTensorVectorCalculator"
        input_stream: "face_landmark_side_model_cpu__output_tensors"
        output_stream: "face_landmark_side_model_cpu__landmark_tensors"
        output_stream: "face_landmark_side_model_cpu__face_flag_tensor"
            options: {
              [mediapipe.SplitVectorCalculatorOptions.ext] {
                ranges: { begin: 0 end: 1 }
                ranges: { begin: 1 end: 2 }
              }
            }
    }
    )"""";
    const char* SplitTensorVectorCalculatorWithAttention =
    R""""(
        # Splits a vector of tensors into landmark tensors and face flag tensor.
        node {
            calculator: "SplitTensorVectorCalculator"
            input_stream: "face_landmark_side_model_cpu__output_tensors"
            output_stream: "face_landmark_side_model_cpu__landmark_tensors"
            output_stream: "face_landmark_side_model_cpu__face_flag_tensor"
                options: {
                  [mediapipe.SplitVectorCalculatorOptions.ext] {
                    ranges: { begin: 0 end: 6 }
                    ranges: { begin: 6 end: 7 }
                  }
                }
        }
    )"""";
    const char* TensorsToLandmarksSubgraph =
    R""""(
    # Decodes the landmark tensors into a vector of lanmarks, where the landmark
    # coordinates are normalized by the size of the input image to the model.
    node {
      calculator: "TensorsToLandmarksCalculator"
      input_stream: "TENSORS:face_landmark_side_model_cpu__ensured_landmark_tensors"
      output_stream: "NORM_LANDMARKS:face_landmark_side_model_cpu__landmarks"
      options: {
        [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
          num_landmarks: 468
          input_image_width: 192
          input_image_height: 192
        }
      }
    }
    )"""";
    
    const char* TensorsToLandmarksWithAttentionSubgraph =
    R""""(
    # MediaPipe graph to transform model output tensors into 478 facial landmarks
    # with refined lips, eyes and irises.
    # Splits a vector of tensors into multiple vectors.
    node {
      calculator: "SplitTensorVectorCalculator"
      input_stream: "face_landmark_side_model_cpu__ensured_landmark_tensors"
      output_stream: "mesh_tensor"
      output_stream: "lips_tensor"
      output_stream: "left_eye_tensor"
      output_stream: "right_eye_tensor"
      output_stream: "left_iris_tensor"
      output_stream: "right_iris_tensor"
      options: {
        [mediapipe.SplitVectorCalculatorOptions.ext] {
          ranges: { begin: 0 end: 1 }
          ranges: { begin: 1 end: 2 }
          ranges: { begin: 2 end: 3 }
          ranges: { begin: 3 end: 4 }
          ranges: { begin: 4 end: 5 }
          ranges: { begin: 5 end: 6 }
        }
      }
    }

    # Decodes mesh landmarks tensor into a vector of normalized lanmarks.
    node {
      calculator: "TensorsToLandmarksCalculator"
      input_stream: "TENSORS:mesh_tensor"
      output_stream: "NORM_LANDMARKS:mesh_landmarks"
      options: {
        [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
          num_landmarks: 468
          input_image_width: 192
          input_image_height: 192
        }
      }
    }

    # Decodes lips landmarks tensor into a vector of normalized lanmarks.
    node {
      calculator: "TensorsToLandmarksCalculator"
      input_stream: "TENSORS:lips_tensor"
      output_stream: "NORM_LANDMARKS:lips_landmarks"
      options: {
        [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
          num_landmarks: 80
          input_image_width: 192
          input_image_height: 192
        }
      }
    }

    # Decodes left eye landmarks tensor into a vector of normalized lanmarks.
    node {
      calculator: "TensorsToLandmarksCalculator"
      input_stream: "TENSORS:left_eye_tensor"
      output_stream: "NORM_LANDMARKS:left_eye_landmarks"
      options: {
        [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
          num_landmarks: 71
          input_image_width: 192
          input_image_height: 192
        }
      }
    }

    # Decodes right eye landmarks tensor into a vector of normalized lanmarks.
    node {
      calculator: "TensorsToLandmarksCalculator"
      input_stream: "TENSORS:right_eye_tensor"
      output_stream: "NORM_LANDMARKS:right_eye_landmarks"
      options: {
        [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
          num_landmarks: 71
          input_image_width: 192
          input_image_height: 192
        }
      }
    }

    # Decodes left iris landmarks tensor into a vector of normalized lanmarks.
    node {
      calculator: "TensorsToLandmarksCalculator"
      input_stream: "TENSORS:left_iris_tensor"
      output_stream: "NORM_LANDMARKS:left_iris_landmarks"
      options: {
        [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
          num_landmarks: 5
          input_image_width: 192
          input_image_height: 192
        }
      }
    }

    # Decodes right iris landmarks tensor into a vector of normalized lanmarks.
    node {
      calculator: "TensorsToLandmarksCalculator"
      input_stream: "TENSORS:right_iris_tensor"
      output_stream: "NORM_LANDMARKS:right_iris_landmarks"
      options: {
        [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
          num_landmarks: 5
          input_image_width: 192
          input_image_height: 192
        }
      }
    }

    # Refine mesh landmarks with lips, eyes and irises.
    node {
      calculator: "LandmarksRefinementCalculator"
      input_stream: "LANDMARKS:0:mesh_landmarks"
      input_stream: "LANDMARKS:1:lips_landmarks"
      input_stream: "LANDMARKS:2:left_eye_landmarks"
      input_stream: "LANDMARKS:3:right_eye_landmarks"
      input_stream: "LANDMARKS:4:left_iris_landmarks"
      input_stream: "LANDMARKS:5:right_iris_landmarks"
      output_stream: "REFINED_LANDMARKS:face_landmark_side_model_cpu__landmarks"
      options: {
        [mediapipe.LandmarksRefinementCalculatorOptions.ext] {
          # 0 - mesh
          refinement: {
            indexes_mapping: [
              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
              20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
              37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
              54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
              71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
              88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
              104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
              118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
              132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
              146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
              160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
              174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
              188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201,
              202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
              216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229,
              230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
              244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257,
              258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
              272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
              286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,
              300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313,
              314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327,
              328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341,
              342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355,
              356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369,
              370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383,
              384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397,
              398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411,
              412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425,
              426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
              440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453,
              454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467
            ]
            z_refinement: { copy {} }
          }
          # 1 - lips
          refinement: {
            indexes_mapping: [
              # Lower outer.
              61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              # Upper outer (excluding corners).
              185, 40, 39, 37, 0, 267, 269, 270, 409,
              # Lower inner.
              78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
              # Upper inner (excluding corners).
              191, 80, 81, 82, 13, 312, 311, 310, 415,
              # Lower semi-outer.
              76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306,
              # Upper semi-outer (excluding corners).
              184, 74, 73, 72, 11, 302, 303, 304, 408,
              # Lower semi-inner.
              62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292,
              # Upper semi-inner (excluding corners).
              183, 42, 41, 38, 12, 268, 271, 272, 407
            ]
            z_refinement: { none {} }
          }
          # 2 - left eye
          refinement: {
            indexes_mapping: [
              # Lower contour.
              33, 7, 163, 144, 145, 153, 154, 155, 133,
              # upper contour (excluding corners).
              246, 161, 160, 159, 158, 157, 173,
              # Halo x2 lower contour.
              130, 25, 110, 24, 23, 22, 26, 112, 243,
              # Halo x2 upper contour (excluding corners).
              247, 30, 29, 27, 28, 56, 190,
              # Halo x3 lower contour.
              226, 31, 228, 229, 230, 231, 232, 233, 244,
              # Halo x3 upper contour (excluding corners).
              113, 225, 224, 223, 222, 221, 189,
              # Halo x4 upper contour (no lower because of mesh structure) or
              # eyebrow inner contour.
              35, 124, 46, 53, 52, 65,
              # Halo x5 lower contour.
              143, 111, 117, 118, 119, 120, 121, 128, 245,
              # Halo x5 upper contour (excluding corners) or eyebrow outer contour.
              156, 70, 63, 105, 66, 107, 55, 193
            ]
            z_refinement: { none {} }
          }
          # 3 - right eye
          refinement: {
            indexes_mapping: [
              # Lower contour.
              263, 249, 390, 373, 374, 380, 381, 382, 362,
              # Upper contour (excluding corners).
              466, 388, 387, 386, 385, 384, 398,
              # Halo x2 lower contour.
              359, 255, 339, 254, 253, 252, 256, 341, 463,
              # Halo x2 upper contour (excluding corners).
              467, 260, 259, 257, 258, 286, 414,
              # Halo x3 lower contour.
              446, 261, 448, 449, 450, 451, 452, 453, 464,
              # Halo x3 upper contour (excluding corners).
              342, 445, 444, 443, 442, 441, 413,
              # Halo x4 upper contour (no lower because of mesh structure) or
              # eyebrow inner contour.
              265, 353, 276, 283, 282, 295,
              # Halo x5 lower contour.
              372, 340, 346, 347, 348, 349, 350, 357, 465,
              # Halo x5 upper contour (excluding corners) or eyebrow outer contour.
              383, 300, 293, 334, 296, 336, 285, 417
            ]
            z_refinement: { none {} }
          }
          # 4 - left iris
          refinement: {
            indexes_mapping: [
              # Center.
              468,
              # Iris right edge.
              469,
              # Iris top edge.
              470,
              # Iris left edge.
              471,
              # Iris bottom edge.
              472
            ]
            z_refinement: {
              assign_average: {
                indexes_for_average: [
                  # Lower contour.
                  33, 7, 163, 144, 145, 153, 154, 155, 133,
                  # Upper contour (excluding corners).
                  246, 161, 160, 159, 158, 157, 173
                ]
              }
            }
          }
          # 5 - right iris
          refinement: {
            indexes_mapping: [
              # Center.
              473,
              # Iris right edge.
              474,
              # Iris top edge.
              475,
              # Iris left edge.
              476,
              # Iris bottom edge.
              477
            ]
            z_refinement: {
              assign_average: {
                indexes_for_average: [
                  # Lower contour.
                  263, 249, 390, 373, 374, 380, 381, 382, 362,
                  # Upper contour (excluding corners).
                  466, 388, 387, 386, 385, 384, 398
                ]
              }
            }
          }
        }
      }
    }
    )"""";
    
    if (with_attention) {
        face_landmark_model_path = face_landmark_with_attention_model_path;
    }
    std::string TensorsToLandmarksStr = "$TensorsToLandmarksSubgraph";
    std::string SplitTensorVectorStr = "$SplitTensorVectorCalculator";
    std::string WindowSizeStr = "$window_size_param";
    std::string VelocityScaleStr = "$velocity_scale_param";
    //std::string GeometryPipelineMetadataLandmarksPathStr = "$geometryPipelineMetadataLandmarksPath";
    auto preparedGraphConfig = face_mesh_graph::graphConfig.replace(face_mesh_graph::graphConfig.find(TensorsToLandmarksStr),
     TensorsToLandmarksStr.size(), with_attention ? TensorsToLandmarksWithAttentionSubgraph : TensorsToLandmarksSubgraph);
    preparedGraphConfig = preparedGraphConfig.replace(preparedGraphConfig.find(SplitTensorVectorStr),
     SplitTensorVectorStr.size(), with_attention ? SplitTensorVectorCalculatorWithAttention : SplitTensorVectorCalculator);
    //preparedGraphConfig = preparedGraphConfig.replace(preparedGraphConfig.find(GeometryPipelineMetadataLandmarksPathStr),
    // GeometryPipelineMetadataLandmarksPathStr.size(), geometry_pipeline_metadata_landmarks_path);
    preparedGraphConfig = preparedGraphConfig.replace(preparedGraphConfig.find(WindowSizeStr),
     WindowSizeStr.size(), window_size_param);
    preparedGraphConfig = preparedGraphConfig.replace(preparedGraphConfig.find(VelocityScaleStr),
     VelocityScaleStr.size(), velocity_scale_param);
    LOG(INFO) << "Get calculator graph config contents: " << preparedGraphConfig;
    
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(preparedGraphConfig);
    LOG(INFO) << "Initialize the calculator graph.";
    
    // Prepare graph config.
    std::map<std::string, mediapipe::Packet> extra_side_packets;
    extra_side_packets.insert({ "num_faces", mediapipe::MakePacket<int>(std::max(numFaces, 1)) });
    extra_side_packets.insert({ "with_attention", mediapipe::MakePacket<bool>(with_attention) });
    extra_side_packets.insert({ "face_detection_model_path", mediapipe::MakePacket<std::string>(face_detection_model_path) });
    extra_side_packets.insert({ "face_landmark_model_path", mediapipe::MakePacket<std::string>(face_landmark_model_path) });

    MP_RETURN_IF_ERROR(graph.Initialize(config, extra_side_packets));

    LOG(INFO) << "Start running the calculator graph.";

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarks_poller,
        graph.AddOutputStreamPoller(kOutputStream_landmarks));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller face_count_poller,
        graph.AddOutputStreamPoller(kOutputStream_faceCount));
    ASSIGN_OR_RETURN(
        mediapipe::OutputStreamPoller face_rects_from_landmarks_poller,
        graph.AddOutputStreamPoller(kOutputStream_face_rects_from_landmarks));
//    ASSIGN_OR_RETURN(
//            mediapipe::OutputStreamPoller poses_poller,
//            graph.AddOutputStreamPoller(kOutputStream_poses));
    
    landmarks_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
        std::move(landmarks_poller));
    face_count_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
        std::move(face_count_poller));
    face_rects_from_landmarks_poller_ptr =
        std::make_unique<mediapipe::OutputStreamPoller>(
            std::move(face_rects_from_landmarks_poller));
//    poses_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
//            std::move(poses_poller));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    LOG(INFO) << "MPFaceMeshDetector constructed successfully.";

    return absl::OkStatus();
}

absl::Status
MPFaceMeshDetector::DetectFacesWithStatus(const cv::Mat& camera_frame,
    cv::Rect* multi_face_bounding_boxes,
    int fps,
    int* numFaces) {
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
    // Send calibration packet into the graph.
    //MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
    //    kInputStream_camera_matrix, mediapipe::MakePacket<cv::Mat>(m_cameraMatrix)
    //    .At(mediapipe::Timestamp(frame_timestamp_us))));
    // Send fps packet into the graph.
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream_fps, mediapipe::MakePacket<int>(fps)
        .At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get face count.
    mediapipe::Packet face_count_packet;
    if (!face_count_poller_ptr ||
        !face_count_poller_ptr->Next(&face_count_packet)) {
        return absl::CancelledError(
            "Failed during getting next face_count_packet.");
    }

    auto& face_count_val = face_count_packet.Get<int>();

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

    auto& face_bounding_boxes =
        face_rects_from_landmarks_packet
        .Get<::std::vector<::mediapipe::NormalizedRect>>();

    image_width = camera_frame.cols;
    image_height = camera_frame.rows;
    const auto image_width_f = static_cast<float>(image_width);
    const auto image_height_f = static_cast<float>(image_height);

    // Convert vector<NormalizedRect> (center based Rects) to cv::Rect*
    // (leftTop based Rects).
    for (int i = 0; i < face_count_val; ++i) {
        const auto& normalized_bounding_box = face_bounding_boxes[i];
        auto& bounding_box = multi_face_bounding_boxes[i];

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
    
    // Get face poses.
    //if (!poses_poller_ptr ||
    //    !poses_poller_ptr->Next(&poses_packet)) {
    //    return absl::CancelledError(
    //        "Failed during getting next poses_packet.");
    //}

    // Get face landmarks.
    if (!landmarks_poller_ptr ||
        !landmarks_poller_ptr->Next(&face_landmarks_packet)) {
        return absl::CancelledError("Failed during getting next landmarks_packet.");
    }

    *numFaces = face_count_val;
    face_count = face_count_val;

    return absl::OkStatus();
}

void MPFaceMeshDetector::DetectFaces(const cv::Mat& camera_frame,
    cv::Rect* multi_face_bounding_boxes,
    int fps,
    int* numFaces) {
    const auto status =
        DetectFacesWithStatus(camera_frame, multi_face_bounding_boxes, fps, numFaces);
    if (!status.ok()) {
        LOG(INFO) << "MPFaceMeshDetector::DetectFaces failed: " << status.message();
    }
}

//absl::Status MPFaceMeshDetector::DetectFacePosesWithStatus(cv::Mat* multi_face_poses) {
//  if (poses_packet.IsEmpty()) {
//      return absl::CancelledError("Face poses packet is empty.");
//  }
//  if (!multi_face_poses) {
//      return absl::InvalidArgumentError(
//          "MPFaceMeshDetector::DetectFacesWithStatus requires notnull pointer to "
//          "save results data.");
//  }
//  auto& face_poses = poses_packet.Get<::std::vector<Eigen::Matrix4f>>();
//  for (int i = 0; i < face_count; ++i) {
//      for (int k = 0; k < 4; ++k) {
//          for (int j = 0; j < 4; ++j) {
//              multi_face_poses[i].at<double>(k, j) = face_poses[i](k, j);
//          }
//      }
//  }
//
//    return absl::OkStatus();
//}
//
//void MPFaceMeshDetector::DetectFacePoses(cv::Mat* multi_face_poses, int* numFaces) {
//  *numFaces = 0;
//  const auto status = DetectFacePosesWithStatus(multi_face_poses);
//  if (!status.ok()) {
//      LOG(INFO) << "MPFaceMeshDetector::DetectFacePoses failed: "
//          << status.message();
//  }
//  *numFaces = face_count;
//}

absl::Status MPFaceMeshDetector::DetectLandmarksWithStatus(
    cv::Point2f** multi_face_landmarks) {

    if (face_landmarks_packet.IsEmpty()) {
        return absl::CancelledError("Face landmarks packet is empty.");
    }

    auto& face_landmarks =
        face_landmarks_packet
        .Get<::std::vector<::mediapipe::NormalizedLandmarkList>>();

    const auto image_width_f = static_cast<float>(image_width);
    const auto image_height_f = static_cast<float>(image_height);

    // Convert landmarks to cv::Point2f**.
    for (int i = 0; i < face_count; ++i) {
        const auto& normalizedLandmarkList = face_landmarks[i];
        const auto landmarks_num = normalizedLandmarkList.landmark_size();

        if (landmarks_num != kLandmarksNum) {
            return absl::CancelledError("Detected unexpected landmarks number.");
        }

        auto& face_landmarks = multi_face_landmarks[i];

        for (int j = 0; j < landmarks_num; ++j) {
            const auto& landmark = normalizedLandmarkList.landmark(j);
            face_landmarks[j].x = landmark.x() * image_width_f;
            face_landmarks[j].y = landmark.y() * image_height_f;
        }
    }

    return absl::OkStatus();
}

absl::Status MPFaceMeshDetector::DetectLandmarksWithStatus(
    cv::Point3f** multi_face_landmarks) {

    if (face_landmarks_packet.IsEmpty()) {
        return absl::CancelledError("Face landmarks packet is empty.");
    }

    auto& face_landmarks =
        face_landmarks_packet
        .Get<::std::vector<::mediapipe::NormalizedLandmarkList>>();

    const auto image_width_f = static_cast<float>(image_width);
    const auto image_height_f = static_cast<float>(image_height);

    // Convert landmarks to cv::Point3f**.
    for (int i = 0; i < face_count; ++i) {
        const auto& normalized_landmark_list = face_landmarks[i];
        const auto landmarks_num = normalized_landmark_list.landmark_size();

        if (landmarks_num != kLandmarksNum) {
            return absl::CancelledError("Detected unexpected landmarks number.");
        }

        auto& face_landmarks = multi_face_landmarks[i];

        for (int j = 0; j < landmarks_num; ++j) {
            const auto& landmark = normalized_landmark_list.landmark(j);
            face_landmarks[j].x = landmark.x() * image_width_f;
            face_landmarks[j].y = landmark.y() * image_height_f;
            face_landmarks[j].z = landmark.z();
        }
    }

    return absl::OkStatus();
}

void MPFaceMeshDetector::DetectLandmarks(cv::Point2f** multi_face_landmarks,
    int* numFaces) {
    *numFaces = 0;
    const auto status = DetectLandmarksWithStatus(multi_face_landmarks);
    if (!status.ok()) {
        LOG(INFO) << "MPFaceMeshDetector::DetectLandmarks failed: "
            << status.message();
    }
    *numFaces = face_count;
}

void MPFaceMeshDetector::DetectLandmarks(cv::Point3f** multi_face_landmarks,
    int* numFaces) {
    *numFaces = 0;
    const auto status = DetectLandmarksWithStatus(multi_face_landmarks);
    if (!status.ok()) {
        LOG(INFO) << "MPFaceMeshDetector::DetectLandmarks failed: "
            << status.message();
    }
    *numFaces = face_count;
}

extern "C" {
    MPFaceMeshDetector* MPFaceMeshDetectorConstruct(int numFaces,
        /*cv::Mat cameraMatrix,*/
        bool with_attention,
        const char* face_detection_model_path,
        const char* face_landmark_model_path,
        const char* face_landmark_model_with_attention_path
        /*const char* geometry_pipeline_metadata_landmarks_path*/) {
        return new MPFaceMeshDetector(numFaces, /*cameraMatrix,*/ with_attention, face_detection_model_path,
            face_landmark_model_path, face_landmark_model_with_attention_path
            /*geometry_pipeline_metadata_landmarks_path*/);
    }

    void MPFaceMeshDetectorDestruct(MPFaceMeshDetector* detector) {
        delete detector;
    }

    void MPFaceMeshDetectorDetectFaces(
        MPFaceMeshDetector* detector, const cv::Mat& camera_frame,
        cv::Rect* multi_face_bounding_boxes, int fps, int* numFaces) {
        detector->DetectFaces(camera_frame, multi_face_bounding_boxes, fps, numFaces);
    }
    void MPFaceMeshDetectorDetect2DLandmarks(MPFaceMeshDetector* detector,
        cv::Point2f** multi_face_landmarks,
        int* numFaces) {
        detector->DetectLandmarks(multi_face_landmarks, numFaces);
    }
    void MPFaceMeshDetectorDetect3DLandmarks(MPFaceMeshDetector* detector,
        cv::Point3f** multi_face_landmarks,
        int* numFaces) {
        detector->DetectLandmarks(multi_face_landmarks, numFaces);
    }

    const int MPFaceMeshDetectorLandmarksNum =
        MPFaceMeshDetector::kLandmarksNum;
}
