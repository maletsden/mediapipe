#ifndef FACE_MESH_GRAPH_H
#define FACE_MESH_GRAPH_H

#include <string>
#include <typeinfo>

#include "mediapipe/calculators/core/flow_limiter_calculator.h"
#include "mediapipe/calculators/core/begin_loop_calculator.h"
#include "mediapipe/calculators/core/constant_side_packet_calculator.h"
#include "mediapipe/calculators/core/clip_vector_size_calculator.h"
#include "mediapipe/calculators/core/end_loop_calculator.h"
#include "mediapipe/calculators/core/gate_calculator.h"
#include "mediapipe/calculators/core/previous_loopback_calculator.h"
#include "mediapipe/calculators/core/split_vector_calculator.h"

#include "mediapipe/calculators/image/image_properties_calculator.h"

#include "mediapipe/calculators/tensor/image_to_tensor_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator_cpu.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.h"
#include "mediapipe/calculators/tensor/tensors_to_floats_calculator.h"
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.h"

#include "mediapipe/calculators/tflite/tflite_model_calculator.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator.h"
#include "mediapipe/calculators/tflite/tflite_custom_op_resolver_calculator.h"

#include "mediapipe/calculators/util/local_file_contents_calculator.h"
#include "mediapipe/calculators/util/association_norm_rect_calculator.h"
#include "mediapipe/calculators/util/collection_has_min_size_calculator.h"
#include "mediapipe/calculators/util/counting_vector_size_calculator.h"
#include "mediapipe/calculators/util/detection_projection_calculator.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.h"
#include "mediapipe/calculators/util/detections_to_rects_calculator.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.h"
#include "mediapipe/calculators/util/landmark_projection_calculator.h"
#include "mediapipe/calculators/util/thresholding_calculator.h"
#include "mediapipe/calculators/util/landmarks_to_detection_calculator.h"
#include "mediapipe/calculators/util/to_image_calculator.h"
#include "mediapipe/calculators/util/landmarks_refinement_calculator.h"

#include "mediapipe/framework/stream_handler/in_order_output_stream_handler.h"
#include "mediapipe/framework/stream_handler/default_input_stream_handler.h"
#include "mediapipe/framework/stream_handler/immediate_input_stream_handler.h"
#include "mediapipe/framework/tool/switch_container.h"

//#include "mediapipe/modules/face_geometry/env_generator_calculator.h"
//#include "mediapipe/modules/face_geometry/geometry_pipeline_calculator_with_pose_output.h"
#include "mediapipe/calculators/util/multi_face_landmarks_smoothing_calculator.h"

namespace face_mesh_graph {
  void staticLinkAllCalculators() {
    typeid(::mediapipe::FlowLimiterCalculator);
    typeid(::mediapipe::ConstantSidePacketCalculator);
    typeid(::mediapipe::TfLiteCustomOpResolverCalculator);
    typeid(::mediapipe::BeginLoopDetectionCalculator);
    typeid(::mediapipe::BeginLoopNormalizedRectCalculator);
    typeid(::mediapipe::ClipNormalizedRectVectorSizeCalculator);
    typeid(::mediapipe::ClipDetectionVectorSizeCalculator);
    typeid(::mediapipe::EndLoopNormalizedRectCalculator);
    typeid(::mediapipe::EndLoopNormalizedLandmarkListVectorCalculator);
    typeid(::mediapipe::GateCalculator);
    typeid(::mediapipe::api2::PreviousLoopbackCalculator);
    typeid(::mediapipe::SplitTensorVectorCalculator);

    typeid(::mediapipe::api2::ImagePropertiesCalculator);

    typeid(::mediapipe::api2::ImageToTensorCalculator);
    typeid(::mediapipe::api2::InferenceCalculator);
    typeid(::mediapipe::api2::InferenceCalculatorCpu);
    typeid(::mediapipe::api2::InferenceCalculatorCpuImpl);
    typeid(::mediapipe::api2::TensorsToDetectionsCalculator);
    typeid(::mediapipe::api2::TensorsToFloatsCalculator);
    typeid(::mediapipe::api2::TensorsToLandmarksCalculator);


    typeid(::mediapipe::TfLiteModelCalculator);
    typeid(::mediapipe::SsdAnchorsCalculator);

    typeid(::mediapipe::LocalFileContentsCalculator);
    typeid(::mediapipe::AssociationNormRectCalculator);
    typeid(::mediapipe::NormalizedRectVectorHasMinSizeCalculator);
    typeid(::mediapipe::CountingNormalizedLandmarkListVectorSizeCalculator);
    typeid(::mediapipe::DetectionProjectionCalculator);
    typeid(::mediapipe::NonMaxSuppressionCalculator);
    typeid(::mediapipe::DetectionsToRectsCalculator);
    typeid(::mediapipe::RectTransformationCalculator);
    typeid(::mediapipe::LandmarkProjectionCalculator);
    typeid(::mediapipe::ThresholdingCalculator);
    typeid(::mediapipe::LandmarksToDetectionCalculator);
    typeid(::mediapipe::ToImageCalculator);
    typeid(::mediapipe::api2::LandmarksRefinementCalculator);

    typeid(::mediapipe::InOrderOutputStreamHandler);
    typeid(::mediapipe::DefaultInputStreamHandler);
    typeid(::mediapipe::ImmediateInputStreamHandler);
    typeid(::mediapipe::tool::SwitchContainer);
    
  //  typeid(::mediapipe::FaceGeometryEnvGeneratorCalculator);
  //  typeid(::mediapipe::FaceGeometryPipelineCalculatorWithPoseOutput);
    typeid(::mediapipe::MultiFaceLandmarksSmoothingCalculator);
  }

std::string graphConfig = R"pb(
# MediaPipe graph that performs face mesh with TensorFlow Lite on CPU.

# Input image. (ImageFrame)
input_stream: "input_video"

# Camera calibration Matrix.
# input_stream: "camera_matrix"

# Real-time frame per second.
input_stream: "fps"

# Max number of faces to detect/process. (int)
input_side_packet: "num_faces"

# Whether to run face mesh model with attention on lips and eyes. (bool)
input_side_packet: "with_attention"

# Path to face detection model. (string)
input_side_packet: "face_detection_model_path"

# Path to face landmark model. (string)
input_side_packet: "face_landmark_model_path"

# Collection of detected/processed faces, each represented as a list of
# landmarks. (std::vector<NormalizedLandmarkList>)
output_stream: "filtered_multi_face_landmarks"

# Detected faces count. (int)
output_stream: "face_count"

# Regions of interest calculated based on landmarks.
# (std::vector<NormalizedRect>)
output_stream: "face_rects_from_landmarks"

# A vector of face pose data.
# std::vector<Eigen::Matrix4f>
#output_stream: "multi_face_poses"

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

# Determines if an input vector of NormalizedRect has a size greater than or
# equal to the provided num_faces.
node {
  calculator: "NormalizedRectVectorHasMinSizeCalculator"
  input_stream: "ITERABLE:prev_face_rects_from_landmarks"
  input_side_packet: "num_faces"
  output_stream: "prev_has_enough_faces"
}

# Drops the incoming image if FaceLandmarkCpu was able to identify face presence
# in the previous image. Otherwise, passes the incoming image through to trigger
# a new round of face detection in FaceDetectionShortRangeCpu.
node {
  calculator: "GateCalculator"
  input_stream: "throttled_input_video"
  input_stream: "DISALLOW:prev_has_enough_faces"
  output_stream: "gated_image"
  options: {
    [mediapipe.GateCalculatorOptions.ext] {
      empty_packets_as_allow: true
    }
  }
}


# ----- FaceDetectionShortRangeSideModelCpu Start ----- #


# Converts the input CPU image (ImageFrame) to the multi-backend image type
# (Image).
node: {
  calculator: "ToImageCalculator"
  input_stream: "IMAGE_CPU:gated_image"
  output_stream: "IMAGE:face_detection_short_range_side_model_cpu__multi_backend_image"
}

# Transforms the input image into a 128x128 tensor while keeping the aspect
# ratio (what is expected by the corresponding face detection model), resulting
# in potential letterboxing in the transformed image.
node: {
  calculator: "ImageToTensorCalculator"
  input_stream: "IMAGE:face_detection_short_range_side_model_cpu__multi_backend_image"
  output_stream: "TENSORS:face_detection_short_range_side_model_cpu__input_tensors"
  output_stream: "MATRIX:face_detection_short_range_side_model_cpu__transform_matrix"
  options: {
    [mediapipe.ImageToTensorCalculatorOptions.ext] {
      output_tensor_width: 128
      output_tensor_height: 128
      keep_aspect_ratio: true
      output_tensor_float_range {
        min: -1.0
        max: 1.0
      }
      border_mode: BORDER_ZERO
    }
  }
}

# Runs a TensorFlow Lite model on CPU that takes an image tensor and outputs a
# vector of tensors representing, for instance, detection boxes/keypoints and
# scores.
node {
  calculator: "InferenceCalculator"
  input_stream: "TENSORS:face_detection_short_range_side_model_cpu__input_tensors"
  output_stream: "TENSORS:face_detection_short_range_side_model_cpu__detection_tensors"
  input_side_packet: "MODEL:face_detection_model"
  options {
    [mediapipe.InferenceCalculatorOptions.ext] {
      delegate { tflite {} }
    }
  }
}


# ----- FaceDetectionShortRangeCommon Start ----- #


# Performs tensor post processing to generate face detections.

# Generates a single side packet containing a vector of SSD anchors based on
# the specification in the options.
node {
  calculator: "SsdAnchorsCalculator"
  output_side_packet: "face_detection_short_range_common__anchors"
  options: {
    [mediapipe.SsdAnchorsCalculatorOptions.ext] {
      num_layers: 4
      min_scale: 0.1484375
      max_scale: 0.75
      input_size_height: 128
      input_size_width: 128
      anchor_offset_x: 0.5
      anchor_offset_y: 0.5
      strides: 8
      strides: 16
      strides: 16
      strides: 16
      aspect_ratios: 1.0
      fixed_anchor_size: true
    }
  }
}

# Decodes the detection tensors generated by the TensorFlow Lite model, based on
# the SSD anchors and the specification in the options, into a vector of
# detections. Each detection describes a detected object.
node {
  calculator: "TensorsToDetectionsCalculator"
  input_stream: "TENSORS:face_detection_short_range_side_model_cpu__detection_tensors"
  input_side_packet: "ANCHORS:face_detection_short_range_common__anchors"
  output_stream: "DETECTIONS:face_detection_short_range_common__unfiltered_detections"
  options: {
    [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
      num_classes: 1
      num_boxes: 896
      num_coords: 16
      box_coord_offset: 0
      keypoint_coord_offset: 4
      num_keypoints: 6
      num_values_per_keypoint: 2
      sigmoid_score: true
      score_clipping_thresh: 100.0
      reverse_output_order: true
      x_scale: 128.0
      y_scale: 128.0
      h_scale: 128.0
      w_scale: 128.0
      min_score_thresh: 0.5
    }
  }
}

# Performs non-max suppression to remove excessive detections.
node {
  calculator: "NonMaxSuppressionCalculator"
  input_stream: "face_detection_short_range_common__unfiltered_detections"
  output_stream: "face_detection_short_range_common__filtered_detections"
  options: {
    [mediapipe.NonMaxSuppressionCalculatorOptions.ext] {
      min_suppression_threshold: 0.3
      overlap_type: INTERSECTION_OVER_UNION
      algorithm: WEIGHTED
    }
  }
}

# Projects the detections from input tensor to the corresponding locations on
# the original image (input to the graph).
node {
  calculator: "DetectionProjectionCalculator"
  input_stream: "DETECTIONS:face_detection_short_range_common__filtered_detections"
  input_stream: "PROJECTION_MATRIX:face_detection_short_range_side_model_cpu__transform_matrix"
  output_stream: "DETECTIONS:all_face_detections"
}



# ----- FaceDetectionShortRangeCommon End ----- #



# ----- FaceDetectionShortRangeSideModelCpu End ----- #


# Makes sure there are no more detections than the provided num_faces.
node {
  calculator: "ClipDetectionVectorSizeCalculator"
  input_stream: "all_face_detections"
  output_stream: "face_detections"
  input_side_packet: "num_faces"
}

# Calculate size of the image.
node {
  calculator: "ImagePropertiesCalculator"
  input_stream: "IMAGE:gated_image"
  output_stream: "SIZE:gated_image_size"
}

# Outputs each element of face_detections at a fake timestamp for the rest of
# the graph to process. Clones the image size packet for each face_detection at
# the fake timestamp. At the end of the loop, outputs the BATCH_END timestamp
# for downstream calculators to inform them that all elements in the vector have
# been processed.
node {
  calculator: "BeginLoopDetectionCalculator"
  input_stream: "ITERABLE:face_detections"
  input_stream: "CLONE:gated_image_size"
  output_stream: "ITEM:face_detection"
  output_stream: "CLONE:detections_loop_image_size"
  output_stream: "BATCH_END:detections_loop_end_timestamp"
}



# ----- FaceDetectionFrontDetectionToRoi Start ----- #


# Converts results of face detection into a rectangle (normalized by image size)
# that encloses the face and is rotated such that the line connecting left eye
# and right eye is aligned with the X-axis of the rectangle.
node {
  calculator: "DetectionsToRectsCalculator"
  input_stream: "DETECTION:face_detection"
  input_stream: "IMAGE_SIZE:detections_loop_image_size"
  output_stream: "NORM_RECT:face_detection_front_detection_to_roi__initial_roi"
  options: {
    [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
      rotation_vector_start_keypoint_index: 0  # Left eye.
      rotation_vector_end_keypoint_index: 1  # Right eye.
      rotation_vector_target_angle_degrees: 0
    }
  }
}


# Expands and shifts the rectangle that contains the face so that it's likely
# to cover the entire face.
node {
  calculator: "RectTransformationCalculator"
  input_stream: "NORM_RECT:face_detection_front_detection_to_roi__initial_roi"
  input_stream: "IMAGE_SIZE:detections_loop_image_size"
  output_stream: "face_rect_from_detection"
  options: {
    [mediapipe.RectTransformationCalculatorOptions.ext] {
      scale_x: 1.5
      scale_y: 1.5
      square_long: true
    }
  }
}


# ----- FaceDetectionFrontDetectionToRoi End ----- #


# Counting a multi_faceLandmarks vector size. The image stream is only used to
# make the calculator work even when there is no input vector.
node {
  calculator: "CountingNormalizedLandmarkListVectorSizeCalculator"
  input_stream: "CLOCK:input_video"
  input_stream: "VECTOR:multi_face_landmarks"
  output_stream: "COUNT:face_count"
}

# Collects a NormalizedRect for each face into a vector. Upon receiving the
# BATCH_END timestamp, outputs the vector of NormalizedRect at the BATCH_END
# timestamp.
node {
  calculator: "EndLoopNormalizedRectCalculator"
  input_stream: "ITEM:face_rect_from_detection"
  input_stream: "BATCH_END:detections_loop_end_timestamp"
  output_stream: "ITERABLE:face_rects_from_detections"
}

# Performs association between NormalizedRect vector elements from previous
# image and rects based on face detections from the current image. This
# calculator ensures that the output face_rects vector doesn't contain
# overlapping regions based on the specified min_similarity_threshold.
node {
  calculator: "AssociationNormRectCalculator"
  input_stream: "face_rects_from_detections"
  input_stream: "prev_face_rects_from_landmarks"
  output_stream: "face_rects"
  options: {
    [mediapipe.AssociationCalculatorOptions.ext] {
      min_similarity_threshold: 0.5
    }
  }
}

# Calculate size of the image.
node {
  calculator: "ImagePropertiesCalculator"
  input_stream: "IMAGE:input_video"
  output_stream: "SIZE:image_size"
}

# Outputs each element of face_rects at a fake timestamp for the rest of the
# graph to process. Clones image and image size packets for each
# single_face_rect at the fake timestamp. At the end of the loop, outputs the
# BATCH_END timestamp for downstream calculators to inform them that all
# elements in the vector have been processed.
node {
  calculator: "BeginLoopNormalizedRectCalculator"
  input_stream: "ITERABLE:face_rects"
  input_stream: "CLONE:0:input_video"
  input_stream: "CLONE:1:image_size"
  output_stream: "ITEM:face_rect"
  output_stream: "CLONE:0:landmarks_loop_image"
  output_stream: "CLONE:1:landmarks_loop_image_size"
  output_stream: "BATCH_END:landmarks_loop_end_timestamp"
}



# ------ FaceLandmarkSideModelCpu Start ------ #


# Transforms the input image into a 192x192 tensor.
node: {
  calculator: "ImageToTensorCalculator"
  input_stream: "IMAGE:landmarks_loop_image"
  input_stream: "NORM_RECT:face_rect"
  output_stream: "TENSORS:face_landmark_side_model_cpu__input_tensors"
  options: {
    [mediapipe.ImageToTensorCalculatorOptions.ext] {
      output_tensor_width: 192
      output_tensor_height: 192
      output_tensor_float_range {
        min: 0.0
        max: 1.0
      }
    }
  }
}

# Generates a single side packet containing a TensorFlow Lite op resolver that
# supports custom ops needed by the model used in this graph.
node {
  calculator: "TfLiteCustomOpResolverCalculator"
  output_side_packet: "op_resolver"
}

# Runs a TensorFlow Lite model on CPU that takes an image tensor and outputs a
# vector of tensors representing, for instance, detection boxes/keypoints and
# scores.
node {
  calculator: "InferenceCalculator"
  input_stream: "TENSORS:face_landmark_side_model_cpu__input_tensors"
  input_side_packet: "MODEL:face_landmark_model"
  input_side_packet: "CUSTOM_OP_RESOLVER:op_resolver"
  output_stream: "TENSORS:face_landmark_side_model_cpu__output_tensors"
  options {
    [mediapipe.InferenceCalculatorOptions.ext] {
      delegate { tflite {} }
    }
  }
}

$SplitTensorVectorCalculator

# Converts the face-flag tensor into a float that represents the confidence
# score of face presence.
node {
  calculator: "TensorsToFloatsCalculator"
  input_stream: "TENSORS:face_landmark_side_model_cpu__face_flag_tensor"
  output_stream: "FLOAT:face_landmark_side_model_cpu__face_presence_score"
  options {
    [mediapipe.TensorsToFloatsCalculatorOptions.ext] {
      activation: SIGMOID
    }
  }
}

# Applies a threshold to the confidence score to determine whether a face is
# present.
node {
  calculator: "ThresholdingCalculator"
  input_stream: "FLOAT:face_landmark_side_model_cpu__face_presence_score"
  output_stream: "FLAG:face_landmark_side_model_cpu__face_presence"
  options: {
    [mediapipe.ThresholdingCalculatorOptions.ext] {
      threshold: 0.5
    }
  }
}

# Drop landmarks tensors if face is not present.
node {
  calculator: "GateCalculator"
  input_stream: "face_landmark_side_model_cpu__landmark_tensors"
  input_stream: "ALLOW:face_landmark_side_model_cpu__face_presence"
  output_stream: "face_landmark_side_model_cpu__ensured_landmark_tensors"
}

# Decodes the landmark tensors into a vector of landmarks, where the landmark
# coordinates are normalized by the size of the input image to the model.
$TensorsToLandmarksSubgraph

# Projects the landmarks from the cropped face image to the corresponding
# locations on the full image before cropping (input to the graph).
node {
  calculator: "LandmarkProjectionCalculator"
  input_stream: "NORM_LANDMARKS:face_landmark_side_model_cpu__landmarks"
  input_stream: "NORM_RECT:face_rect"
  output_stream: "NORM_LANDMARKS:face_landmarks"
}

# ------ FaceLandmarkSideModelCpu End ------ #

# ------ FaceLandmarksToRoi Start ------ #

node {
  calculator: "LandmarksToDetectionCalculator"
  input_stream: "NORM_LANDMARKS:face_landmarks"
  output_stream: "DETECTION:face_landmarks_to_roi__face_detection"
}

# the rectangle.
node {
  calculator: "DetectionsToRectsCalculator"
  input_stream: "DETECTION:face_landmarks_to_roi__face_detection"
  input_stream: "IMAGE_SIZE:landmarks_loop_image_size"
  output_stream: "NORM_RECT:face_landmarks_to_roi__face_rect_from_landmarks"
  options: {
    [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
      rotation_vector_start_keypoint_index: 33  # Left side of left eye.
      rotation_vector_end_keypoint_index: 263  # Right side of right eye.
      rotation_vector_target_angle_degrees: 0
    }
  }
}

node {
  calculator: "RectTransformationCalculator"
  input_stream: "NORM_RECT:face_landmarks_to_roi__face_rect_from_landmarks"
  input_stream: "IMAGE_SIZE:landmarks_loop_image_size"
  output_stream: "face_rect_from_landmarks"
  options: {
    [mediapipe.RectTransformationCalculatorOptions.ext] {
      scale_x: 1.5
      scale_y: 1.5
      square_long: true
    }
  }
}


# ------ FaceLandmarksToRoi End ------ #

node {
  calculator: "EndLoopNormalizedLandmarkListVectorCalculator"
  input_stream: "ITEM:face_landmarks"
  input_stream: "BATCH_END:landmarks_loop_end_timestamp"
  output_stream: "ITERABLE:multi_face_landmarks"
}

node {
  calculator: "EndLoopNormalizedRectCalculator"
  input_stream: "ITEM:face_rect_from_landmarks"
  input_stream: "BATCH_END:landmarks_loop_end_timestamp"
  output_stream: "ITERABLE:face_rects_from_landmarks"
}

node {
  calculator: "PreviousLoopbackCalculator"
  input_stream: "MAIN:input_video"
  input_stream: "LOOP:face_rects_from_landmarks"
  input_stream_info: {
    tag_index: "LOOP"
    back_edge: true
  }
  output_stream: "PREV_LOOP:prev_face_rects_from_landmarks"
}

# Applies smoothing to a face landmark list. The filter options were handpicked
# to achieve better visual results.
node {
  calculator: "MultiFaceLandmarksSmoothingCalculator"
  input_stream: "NORM_MULTI_FACE_LANDMARKS:multi_face_landmarks"
  input_stream: "IMAGE_SIZE:image_size"
  input_stream: "FPS:fps"
  output_stream: "NORM_FILTERED_MULTI_FACE_LANDMARKS:filtered_multi_face_landmarks"
  node_options: {
    [type.googleapis.com/mediapipe.LandmarksSmoothingCalculatorOptions] {
      velocity_filter: {
        window_size: $window_size_param
        velocity_scale: $velocity_scale_param
      }
    }
  }
}
)pb";
}

#endif // FACE_MESH_GRAPH_H
