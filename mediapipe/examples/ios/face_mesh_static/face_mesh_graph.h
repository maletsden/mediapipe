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

  }

  const std::string graphConfig = R"pb(
# MediaPipe graph that performs face mesh with TensorFlow Lite on CPU.

# Input image. (ImageFrame)
input_stream: "input_video"

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

# Defines whether landmarks on the previous image should be used to help
# localize landmarks on the current image.
node {
  name: "ConstantSidePacketCalculator"
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:use_prev_landmarks"
  options: {
    [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
      packet { bool_value: true }
    }
  }
}
node {
  calculator: "GateCalculator"
  input_side_packet: "ALLOW:use_prev_landmarks"
  input_stream: "prev_face_rects_from_landmarks"
  output_stream: "gated_prev_face_rects_from_landmarks"
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

# Splits a vector of tensors into landmark tensors and face flag tensor.
node {
  calculator: "SwitchContainer"
  input_side_packet: "ENABLE:with_attention"
  input_stream: "face_landmark_side_model_cpu__output_tensors"
  output_stream: "face_landmark_side_model_cpu__landmark_tensors"
  output_stream: "face_landmark_side_model_cpu__face_flag_tensor"
  options: {
    [mediapipe.SwitchContainerOptions.ext] {
      contained_node: {
        calculator: "SplitTensorVectorCalculator"
        options: {
          [mediapipe.SplitVectorCalculatorOptions.ext] {
            ranges: { begin: 0 end: 1 }
            ranges: { begin: 1 end: 2 }
          }
        }
      }
      contained_node: {
        calculator: "SplitTensorVectorCalculator"
        options: {
          [mediapipe.SplitVectorCalculatorOptions.ext] {
            ranges: { begin: 0 end: 6 }
            ranges: { begin: 6 end: 7 }
          }
        }
      }
    }
  }
}

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
node {
  calculator: "SwitchContainer"
  input_side_packet: "ENABLE:with_attention"
  options: {
    [mediapipe.SwitchContainerOptions.ext] {
      contained_node: {
                # Decodes the landmark tensors into a vector of lanmarks, where the landmark
                # coordinates are normalized by the size of the input image to the model.
                node {
                    calculator: "TensorsToLandmarksCalculator"
                    input_stream: "TENSORS:face_landmark_side_model_cpu__ensured_landmark_tensors"
                    output_stream: "LANDMARKS:face_landmark_side_model_cpu__landmarks"
                    options: {
                        [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
                            num_landmarks: 468
                            input_image_width: 192
                            input_image_height: 192
                        }
                    }
                }
      }
      contained_node: {
        # Splits a vector of tensors into multiple vectors.
        node {
              calculator: "SplitTensorVectorCalculator"
              input_stream: "TENSORS:face_landmark_side_model_cpu__ensured_landmark_tensors"
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
              output_stream: "LANDMARKS:face_landmark_side_model_cpu__landmarks"
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
      }
    }
  }
}

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


# Converts face landmarks to a detection that tightly encloses all landmarks.
node {
  calculator: "LandmarksToDetectionCalculator"
  input_stream: "NORM_LANDMARKS:face_landmarks"
  output_stream: "DETECTION:face_landmarks_to_roi__face_detection"
}

# Converts the face detection into a rectangle (normalized by image size)
# that encloses the face and is rotated such that the line connecting left side
# of the left eye and right side of the right eye is aligned with the X-axis of
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

# Expands the face rectangle so that in the next video image it's likely to
# still contain the face even with some motion.
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


# Collects a set of landmarks for each face into a vector. Upon receiving the
# BATCH_END timestamp, outputs the vector of landmarks at the BATCH_END
# timestamp.
node {
  calculator: "EndLoopNormalizedLandmarkListVectorCalculator"
  input_stream: "ITEM:face_landmarks"
  input_stream: "BATCH_END:landmarks_loop_end_timestamp"
  output_stream: "ITERABLE:multi_face_landmarks"
}

# Collects a NormalizedRect for each face into a vector. Upon receiving the
# BATCH_END timestamp, outputs the vector of NormalizedRect at the BATCH_END
# timestamp.
node {
  calculator: "EndLoopNormalizedRectCalculator"
  input_stream: "ITEM:face_rect_from_landmarks"
  input_stream: "BATCH_END:landmarks_loop_end_timestamp"
  output_stream: "ITERABLE:face_rects_from_landmarks"
}

# Caches face rects calculated from landmarks, and upon the arrival of the next
# input image, sends out the cached rects with timestamps replaced by that of
# the input image, essentially generating a packet that carries the previous
# face rects. Note that upon the arrival of the very first input image, a
# timestamp bound update occurs to jump start the feedback loop.
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


)pb";
}

#endif // FACE_MESH_GRAPH_H