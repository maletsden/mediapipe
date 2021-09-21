# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_static_framework",
)

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

objc_library(
    name = "face_mesh_lib",
    hdrs = [
        "face_mesh_graph.h",
        "face_mesh_lib.h",
    ],
    srcs = [
       "face_mesh_lib.cpp",
    ],
    copts = [
        "-Wno-shorten-64-to-32",
        "-std=c++17",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",

        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/tflite:tflite_model_calculator",
        "//mediapipe/calculators/util:local_file_contents_calculator",
        "//mediapipe/modules/face_landmark:face_landmark_front_side_model_cpu_with_face_counter",
    ],
)

ios_static_framework(
    name = "FaceMeshCpu",
    bundle_name = "FaceMeshCpu",
    families = [
        "iphone",
        "ipad",
    ],
    hdrs = [
        "face_mesh_lib.h",
    ],
    umbrella_header = "FaceMeshCpu.h",
    minimum_os_version = "11.0",
    deps = [":face_mesh_lib"]
)
