#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/modules/face_geometry/geometry_pipeline_calculator.pb.h"
#include "mediapipe/modules/face_geometry/libs/geometry_pipeline.h"
#include "mediapipe/modules/face_geometry/libs/validation_utils.h"
#include "mediapipe/modules/face_geometry/protos/environment.pb.h"
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"
#include "mediapipe/modules/face_geometry/protos/geometry_pipeline_metadata.pb.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {
namespace {

static constexpr char kWithAttentionTag[] = "WITH_ATTENTION";
static constexpr char kImageSizeTag[] = "IMAGE_SIZE";
static constexpr char kMultiFacePosesTag[] = "MULTI_FACE_POSES";
static constexpr char kMultiFaceLandmarksTag[] = "MULTI_FACE_LANDMARKS";

// A calculator that renders a visual effect for multiple faces.
//
// Inputs:
//   IMAGE_SIZE (`std::pair<int, int>`, required):
//     The size of the current frame. The first element of the pair is the frame
//     width; the other one is the frame height.
//
//     The face landmarks should have been detected on a frame with the same
//     ratio. If used as-is, the resulting face geometry visualization should be
//     happening on a frame with the same ratio as well.
//
//   MULTI_FACE_LANDMARKS (`std::vector<NormalizedLandmarkList>`, required):
//     A vector of face landmark lists.
//
// Input side packets:
//   ENVIRONMENT (`face_geometry::Environment`, required)
//     Describes an environment; includes the camera frame origin point location
//     as well as virtual camera parameters.
//
// Output:
//   MULTI_FACE_POSES (`std::vector<Eigen::Matrix4f>`, required):
//     A vector of face pose data.
//
// Options:
//   metadata_path (`string`, optional):
//     Defines a path for the geometry pipeline metadata file.
//
//     The geometry pipeline metadata file format must be the binary
//     `face_geometry.GeometryPipelineMetadata` proto.
//
class GeometryPipelineCalculator : public CalculatorBase {
private:
    static constexpr char kEnvironmentTag[] = "ENVIRONMENT";
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets()
        .Tag(kEnvironmentTag)
        .Set<face_geometry::Environment>();
    cc->InputSidePackets().Tag(kWithAttentionTag).Set<bool>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    cc->Inputs()
        .Tag(kMultiFaceLandmarksTag)
        .Set<std::vector<NormalizedLandmarkList>>();
    cc->Outputs()
        .Tag(kMultiFacePosesTag)
        .Set<std::vector<Eigen::Matrix4f>>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(mediapipe::TimestampDiff(0));

    const auto& options = cc->Options<FaceGeometryPipelineCalculatorOptions>();

    ASSIGN_OR_RETURN(
        face_geometry::GeometryPipelineMetadata metadata,
        ReadMetadataFromFile(options.metadata_path()),
        _ << "Failed to read the geometry pipeline metadata from file!");

    MP_RETURN_IF_ERROR(
        face_geometry::ValidateGeometryPipelineMetadata(metadata))
        << "Invalid geometry pipeline metadata!";

    const face_geometry::Environment& environment =
        cc->InputSidePackets()
            .Tag(kEnvironmentTag)
            .Get<face_geometry::Environment>();
    MP_RETURN_IF_ERROR(face_geometry::ValidateEnvironment(environment))
        << "Invalid environment!";

    ASSIGN_OR_RETURN(
        geometry_pipeline_,
        face_geometry::CreateGeometryPipeline(environment, metadata),
        _ << "Failed to create a geometry pipeline!");
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Both the `IMAGE_SIZE` and the `MULTI_FACE_LANDMARKS` streams are required
    // to have a non-empty packet. In case this requirement is not met, there's
    // nothing to be processed at the current timestamp.
    if (cc->Inputs().Tag(kImageSizeTag).IsEmpty() ||
        cc->Inputs().Tag(kMultiFaceLandmarksTag).IsEmpty()) {
      return absl::OkStatus();
    }

    const bool& with_attention =
        cc->InputSidePackets().Tag(kWithAttentionTag).Get<bool>();
    const auto& image_size =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    const auto& multi_face_landmarks =
        cc->Inputs()
            .Tag(kMultiFaceLandmarksTag)
            .Get<std::vector<NormalizedLandmarkList>>();

    auto multi_face_poses =
        absl::make_unique<std::vector<Eigen::Matrix4f>>();
    ASSIGN_OR_RETURN(
        *multi_face_poses,
        geometry_pipeline_->EstimateFacePoses(
            multi_face_landmarks,  //
            with_attention,
            /*frame_width*/ image_size.first,
            /*frame_height*/ image_size.second),
        _ << "Failed to estimate face pose for multiple faces!");
    cc->Outputs()
        .Tag(kMultiFacePosesTag)
        .AddPacket(mediapipe::MakePacket<std::vector<Eigen::Matrix4f>>(*multi_face_poses)
                       .At(cc->InputTimestamp()));
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  static absl::StatusOr<face_geometry::GeometryPipelineMetadata>
  ReadMetadataFromFile(const std::string& metadata_path) {
    ASSIGN_OR_RETURN(std::string metadata_blob,
                     ReadContentBlobFromFile(metadata_path),
                     _ << "Failed to read a metadata blob from file!");

    face_geometry::GeometryPipelineMetadata metadata;
    RET_CHECK(metadata.ParseFromString(metadata_blob))
        << "Failed to parse a metadata proto from a binary blob!";

    return metadata;
  }

  static absl::StatusOr<std::string> ReadContentBlobFromFile(
      const std::string& unresolved_path) {
    ASSIGN_OR_RETURN(std::string resolved_path,
                     mediapipe::PathToResourceAsFile(unresolved_path),
                     _ << "Failed to resolve path! Path = " << unresolved_path);

    std::string content_blob;
    MP_RETURN_IF_ERROR(
        mediapipe::GetResourceContents(resolved_path, &content_blob))
        << "Failed to read content blob! Resolved path = " << resolved_path;

    return content_blob;
  }

  std::unique_ptr<face_geometry::GeometryPipeline> geometry_pipeline_;
};

}  // namespace

using FaceGeometryPipelineCalculatorWithPoseOutput = GeometryPipelineCalculator;

REGISTER_CALCULATOR(FaceGeometryPipelineCalculatorWithPoseOutput);

}  // namespace mediapipe
