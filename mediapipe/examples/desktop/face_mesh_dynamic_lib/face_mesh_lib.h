#ifndef FACE_MESH_DYNAMIC_LIBRARY_H
#define FACE_MESH_DYNAMIC_LIBRARY_H

#if defined(_WIN32) || defined(WIN32)

  #include <windows.h>

  #ifdef COMPILING_DLL
    #define DLLEXPORT __declspec(dllexport)
  #else
    #define DLLEXPORT __declspec(dllimport)
  #endif
#else

  #define DLLEXPORT

#endif

#include "MPFaceMeshDetector.h"

#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT MPFaceMeshDetector * MPFaceMeshDetectorConstruct(int numFaces,
                                                           const char *face_detection_model_path,
                                                           const char *face_landmark_model_path);

DLLEXPORT void MPFaceMeshDetectorDestruct(MPFaceMeshDetector *detector);

DLLEXPORT void MPFaceMeshDetectorDetectFaces(MPFaceMeshDetector *detector,
                                             const cv::Mat &camera_frame,
                                             cv::Rect *multi_face_bounding_boxes,
                                             int *numFaces);

DLLEXPORT void MPFaceMeshDetectorDetect2DLandmarks(MPFaceMeshDetector *detector,
                                                   cv::Point2f **multi_face_landmarks,
                                                   int *numFaces);

DLLEXPORT void MPFaceMeshDetectorDetect3DLandmarks(MPFaceMeshDetector *detector,
                                                   cv::Point3f **multi_face_landmarks,
                                                   int *numFaces);

DLLEXPORT extern const int MPFaceMeshDetectorLandmarksNum;

#ifdef __cplusplus
};
#endif

#endif //FACE_MESH_DYNAMIC_LIBRARY_H