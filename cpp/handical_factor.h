#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace handical {

/// A binary factor for Hand-Eye Calibration between two unknowns:
///    - X_BC: The camera pose C in the robot's base frame B
///    - X_EM: The marker's pose M in the end-effector frame E
template <class CALIBRATION>
class HandiCalFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;

public:
  /// Default constructor.
  HandiCalFactor() {}
  /// Constructor.
  /// @param key_BC key of the X_BC variable node in the graph
  /// @param key_EM key of the X_EM variable node in the graph
  /// @param K calibration matrix
  /// @param X_BE Pose of the end-effector in the robot's base frame,
  ///             assumed to be known
  /// @param P_MP 3D coordinate of the measurement in the marker frame
  /// @param measured 2D image coordinate of the measurement
  /// @param model Noise model of the measurement
  HandiCalFactor(gtsam::Key key_BC, gtsam::Key key_EM,
                 const boost::shared_ptr<CALIBRATION> &K,
                 const gtsam::Pose3 &X_BE, const gtsam::Point3 &P_MP,
                 const gtsam::Point2 &measured,
                 const gtsam::SharedNoiseModel &model)
      : Base(model, key_BC, key_EM), K_(K), X_BE_(X_BE), P_MP_(P_MP),
        measured_(measured) {}

  virtual ~HandiCalFactor() {}

  virtual gtsam::NonlinearFactor::shared_ptr clone() const {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new HandiCalFactor(*this)));
  }

  /// Compute the error of this factor.
  /// The measurement model is:
  ///    z =  camera_in_marker_frame.project(P_M)
  ///    where the camera pose in the marker frame X_MC is a function of
  /// the two unknowns, X_EM and X_BC, as follows
  ///      X_MC = (X_BE*X_EM).inverse()*X_BC
  ///
  /// TODO(duy): Treat X_BE as an unknown with a strong prior!!!
  gtsam::Vector evaluateError(
      const gtsam::Pose3 &X_BC, const gtsam::Pose3 &X_EM,
      boost::optional<gtsam::Matrix &> J1 = boost::none,
      boost::optional<gtsam::Matrix &> J2 = boost::none) const override {
    gtsam::Point2 predict;
    gtsam::Pose3 X_MC = (X_BE_.compose(X_EM)).inverse().compose(X_BC);
    gtsam::PinholeCamera<CALIBRATION> camera(X_MC, *K_);
    if (J1 || J2) {
      gtsam::Matrix6 D_MC_BC = gtsam::Matrix6::Identity();
      gtsam::Matrix6 D_MC_EM = -X_MC.inverse().AdjointMap();
      Eigen::Matrix<double, 2, 6> D_pr_MC;
      predict = camera.project(P_MP_, D_pr_MC, boost::none, boost::none);
      if (J1)
        *J1 = D_pr_MC * D_MC_BC;
      if (J2)
        *J2 = D_pr_MC * D_MC_EM;
    } else {
      predict = camera.project(P_MP_, boost::none, boost::none, boost::none);
    }
    return predict - measured_;
  }

private:
  boost::shared_ptr<CALIBRATION> K_; /// Intrinsic parameters of the camera
  gtsam::Pose3 X_BE_;      /// The pose of the End-effector in the robot's Base
                           /// frame, assumed to be known (fixed)
  gtsam::Point3 P_MP_;     /// The 3D point position in the marker's frame M
  gtsam::Point2 measured_; /// The point measurement in the camera image
};

} // namespace handical
