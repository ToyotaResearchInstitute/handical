class gtsam::Pose3;
class gtsam::Cal3_S2;
class gtsam::Cal3DS2;
virtual class gtsam::NonlinearFactor;
virtual class gtsam::NoiseModelFactor : gtsam::NonlinearFactor;
class gtsam::noiseModel::Base;

namespace handical {

#include <handical_factor.h>
template <T = {gtsam::Cal3_S2, gtsam::Cal3DS2}>
virtual class HandiCalFactor : gtsam::NoiseModelFactor {
  HandiCalFactor(size_t key_BC, size_t key_EM,
                 const T* K,
                 const gtsam::Pose3 &X_BE, const gtsam::Point3 &p_MP,
                 const gtsam::Point2 &measured, const gtsam::noiseModel::Base* model);
  Vector evaluateError(const gtsam::Pose3 &X_BC, const gtsam::Pose3 &X_EM) const;
};

}
