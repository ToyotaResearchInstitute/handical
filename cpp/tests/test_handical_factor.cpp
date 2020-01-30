#include "handical_factor.h"

#include <iostream>

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Cal3_S2.h>

using namespace gtsam;
using namespace handical;

static double fov = 60; // degrees
static size_t w = 640, h = 480;
static Cal3_S2::shared_ptr K(new Cal3_S2(fov, w, h));
static Pose3 X_BE;
static Point3 P(0., 0., 0.);
static SharedNoiseModel noise(noiseModel::Unit::Create(2));
static Point2 z;
static Key key_BC = 0, key_EM = 1;

TEST(HandiCalFactor, Error) {
  HandiCalFactor<Cal3_S2> factor(key_BC, key_EM, K, X_BE, P, z, noise);
  Pose3 X_BC(Rot3::RzRyRx(0., M_PI, 0.), Point3(0., 0., 1.0));
  Pose3 X_EM;
  Vector actualError = factor.evaluateError(X_BC, X_EM);
  Vector expectedError = Vector2(w / 2., h / 2.);
  CHECK(assert_equal(expectedError, actualError, 1e-9));
}

TEST(HandiCalFactor, Jacobians) {
  HandiCalFactor<Cal3_S2> factor(key_BC, key_EM, K, X_BE, P, z, noise);
  Pose3 X_BC(Rot3::RzRyRx(M_PI / 3.0, M_PI, 0.), Point3(0.3, 0., 1.0));
  Pose3 X_EM;
  Matrix J_BC, J_EM;
  Vector actualError = factor.evaluateError(X_BC, X_EM, J_BC, J_EM);
  Vector expectedError = Vector2(w / 2., h / 2.);

  boost::function<Vector(const Pose3 &, const Pose3 &)> func =
      boost::bind(&HandiCalFactor<Cal3_S2>::evaluateError, factor, _1, _2,
                  boost::none, boost::none);
  Matrix expected_J_BC = numericalDerivative21(func, X_BC, X_EM);
  Matrix expected_J_EM = numericalDerivative22(func, X_BC, X_EM);
  CHECK(assert_equal(J_BC, expected_J_BC, 1e-5));
  CHECK(assert_equal(J_EM, expected_J_EM, 1e-5));
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
