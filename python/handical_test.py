import unittest
from handicalpy import *
import argparse

DEGREE = math.pi / 180.


class DIM:
    X = 0
    Y = 1
    Z = 2
    ROLL = 3
    PITCH = 4
    YAW = 5


class HandicalTestData:
    """
    Generate ground truth and noisy data for testing hand-eye-calibration
    """
    def __init__(self,
                 chess_board,
                 linspaces,
                 noise=None,
                 gt_X_EM=gtsam.Pose3(),
                 gt_X_BE0=gtsam.Pose3()):
        # using the same intrisic parameters for all cameras
        self.camera_intrinsic = gtsam.Cal3_S2(4000., 4000., 0., 320., 240.)
        self.img_nrows = 480
        self.img_ncols = 640
        # ground truth camera poses in the robot's base frame
        self.gt_X_BCs = [None] * 3
        self.gt_X_BCs[0] = gtsam.Pose3(gtsam.Rot3.Ry(180. * DEGREE),
                                       gtsam.Point3(0., 0., 4.))
        self.gt_X_BCs[1] = gtsam.Pose3(gtsam.Rot3.Ry(-135. * DEGREE),
                                       gtsam.Point3(4., 0., 4.))
        self.gt_X_BCs[2] = gtsam.Pose3(gtsam.Rot3.Ry(135. * DEGREE),
                                       gtsam.Point3(-4., 0., 4.))
        self.chess_board = chess_board
        self.gt_X_EM = gt_X_EM
        self.gt_X_BE0 = gt_X_BE0
        self.gt_X_BEs = self.generate_gt_X_BEs(gt_X_BE0, linspaces)
        self.noise = noise
        self.sampler = None
        if noise:
            self.sampler = gtsam.Sampler(noise, 1403)
        self.all_measurements = self.generate_all_measurements()

    def init_X_EM(self):
        return self.gt_X_EM

    def nr_cameras(self):
        return len(self.gt_X_BCs)

    def camera_intrinsics(self):
        return [self.camera_intrinsic] * self.nr_cameras()

    def nr_end_effector_poses(self):
        return len(self.gt_X_BEs)

    def nr_points(self):
        return len(self.chess_board.get_3d_points())

    def X_BEs(self):
        return self.gt_X_BEs

    def X_BE(self, pose_id):
        return self.gt_X_BEs[pose_id]

    def X_BCs(self):
        return self.gt_X_BCs

    def X_CM(self, pose_id, cam_id):
        X_CM = self.gt_X_BCs[cam_id].inverse().compose(
            self.X_BE(pose_id)).compose(self.init_X_EM())
        return X_CM

    def measurements(self, pose_id, cam_id):
        return self.all_measurements[pose_id][cam_id]

    def measurement(self, pose_id, cam_id, point_id):
        return self.all_measurements[pose_id][cam_id][point_id]

    def generate_gt_X_BEs(self, gt_X_BE0, linspaces):
        gt_X_BEs = []
        for z in linspaces[DIM.Y]:
            for x in linspaces[DIM.X]:
                for y in linspaces[DIM.Y]:
                    for yaw in linspaces[DIM.YAW]:
                        for pitch in linspaces[DIM.PITCH]:
                            for roll in linspaces[DIM.ROLL]:
                                gt_X_BEs.append(
                                    gtsam.Pose3(
                                        gtsam.Rot3.Ypr(yaw, pitch, roll),
                                        gtsam.Point3(x, y, z)))
        for i in range(len(gt_X_BEs)):
            gt_X_BEs[i] = self.gt_X_BE0.compose(gt_X_BEs[i])
        return gt_X_BEs

    def inside_image(self, p, nrows, ncols):
        return p.x() >= 0 and p.x() < ncols and p.y() >= 0 and p.y() < nrows

    def camera_in_marker_frame(self, pose_id, cam_id):
        X_MC = self.X_CM(pose_id, cam_id).inverse()
        return gtsam.PinholeCameraCal3_S2(X_MC, self.camera_intrinsic)

    def generate_measurements(self, pose_id, cam_id):
        measurements = np.empty((0, 2), dtype=np.float32)
        camera = self.camera_in_marker_frame(pose_id, cam_id)
        for P in self.chess_board.get_3d_points():
            p = camera.project(gtsam.Point3(P))
            if self.sampler:
                eta = self.sampler.sample()
                p = gtsam.Point2(p.x() + eta[0], p.y() + eta[1])
            if not self.inside_image(p, self.img_nrows, self.img_ncols):
                measurements = np.empty((0, 2), dtype=np.float32)
                break
            measurements = np.append(measurements, [[p.x(), p.y()]], axis=0)
        return measurements

    def generate_all_measurements(self):
        """
        For each end-effector poses in gt_X_BEs, generate 2D measurements of
        the chess board points in each camera images
        Returns:
          all_measurements[end_effector_pose_i][camera_k][point_j]
        """
        all_measurements = []
        for pose_id in range(self.nr_end_effector_poses()):
            measurements_at_pose = []
            for cam_id in range(self.nr_cameras()):
                measurements_at_pose.append(
                    self.generate_measurements(pose_id, cam_id))
            all_measurements.append(measurements_at_pose)
        return all_measurements

    def draw_measurements(self, img, measurements, color):
        for j in range(len(measurements)):
            x = int(round(measurements[j][0]))
            y = int(round(measurements[j][1]))
            cv2.circle(img, (x, y), 5, color)

    def draw_chess_board(self, img, pose_id, cam_id):
        marker_corners = self.chess_board.generate_black_cell_corners()
        camera = self.camera_in_marker_frame(pose_id, cam_id)
        pts = np.zeros((4, 2), 'int32')
        for corners in marker_corners:
            for i in range(4):
                p = camera.project(corners[i])
                pts[i, 0] = int(round(p.x()))
                pts[i, 1] = int(round(p.y()))
            cv2.fillConvexPoly(img, pts, (0, 0, 0))

    def generate_test_image(self, pose_id, cam_id):
        img = np.full((self.img_nrows, self.img_ncols, 3), 255, np.uint8)
        self.draw_chess_board(img, pose_id, cam_id)
        return img

    def save_test_images(self, cam_id):
        for pose_id in range(len(self.gt_X_BEs)):
            img = self.generate_test_image(pose_id, cam_id)
            filename = 'data/img_{}_{}.jpg'.format(cam_id, pose_id)
            cv2.imwrite(filename, img)


class TestHandicalData(unittest.TestCase):
    def test_board_size(self):
        self.assertRaises(ValueError, ChessBoard, [4, 12], 0.1)
        self.assertRaises(ValueError, ChessBoard, [9, 3], 0.1)


def prepare_test_data(save_images=False):
    spaces = [None] * 6
    spaces[DIM.X] = np.linspace(-0.1, 0.1, 2)
    spaces[DIM.Y] = np.linspace(-0.1, 0.1, 2)
    spaces[DIM.Z] = np.linspace(-0.1, 0.1, 2)
    spaces[DIM.ROLL] = np.linspace(-30 * DEGREE, 30 * DEGREE, 2)
    spaces[DIM.PITCH] = np.linspace(-30 * DEGREE, 30 * DEGREE, 2)
    spaces[DIM.YAW] = np.linspace(0 * DEGREE, 360 * DEGREE, 2)

    board = ChessBoard([9, 4], 0.025)
    X_EM = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(-0.15 / 2., -0.125 / 2.,
                                                  0.0))
    data = HandicalTestData(board, spaces, None, X_EM, gtsam.Pose3())
    if save_images:
        data.save_test_images(0)
    return (data, board)


def debug_handical_generate_test_data():
    """
    Generate images captured from camera 0 for all end-effector poses
    and save them to files img_<pose_id>_<cam_id>.jpg for visual inspection
    """
    cam_id = 0
    data, board = prepare_test_data()
    for pose_id in range(data.nr_end_effector_poses()):
        if len(data.measurements(pose_id, cam_id)) > 0:
            img = data.generate_test_image(pose_id, cam_id)
            filename = 'img_{}_{}.jpg'.format(pose_id, cam_id)
            cv2.imwrite(filename, img)


class TestHandicalBackend(unittest.TestCase):
    def test_backend(self):
        data, board = prepare_test_data()
        backend = HandicalBackend(data.chess_board, data.camera_intrinsics(),
                                  data.init_X_EM())
        cam_id = 0
        for pose_id in range(data.nr_end_effector_poses()):
            measurements = data.measurements(pose_id, cam_id)
            if len(measurements) > 0:
                backend.add_measurements(data.X_BE(pose_id), cam_id,
                                         measurements,
                                         data.X_CM(pose_id, cam_id))
        optimizer = gtsam.GaussNewtonOptimizer(backend.graph, backend.values)
        results = optimizer.optimizeSafely()
        self.assertTrue(
            results.atPose3(0).equals(backend.values.atPose3(0), 1e-10))
        self.assertTrue(
            results.atPose3(3).equals(backend.values.atPose3(3), 1e-10))

        self.board = board


class TestHandicalFrontend(unittest.TestCase):
    def test_frontend(self):
        data, board = prepare_test_data()
        cam_id = 0
        for pose_id in range(data.nr_end_effector_poses()):
            img = data.generate_test_image(pose_id, cam_id)
            frontend = HandicalFrontend(data.chess_board)
            measurements, X_CM, _ = frontend.process(img,
                                                     data.camera_intrinsic)
            if len(measurements) == 0:
                continue
            cv2.imwrite('img_test.jpg', img)
            expected_measurements = data.measurements(pose_id, cam_id)
            np.testing.assert_almost_equal(measurements, expected_measurements,
                                           0)
            expected_X_CM = data.X_CM(pose_id, cam_id)
            check_cond = X_CM.equals(expected_X_CM, 1e-1)
            if not check_cond:
                print "actual: ", X_CM
                print "expected: ", expected_X_CM
            self.assertTrue(check_cond)


def to_int(coord):
    return (int(round(coord[0])), int(round(coord[1])))


def draw_2d_axes(img, imgpts):
    corner = to_int(imgpts[0])
    img = cv2.line(img, corner, to_int(imgpts[1]), (0, 0, 255), 5)
    img = cv2.line(img, corner, to_int(imgpts[2]), (0, 255, 0), 5)
    img = cv2.line(img, corner, to_int(imgpts[3]), (255, 0, 0), 5)
    return img


def draw_3d_axes(img, simple_cam):
    axis = [
        gtsam.Point3(0., 0., 0.),
        gtsam.Point3(1., 0., 0.),
        gtsam.Point3(0., 1., 0.),
        gtsam.Point3(0., 0., 1.)
    ]
    axis_imgpts = []
    for P in axis:
        p = simple_cam.project(P)
        axis_imgpts.append((p.x(), p.y()))
    img = draw_2d_axes(img, axis_imgpts)
    return img


def debug_handical_frontend(save=True, visualize=False):
    """
    Run the frontend on all images captured from camera 0
    then save the images with the coordinate frame of the estimated pose on it
    """
    data, board = prepare_test_data()
    frontend = HandicalFrontend(board)
    cam_id = 0
    for pose_id in range(data.nr_end_effector_poses()):
        print "pose_id: ", pose_id
        if len(data.measurements(pose_id, cam_id)):
            # generate input image
            img = np.full((480, 640, 3), 255, np.uint8)
            data.draw_chess_board(img, pose_id, cam_id)
            # run the frontend on this image
            measurements, X_CM = frontend.process(img, data.camera_intrinsic)
            expected_measurements = data.measurements(pose_id, cam_id)
            # draw the coordinate frame
            if len(measurements) > 0:
                # project 3D points to image plane
                simple_cam = gtsam.PinholeCameraCal3_S2(
                    X_CM.inverse(), data.camera_intrinsic)
                img = draw_3d_axes(img, simple_cam)
                data.draw_measurements(img, measurements, (0, 255, 0))
                data.draw_measurements(img, expected_measurements, (0, 0, 255))
                if save:
                    filename = 'frame_{}_{}.jpg'.format(pose_id, cam_id)
                    cv2.imwrite(filename, img)
                if visualize:
                    cv2.imshow('img', img)
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord('q'):
                        break
            else:
                print "Can't find chessboard"
    if visualize:
        cv2.destroyAllWindows()


class TestHandical(unittest.TestCase):
    def test_handical(self):
        data, board = prepare_test_data()

        handical = Handical(board.size, board.cell_size, data.init_X_EM(),
                            data.X_BEs(), data.camera_intrinsics())

        for pose_id in range(data.nr_end_effector_poses()):
            for cam_id in range(data.nr_cameras()):
                img = data.generate_test_image(pose_id, cam_id)
                handical.add_image(img, pose_id, cam_id)

        results, _ = handical.calibrate()

        for k in range(data.nr_cameras()):
            self.assertTrue(data.X_BCs()[k].equals(results.atPose3(k), 1e-3))
        self.assertTrue(
            results.atPose3(handical.backend.key_EM).equals(
                data.gt_X_EM, 1e-3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Testing and debugging Handical')
    parser.add_argument('--debug',
                        choices=['data', 'frontend', 'backend'],
                        required=False,
                        default=False)
    args = parser.parse_args()
    if args.debug == 'data':
        debug_handical_generate_test_data()
    elif args.debug == 'frontend':
        debug_handical_frontend()
    elif args.debug == 'backend':
        pass
    else:
        unittest.main()
