import cv2
import gtsam
import math
import numpy as np
import handical
import yaml
import argparse
import os


class ChessBoard:
    """
    This class stores the chess board's properties and its 3d points
    !!!IMPORTANT!!!: To avoid pose ambiguity, the chessboard
    must have one dimension to be odd and the other even.

    Attributes:
        size: the number of grid points on the x and y axis of the board
        cell_size: the size of a grid cell in physical unit
        P_Ms: 3D coordinates of the board corners, which order corresponds
              to OpenCV-detected 2D chessboard corners.
    """
    def __init__(self, board_size, cell_size):
        """
        Init the chess board with its size.
        @param board_size The number of points on the x and y axis of the board
        @param cell_size The size of a grid cell in physical unit (e.g. cm or
            inch) depending on your camera intrinsic parameters.
        """
        self.size = board_size
        if (self.size[0] + self.size[1]) % 2 != 1:
            raise ValueError("Invalid chessboard size: One must be odd"
                             "and the other even to avoid pose ambiguity!")
        self.cell_size = cell_size
        self.P_Ms = self.generate_3d_points()

    def generate_3d_points(self):
        """
        Genereate 3D coordinates of the chessboard corners for pose estimation.
        The order of those 3D points correspond to the list of 2D corners
        detected by OpenCV.
        """
        P_Ms = np.empty((0, 3), dtype=np.float32)
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                P_Ms = np.append(
                    P_Ms, [[x * self.cell_size, y * self.cell_size, 0.0]],
                    axis=0)
        # Rotate to opencv detection axis.
        # It is found emperically that compared to ours, OpenCV's chessboard is
        # rotated 90 degree around the longer axis and shifted back along the
        # shorter axis to the other chessboard corner.
        if self.size[0] > self.size[1]:
            X_cv_me = gtsam.Pose3(
                gtsam.Rot3.Rx(math.pi),
                gtsam.Point3(0., (self.size[1] - 1) * self.cell_size, 0.))
        else:
            X_cv_me = gtsam.Pose3(
                gtsam.Rot3.Ry(math.pi),
                gtsam.Point3((self.size[0] - 1) * self.cell_size, 0., 0.))
        for i in range(len(P_Ms)):
            P_Ms[i] = X_cv_me.transformFrom(gtsam.Point3(P_Ms[i])).vector()

        return P_Ms

    def get_3d_points(self):
        return self.P_Ms

    def generate_black_cell_corners(self):
        """
        Coordinates of black cells' corners to generate images for unittests.
        """
        black_cells = []
        offset_x = [-1., 0., 0., -1.]
        offset_y = [-1., -1., 0., 0.]
        for y in range(self.size[1] + 1):
            for x in range(self.size[0] + 1):
                if (x + y) % 2 == 0:
                    corners = [None] * 4
                    for i in range(4):
                        corners[i] = gtsam.Point3(
                            (x + offset_x[i]) * self.cell_size,
                            (y + offset_y[i]) * self.cell_size, 0.0)
                    black_cells.append(corners)
        return black_cells


class HandicalBackend:
    """
    Backend of the calibration process.
    """
    def __init__(self, board, intrinsics, init_X_EM):
        """
        Construct the backend.

        @param board the ChessBoard information
        @param intrinsics a list of gtsam.Cal3_S2 for intrinsic params of the
            cameras.
        @param init_X_EM [gtsam.Pose3] initial guess of the chessboard marker
                       pose in the end effector frame
        """
        self.board = board
        self.intrinsics = intrinsics
        self.key_EM = len(intrinsics)
        self.init_X_EM = init_X_EM
        self.values = gtsam.Values()
        self.values.insert(self.key_EM, init_X_EM)
        self.graph = gtsam.NonlinearFactorGraph()

    def add_measurements(self,
                         X_BE,
                         cam_id,
                         measurements,
                         init_X_CM,
                         radius=0.5):
        """
        Add a list of corner measurements detected by the frontend from an
        image.

        @param X_BE [gtsam.Pose3] the known end-effector pose in the robot's
            base coordinate frame.
        @param cam_id id of the camera capturing the image.
        @param measurements a list of 2d points of the chessboard's corners
            detected by the front-end.
        @param init_X_CM [gtsam.Pose3] the initial pose of the chessboard
            marker in the camera frame.
        """
        if len(measurements) == 0:
            return
        P_Ms = self.board.get_3d_points()
        noise = gtsam.noiseModel_Isotropic.Sigma(2, radius)
        for i in range(len(measurements)):
            P = gtsam.Point3(P_Ms[i])
            p = gtsam.Point2(measurements[i])
            self.graph.add(
                handical.HandiCalFactorCal3_S2(cam_id, self.key_EM,
                                               self.intrinsics[cam_id], X_BE,
                                               P, p, noise))
        X_BC = X_BE.compose(self.init_X_EM).compose(init_X_CM.inverse())
        print("X_BC:")
        print X_BC
        if not self.values.exists(cam_id):
            self.values.insert(cam_id, X_BC)
        print("Error: ", self.graph.error(self.values))

    def optimize(self):
        print("Initial values:")
        print(self.values)
        print(self.graph.error(self.values))
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values)
        # optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.values)
        results = optimizer.optimize()
        return (results, self.graph.error(results))


def cv2gtsamPose(rvecs, tvecs):
    """
    Convert an OpenCV pose, e.g. returned by cv2.solvePnP, to a gtsam Pose3.
    """
    return gtsam.Pose3(gtsam.Rot3.Expmap(rvecs), gtsam.Point3(tvecs.ravel()))


def _pdist(p1, p2):
    """
    Distance bwt two points. p1 = (x, y), p2 = (x, y)
    """
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


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
        gtsam.Point3(0.1, 0., 0.),
        gtsam.Point3(0., 0.1, 0.),
        gtsam.Point3(0., 0., 0.1)
    ]
    axis_imgpts = []
    for P in axis:
        p = simple_cam.project(P)
        axis_imgpts.append((p.x(), p.y()))
    img = draw_2d_axes(img, axis_imgpts)
    return img


class HandicalFrontend:
    """
    This class detects the chessboard and its grid points in the input images,
    and estimates the intial X_CMs to pass to the backend.
    """
    def __init__(self, board):
        self.board = board

    def process(self, img, intrinsic, pose_id=-1, cam_id=-1):
        """
        Detect chessboard corners and estimated its initial pose from an image.

        @param img: the input image in BGR.
        @param intrinsic: gtsam.Cal3_S2 intrinsic parameters of the camera.
        @param pose_id, cam_id: if both >=0, both are used for saving debug
            images.

        @returns A tuple of a set of points (empty if not recognize),
            the associating T_CM poses (nondetermined if no point detected),
            and the radius used for subpixel corner search window.
        """
        measurements = []
        X_CM = gtsam.Pose3()
        ret, corners = cv2.findChessboardCorners(
            img, (self.board.size[0], self.board.size[1]),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        K = intrinsic.matrix()
        dist = np.zeros(5)
        radius = 0.5
        if ret == True:
            # subpixel accuracy
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        1000, 0.1)

            # Copied from https://github.com/ros-perception/image_pipeline/blob/cf1a18b5f4265cb0589fc536c2295d6838332444/camera_calibration/src/camera_calibration/calibrator.py#L173
            # Use a radius of half the minimum distance between corners. This
            # should be large enough to snap to the correct corner, but not so
            # large as to include a wrong corner in the search window.
            min_distance = float("inf")
            nrows = self.board.size[0]
            ncols = self.board.size[1]
            for row in range(nrows):
                for col in range(ncols - 1):
                    index = row * ncols + col
                    min_distance = min(
                        min_distance,
                        _pdist(corners[index, 0], corners[index + 1, 0]))
            for row in range(nrows - 1):
                for col in range(ncols):
                    index = row * ncols + col
                    min_distance = min(
                        min_distance,
                        _pdist(corners[index, 0], corners[index + ncols, 0]))
            radius = int(math.ceil(min_distance * 0.5))

            refined_corners = cv2.cornerSubPix(gray, corners, (radius, radius),
                                               (-1, -1), criteria)
            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(self.board.get_3d_points(),
                                             refined_corners,
                                             K,
                                             dist,
                                             flags=0)
            if ret:
                measurements = corners[:, 0, :]
                X_CM = cv2gtsamPose(rvecs, tvecs)

            if pose_id >= 0 and cam_id >= 0:
                dbg_img = img.copy()
                cv2.drawChessboardCorners(dbg_img, self.board.size,
                                          refined_corners, ret)
                simple_cam = gtsam.PinholeCameraCal3_S2(
                    X_CM.inverse(), intrinsic)
                draw_3d_axes(dbg_img, simple_cam)
                cv2.imwrite(
                    'debug/detected_pose{}_{}.png'.format(pose_id, cam_id),
                    dbg_img)

        else:
            print("Can't find corners")

        return measurements, X_CM, radius


def draw_points(img, Ps, X_CM, intrinsic):
    simple_cam = gtsam.PinholeCameraCal3_S2(X_CM.inverse(), intrinsic)
    for i in range(len(Ps)):
        P = gtsam.Point3(Ps[i][0], Ps[i][1], Ps[i][2])
        p = simple_cam.project(P)
        x = int(round(p.x()))
        y = int(round(p.y()))
        cv2.circle(img, (x, y), 3, (100, 100, 100), thickness=2)


class Handical:
    """
    Full hand-eye calibration pipeline.
    """
    def __init__(self, board_size, cell_size, init_X_EM, X_BEs, intrinsics):
        """
        Construct the pipeline.

        @param board_size, cell_size: parameters of the \sa ChessBoard.
        @param init_X_EM: initial estimate of the board in the effector frame.
        @param X_BEs: list of known end-effector poses in gtsam.Pose3
        @param intrinsics: list of intrinsic params of cameras in gtsam.Cal3_S2
        """
        self.board = ChessBoard(board_size, cell_size)
        self.intrinsics = intrinsics
        self.X_BEs = X_BEs
        self.frontend = HandicalFrontend(self.board)
        self.backend = HandicalBackend(self.board, self.intrinsics, init_X_EM)

    def add_image(self, img, pose_id, cam_id, dbg=False):
        """
        Add an image captured by camera cam_id when the end-effector pose
        is pose_id.
        """
        print("Frontend process pose {} cam {}".format(pose_id, cam_id))
        if dbg:
            dbg_pose_id, dbg_cam_id = pose_id, cam_id
        else:
            dbg_pose_id, dbg_cam_id = -1, -1
        measurements, X_CM, radius = self.frontend.process(
            img, self.intrinsics[cam_id], dbg_pose_id, dbg_cam_id)
        if len(measurements) > 0:
            self.backend.add_measurements(self.X_BEs[pose_id], cam_id,
                                          measurements, X_CM, radius)
            if dbg:
                X_BC = self.backend.values.atPose3(cam_id)
                X_BE = self.X_BEs[pose_id]
                X_EM = self.backend.init_X_EM
                X_CM2 = X_BC.inverse().compose(X_BE.compose(X_EM))
                draw_points(img, self.board.get_3d_points(), X_CM2,
                            self.intrinsics[cam_id])
                cv2.imwrite(
                    'debug/predicted_points{}_{}.png'.format(pose_id, cam_id),
                    img)

    def calibrate(self):
        """
        Calibrate with all the data.
        """
        return self.backend.optimize()


class HandicalWristMount(Handical):
    """
    Full hand-eye calibration pipeline.

    Fixed mount
    CM = CB * BE * EM  (1)

        - BE is known from forward kinematics
        - ME we have a good initial guess (at least for rotations)
        - CM comes from the corder detector + PnP solve
        - CB is unknown

    The factor is

    MC = (BE * EM)^(-1) * BC (2)

    The optimizer returns

    Wrist Mount
    CM = CE * EB * BM  (3)

        - EB is known from forward kinematics
        - MB is unknown
        - CE we have an initial guess from where the camera is mounted

    inverting equation (3) gives
    MC = (EB*BM)^(-1) * EC (4)


    Now this is in the same form as equation (2). Thus all we need to do is
    match up terms

    Fixed Mount ----> Wrist Mount

    MC ---> MC
    BE ---> EB
    EM ---> BM
    BC ---> EC

    basically we are swapping E <--> B, C and M stay the same. Just constructs
    a Handical object with the same list


    """
    def __init__(self, board_size, cell_size, init_X_BM, X_EB_list,
                 intrinsics):
        """
        Construct the pipeline.

        Parameters:
            board_size: parameters of the board
            cell_size: cell size on the chessboard

            init_X_BM: gtsam.Pose3
                initial guess of transform from marker to base


            X_EB_list: list of gtsam.Pose3
                list of known transforms from base to end-effector

            intrinsics: list of intrinsic params of cameras in gtsam.Cal3_S2
        """
        Handical.__init__(self, board_size, cell_size, init_X_BM, X_EB_list,
                          intrinsics)


def read_intrinsic(filename):
    """
    Read camera intrinsic parameter from a yaml file.
    Return:
        A gtsam.Cal3_S2 object with distortion parameters
    """
    with open(filename, 'r') as stream:
        try:
            data = yaml.load(stream)
        except yaml.YAMLError as e:
            print e
    K = data['camera_matrix']['data']
    fx, fy, s, cx, cy = K[0], K[4], K[1], K[2], K[5]
    return gtsam.Cal3_S2(fx, fy, s, cx, cy)


def read_config(config_fn):
    with open(config_fn, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as e:
            print e
    return config


def to_pose(xyzrpy):
    x, y, z, r, p, w = xyzrpy
    R = gtsam.Rot3.RzRyRx(r, p, w)
    return gtsam.Pose3(R, gtsam.Point3(x, y, z))


def dict_to_pose(d):
    """
    Parameters
    ------------
    d: dict
        of the form

        quaternion:
          w: 0.2224219193342252
          x: 0.5352550641071433
          y: 0.8012390219799749
          z: 0.14848075903601315
        translation:
          x: 0.3509808820978816
          y: 3.6105292560773724e-05
          z: 0.5842474420103821

    Returns
    ---------
    gtsam.Pose3

    """
    x = d['translation']['x']
    y = d['translation']['y']
    z = d['translation']['z']

    q_w = d['quaternion']['w']
    q_x = d['quaternion']['x']
    q_y = d['quaternion']['y']
    q_z = d['quaternion']['z']

    R = gtsam.Rot3.Quaternion(q_w, q_x, q_y, q_z)
    return gtsam.Pose3(R, gtsam.Point3(x, y, z))


def read_ee_poses(ee_poses_fn):
    ee_poses = []
    with open(ee_poses_fn, 'r') as f:
        for line in f:
            ixyzrpw = [float(a) for a in line.split()]
            ee_poses.append(to_pose(ixyzrpw[1:]))
    return ee_poses


def read_image_to_gray(fn):
    img = cv2.imread(fn, -1)
    if img.ndim == 2 and img.dtype == 'uint16':
        return np.round(img.astype('float') / 1022. * 255).astype('uint8')
    elif img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_image_as_grayscale(image_filename):
    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    return img


def pose_to_list(pose):
    rpw = pose.rotation().rpy().tolist()
    xyz = pose.translation()
    return [xyz.x(), xyz.y(), xyz.z(), rpw[0], rpw[1], rpw[2]]


def save_results(out_fn, results, cam_keys, error):
    res = {}
    for cam, key in cam_keys.iteritems():
        pose = results.atPose3(key)
        res[cam] = pose_to_list(pose)
    ee_key = len(cam_keys)
    res['X_EM'] = pose_to_list(results.atPose3(ee_key))
    res['error'] = error

    # import pdb; pdb.set_trace()
    with open(out_fn, 'w') as stream:
        yaml.dump(res, stream)


def main(input_path, dbg=False):
    """
    @param input_path Path to input data. See "--help" for detail.
    @param dbg Whether or not to output images for debugging purposes.
        Default: False
        Images will be saved to the folder named "debug",
        so make sure it's created.
    """
    config_fn = input_path + "/config.yaml"
    config = read_config(config_fn)
    board = ChessBoard((config['board']['nrows'], config['board']['ncols']),
                       config['board']['cell_size'])
    path_to_intrinsics = config['path_to_intrinsics']
    path_to_extrinsics = input_path
    path_to_images = path_to_extrinsics + '/imgs'
    ee_poses_file = path_to_extrinsics + '/final_ee_poses.txt'
    extrinsic_out_file = path_to_extrinsics + '/extrinsics.txt'
    X_BEs = read_ee_poses(ee_poses_file)
    init_X_EM = to_pose(config['init_X_EM'])
    cam_intrinsics = []
    cam_keys = {}
    key = 0
    for cam_name in config['camera_names']:
        intr_fn = path_to_intrinsics + '/' + cam_name + '/ost.yaml'
        cam_intrinsics.append(read_intrinsic(intr_fn))
        cam_keys[cam_name] = key
        key += 1

    handical = Handical(board.size, board.cell_size, init_X_EM, X_BEs,
                        cam_intrinsics)

    for pose_id in range(len(X_BEs)):
        print("X_BE: ")
        print X_BEs[pose_id]
        for cam_name in config['camera_names']:
            if ('exclude' in config) and (cam_name in config['exclude']) and (
                    pose_id in config['exclude'][cam_name]):
                continue
            img_fn = path_to_images + '/_' + cam_name + '_image' + str(
                pose_id) + '.png'
            img = read_image_to_gray(img_fn)
            handical.add_image(img, pose_id, cam_keys[cam_name], dbg)

    print "graph size: ", handical.backend.graph.size()

    results, error = handical.calibrate()
    print("Calibration results:")
    print results, error
    save_results(extrinsic_out_file, results, cam_keys, error)


def wrist_mounted_calibration(calibration_data_folder, debug=False):
    """
    Parse our config file and run handical.
    """

    extrinsics_out_file = os.path.join(calibration_data_folder,
                                       'extrinsics.txt')

    config_filename = os.path.join(calibration_data_folder, 'robot_data.yaml')
    config = read_config(config_filename)

    ncols = config['header']['target']['width']
    nrows = config['header']['target']['height']
    cell_size = config['header']['target']['square_edge_length']

    board = ChessBoard((nrows, ncols), cell_size)

    # read the X_BE poses
    # X_BE_poses = []
    X_EB_list = []

    for idx, data in enumerate(config['data_list']):
        ee_to_base = dict_to_pose(data['hand_frame'])
        base_to_ee = ee_to_base.inverse()
        # X_BE_poses.append(ee_to_base)
        X_EB_list.append(base_to_ee)

    init_X_BM = dict_to_pose(
        config['header']['target']['transform_to_robot_base'])

    cam_intrinsics = []
    cam_keys = {}

    image_type = config['header']['image_type']
    print "image_type: ", image_type
    # we only have one camera
    if image_type == "ir":
        intrinsics_prefix = "depth"
    else:
        intrinsics_prefix = image_type
    intrinsics_filename = os.path.join(calibration_data_folder,
                                       intrinsics_prefix + "_camera_info.yaml")

    intrinsics = read_intrinsic(intrinsics_filename)

    key = 0
    camera_name = config['header']['camera']
    cam_intrinsics.append(intrinsics)
    cam_keys[camera_name] = key

    handical = HandicalWristMount(board.size, board.cell_size, init_X_BM,
                                  X_EB_list, cam_intrinsics)

    for pose_id, X_BE in enumerate(X_EB_list):
        print "X_BE: ", X_BE
        img_filename = os.path.join(
            calibration_data_folder,
            config['data_list'][pose_id]['images'][image_type]['filename'])
        img = load_image_as_grayscale(img_filename)
        handical.add_image(img, pose_id, cam_keys[camera_name], debug)

    print "graph size: ", handical.backend.graph.size()

    results, error = handical.calibrate()
    print("Calibration results:")
    print results, error
    save_results(extrinsics_out_file, results, cam_keys, error)
    calibration_results = calibration_results_to_dict(results)
    return calibration_results


def gtsam_Pose3_to_dict(pose):
    xyz = pose.translation()
    pose_dict = dict()
    pose_dict["translation"] = {"x": xyz.x(), "y": xyz.y(), "z": xyz.z()}

    # This returns a numpy array.
    quat = pose.rotation().quaternion().tolist()

    # Assuming quaternion is ordered w, x, y, z.
    pose_dict["quaternion"] = {
        "w": quat[0],
        "x": quat[1],
        "y": quat[2],
        "z": quat[3]
    }

    return pose_dict


def calibration_results_to_dict(results):
    """
    first result is camera to wrist mount
    """
    camera_to_wrist_dict = gtsam_Pose3_to_dict(results.atPose3(0))
    print camera_to_wrist_dict
    """
    second result is marker to base
    """
    print "results.atPose3(1)", results.atPose3(1)
    ## don't have a use for this for now, ... could be useful later
    marker_to_base_dict = gtsam_Pose3_to_dict(results.atPose3(1))

    calibration_results = dict()
    calibration_results["camera_to_wrist"] = camera_to_wrist_dict
    calibration_results["marker_to_base"] = marker_to_base_dict
    return calibration_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extrinsic hand-eye calibration',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--input_path',
        required=True,
        help=
        """Path to a input data folder which must include the following items:

        1. A "config.yaml" file with the following content:
######################
    board:  # The chessboard information
      nrows: 7  # number of inner grid points on x axis
      ncols: 6  # number of inner grid points on y axis
      cell_size: 0.027  # size of a grid cell, must agree with intrinsic unit

    init_X_EM: [0., 0., 0., 0., 0., 0.]
    # initial guess of the marker chessboard in an end-effector frame in x,y,z,r,p,yaw

    camera_names:
      [astrapro_yellow_ir,
      astrapro_yellow_rgb,
      astrapro_pink_ir,
      astrapro_pink_rgb]

    # The folowing param is used to load intrinsic data files.
    # The convention for the intrinsic data file follows the results of the
    # camera_calibration ROS package: <path_to_intrinsics>/<camera_name>/ost.yaml
    path_to_intrinsics: "/home/duynguyen/data/calibration"

    # The following is optional. It's used to specified bad images with bad point
    # detection that you don't want to use.
    exclude:
      astrapro_black_ir: [1]
######################

        2. A "final_ee_poses.txt" file, storing end-effector poses during the capture
        with the following content on each line:
                <ee_pose_id(int)> x y z roll pitch yaw

        3. An "imgs" folder storing images for extrinsic calibration
        with the following naming convention:
               "_<camera_name><ee_pose_id>.png"
        """)

    parser.add_argument(
        '--debug',
        action='store_true',
        help="""If specified: debug images will be saved to the "debug" folder.
Make sure the folder exists.
             """)
    args = parser.parse_args()
    main(args.input_path, args.debug)
    cv2.destroyAllWindows()
