import handicalpy
import os
import spartan.utils.utils as spartanUtils

if __name__ == "__main__":
    spartan_source_dir = os.getenv("SPARTAN_SOURCE_DIR")
    
    """
	set args
    """
    rgb_calibration_data_folder = os.path.join(spartan_source_dir, 'calibration_data', '20180223-203939_rgb')
    depth_calibration_data_folder = os.path.join(spartan_source_dir, 'calibration_data', '20180223-205019_ir')
    camera_info_filename_destination = "src/catkin_projects/camera_config/data/carmine_1/master/camera_info.yaml"
    """
    """

    camera_info_filename_destination = os.path.join(spartan_source_dir, camera_info_filename_destination)

    camera_info_dict = dict()
    camera_info_dict["header"] = dict()
    camera_info_dict["header"]["calibration_date"] = "ignored"
    camera_info_dict["header"]["camera_name"] = "carmine_1"
    camera_info_dict["header"]["serial_number"] = 0


    rgb_calibration_results = handicalpy.wrist_mounted_calibration(rgb_calibration_data_folder)
    print ""
    print ""
    print "rgb_calibration_results:", rgb_calibration_results


    camera_info_dict["rgb"] = dict()
    camera_info_dict["rgb"]["extrinsics"] = dict()
    # todo: read the base link from the robot_data.yaml
    camera_info_dict["rgb"]["extrinsics"]["reference_link_name"] = "wsg_50_base_link"
    camera_info_dict["rgb"]["extrinsics"]["transform_to_reference_link"] = rgb_calibration_results["camera_to_wrist"]


    depth_calibration_results = handicalpy.wrist_mounted_calibration(depth_calibration_data_folder)
    print ""
    print ""
    print "depth_calibration_results:", depth_calibration_results


    camera_info_dict["depth"] = dict()
    camera_info_dict["depth"]["extrinsics"] = dict()
    # todo: read the base link from the robot_data.yaml
    camera_info_dict["depth"]["extrinsics"]["reference_link_name"] = "wsg_50_base_link"
    camera_info_dict["depth"]["extrinsics"]["transform_to_reference_link"] = depth_calibration_results["camera_to_wrist"]

    spartanUtils.saveToYaml(camera_info_dict, camera_info_filename_destination)