#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <fstream>

#include <opencv2/opencv.hpp>

class camera_info
{
    public:
    // Read Camera information
    std::string cam_name;
    std::string depth_name;
    int projection;
    double fx;
    double fy;
    double ox;
    double oy;
    cv::Vec3d rot;
    cv::Vec3d tran;
    double depth_min;
    double depth_max;
    int width;
    int height;
    int bit_depth_image;
    int bit_depth_depth;
    private:
};

class spherical_dibr
{
public:

    cv::Vec3d rot2eular(cv::Mat rot_mat);
    cv::Mat eular2rot(cv::Vec3d theta);
    void render(cv::Mat& im, cv::Mat& depth_double
                , cv::Mat& rot_mat, cv::Vec3d t_vec
                , camera_info& cam_info, camera_info& vt_cam_info);

    cv::Mat im_out_forward;
    cv::Mat im_out_inverse_median;
    cv::Mat im_out_inverse_closing;
    cv::Mat depth_out_forward;
    cv::Mat depth_out_median;
    cv::Mat depth_out_closing;
#define HOLE_FILLING
#ifdef HOLE_FILLING
    cv::Mat mask;  // binary mask (1: has value, 0: not) 
#endif

private:
    cv::Vec3d plane2cart(const cv::Vec3d& plane_pixel, double fx, double fy, double ox, double oy);
    cv::Vec3d applyTR(const cv::Vec3d& vec_cartesian, const cv::Mat& rot_mat, const cv::Vec3d t_vec);
    cv::Vec3d applyRT(const cv::Vec3d& vec_cartesian, const cv::Mat& rot_mat, const cv::Vec3d t_vec);
    cv::Vec3d cart2plane(const cv::Vec3d& cart, double fx, double fy, double ox, double oy);
    cv::Mat median_depth(cv::Mat& depth_double, int size);
    cv::Mat closing_depth(cv::Mat& depth_double, int size);
    void image_depth_forward_mapping(cv::Mat& im, cv::Mat& depth_double
                                    , cv::Mat& rot_mat, cv::Vec3d t_vec
                                    , cv::Mat& im_out, cv::Mat& depth_out_double
                                    , camera_info& cam_info, camera_info& vt_cam_info);
    void image_depth_inverse_mapping(cv::Mat& im, cv::Mat& depth_out_double
                                    , cv::Mat& rot_mat_inv, cv::Vec3d t_vec_inv
                                    , cv::Mat& im_out
                                    , camera_info& cam_info, camera_info& vt_cam_info);
    cv::Mat invert_depth(cv::Mat& depth_double, double min_dist, double max_dist);
    cv::Mat revert_depth(cv::Mat& depth_inverted, double min_dist, double max_dist);
};
