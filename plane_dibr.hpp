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
    void render(cv::Mat& im, cv::Mat& depth_double, double min_dist, double max_dist
            , cv::Mat& rot_mat, cv::Vec3d t_vec, double focal, double ox, double oy);

    cv::Mat im_out_forward;
    cv::Mat im_out_inverse_median;
    cv::Mat im_out_inverse_closing;
    cv::Mat depth_out_forward;
    cv::Mat depth_out_median;
    cv::Mat depth_out_closing;

private:
    cv::Vec3d plane2cart(const cv::Vec3d& plane_pixel, double focal, double ox, double oy);
    cv::Vec3d applyTR(const cv::Vec3d& vec_cartesian, const cv::Mat& rot_mat, const cv::Vec3d t_vec);
    cv::Vec3d applyRT(const cv::Vec3d& vec_cartesian, const cv::Mat& rot_mat, const cv::Vec3d t_vec);
    cv::Vec3d cart2plane(const cv::Vec3d& cart, double focal, double ox, double oy);
    cv::Vec3d tr_pixel(const cv::Vec3d& in_vec, const cv::Vec3d& t_vec, const cv::Mat& rot_mat, double focal, double ox, double oy);
    cv::Vec3d rt_pixel(const cv::Vec3d& in_vec, const cv::Vec3d& t_vec, const cv::Mat& rot_mat, double focal, double ox, double oy);
    cv::Mat median_depth(cv::Mat& depth_double, int size);
    cv::Mat closing_depth(cv::Mat& depth_double, int size);
    void image_depth_forward_mapping(cv::Mat& im, cv::Mat& depth_double, cv::Mat& rot_mat, cv::Vec3d t_vec, cv::Mat& im_out, cv::Mat& depth_out_double, double focal, double ox, double oy);
    void image_depth_inverse_mapping(cv::Mat& im, cv::Mat& depth_out_double, cv::Mat& rot_mat_inv, cv::Vec3d t_vec_inv, cv::Mat& im_out, double focal, double ox, double oy);
    cv::Mat invert_depth(cv::Mat& depth_double, double min_dist, double max_dist);
    cv::Mat revert_depth(cv::Mat& depth_inverted, double min_dist, double max_dist);
};