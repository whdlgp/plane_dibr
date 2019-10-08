#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class six_dof_reader
{
    public:
    cv::Mat read_yuv(std::string image_name, int width, int height, int bit_depth);
    cv::Mat depth_double_yuv(std::string image_name, int width, int height, int bit_depth, double min, double max);

    private:
    void get_yuv_chan_10bit(std::ifstream& file, cv::Mat& y_mat, cv::Mat& u_mat, cv::Mat& v_mat, int width, int height);
    void get_yuv_chan_8bit(std::ifstream& file, cv::Mat& y_mat, cv::Mat& u_mat, cv::Mat& v_mat, int width, int height);
    cv::Mat merge_yuv_chan(cv::Mat& y_mat, cv::Mat& u_mat, cv::Mat& v_mat);
};