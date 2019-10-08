#include "plane_dibr.hpp"

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

using namespace std;
using namespace cv;

template <typename T>
T clip(T n, T lower, T upper)
{
    n = ( n > lower ) * n + !( n > lower ) * lower;
    return ( n < upper ) * n + !( n < upper ) * upper;
}

// XYZ-eular rotation 
Mat spherical_dibr::eular2rot(Vec3d theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
     
    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
     
    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
     
    // Combined rotation matrix
    Mat R = R_x * R_y * R_z;
     
    return R;
}

// Rotation matrix to rotation vector in XYZ-eular order
Vec3d spherical_dibr::rot2eular(Mat R)
{
    double sy = sqrt(R.at<double>(2,2) * R.at<double>(2,2) +  R.at<double>(1,2) * R.at<double>(1,2) );
 
    bool singular = sy < 1e-6; // If
 
    double x, y, z;
    if (!singular)
    {
        x = atan2(-R.at<double>(1,2) , R.at<double>(2,2));
        y = atan2(R.at<double>(0,2), sy);
        z = atan2(-R.at<double>(0,1), R.at<double>(0,0));
    }
    else
    {
        x = 0;
        y = atan2(R.at<double>(0,2), sy);
        z = atan2(-R.at<double>(0,1), R.at<double>(0,0));
    }
    return Vec3d(x, y, z);
}

Vec3d spherical_dibr::plane2cart(const Vec3d& plane_pixel, double focal, double ox, double oy)
{
    // pixel coordinate to cartesian coordinate
    //    z
    //   /
    //  /
    // o----- x
    // |
    // |
    // y
    Vec3d tmp;
    tmp[0] = (plane_pixel[1] - ox)*plane_pixel[2]/focal;
    tmp[1] = (plane_pixel[0] - oy)*plane_pixel[2]/focal;
    tmp[2] = plane_pixel[2];

    Vec3d vec;
    vec[0] = -tmp[2];
    vec[1] = -tmp[0];
    vec[2] = -tmp[1];

    return vec;
}

Vec3d spherical_dibr::cart2plane(const Vec3d& cart, double focal, double ox, double oy)
{
    Vec3d tmp;
    tmp[0] = -cart[1];
    tmp[1] = -cart[2];
    tmp[2] = -cart[0];

    Vec3d pixel;
    pixel[0] = tmp[1]*focal/tmp[2] + oy;
    pixel[1] = tmp[0]*focal/tmp[2] + ox;
    pixel[2] = tmp[2];

    return pixel;
}

Vec3d spherical_dibr::applyTR(const Vec3d& vec_cartesian, const Mat& rot_mat, const Vec3d t_vec)
{
    double* rot_mat_data = (double*)rot_mat.data;
    Vec3d vec_cartesian_tran;
    vec_cartesian_tran[0] = vec_cartesian[0] - t_vec[0];
    vec_cartesian_tran[1] = vec_cartesian[1] - t_vec[1];
    vec_cartesian_tran[2] = vec_cartesian[2] - t_vec[2];

    Vec3d vec_cartesian_rot;
    vec_cartesian_rot[0] = rot_mat_data[0]*vec_cartesian_tran[0] + rot_mat_data[1]*vec_cartesian_tran[1] + rot_mat_data[2]*vec_cartesian_tran[2];
    vec_cartesian_rot[1] = rot_mat_data[3]*vec_cartesian_tran[0] + rot_mat_data[4]*vec_cartesian_tran[1] + rot_mat_data[5]*vec_cartesian_tran[2];
    vec_cartesian_rot[2] = rot_mat_data[6]*vec_cartesian_tran[0] + rot_mat_data[7]*vec_cartesian_tran[1] + rot_mat_data[8]*vec_cartesian_tran[2];

    return vec_cartesian_rot;
}

Vec3d spherical_dibr::applyRT(const Vec3d& vec_cartesian, const Mat& rot_mat, const Vec3d t_vec)
{
    double* rot_mat_data = (double*)rot_mat.data;
    Vec3d vec_cartesian_rot;
    vec_cartesian_rot[0] = rot_mat_data[0]*vec_cartesian[0] + rot_mat_data[1]*vec_cartesian[1] + rot_mat_data[2]*vec_cartesian[2];
    vec_cartesian_rot[1] = rot_mat_data[3]*vec_cartesian[0] + rot_mat_data[4]*vec_cartesian[1] + rot_mat_data[5]*vec_cartesian[2];
    vec_cartesian_rot[2] = rot_mat_data[6]*vec_cartesian[0] + rot_mat_data[7]*vec_cartesian[1] + rot_mat_data[8]*vec_cartesian[2];

    Vec3d vec_cartesian_tran;
    vec_cartesian_tran[0] = vec_cartesian_rot[0] - t_vec[0];
    vec_cartesian_tran[1] = vec_cartesian_rot[1] - t_vec[1];
    vec_cartesian_tran[2] = vec_cartesian_rot[2] - t_vec[2];

    return vec_cartesian_tran;
}

// rotate and translate pixel, in_vec as input(row, col)
Vec3d spherical_dibr::rt_pixel(const Vec3d& in_vec, const Vec3d& t_vec, const Mat& rot_mat, double focal, double ox, double oy)
{
    Vec3d vec_cartesian = plane2cart(in_vec, focal, ox, oy);
    Vec3d vec_cartesian_rot = applyRT(vec_cartesian, rot_mat, t_vec);
    Vec3d vec_pixel = cart2plane(vec_cartesian_rot, focal, ox, oy);

    return vec_pixel;
}

// translate and rotate pixel, in_vec as input(row, col)
Vec3d spherical_dibr::tr_pixel(const Vec3d& in_vec, const Vec3d& t_vec, const Mat& rot_mat, double focal, double ox, double oy)
{
    Vec3d vec_cartesian = plane2cart(in_vec, focal, ox, oy);
    Vec3d vec_cartesian_rot = applyTR(vec_cartesian, rot_mat, t_vec);
    Vec3d vec_pixel = cart2plane(vec_cartesian_rot, focal, ox, oy);

    return vec_pixel;
}

Mat spherical_dibr::median_depth(Mat& depth_double, int size)
{
    Mat depth_float, depth_double_median, depth_float_median;
    depth_double.convertTo(depth_float, CV_32FC1);
    medianBlur(depth_float, depth_float_median, size);
    depth_float_median.convertTo(depth_double_median, CV_64FC1);
    return depth_double_median;
}

Mat spherical_dibr::closing_depth(Mat& depth_double, int size)
{
    Mat depth_float, depth_double_median, depth_float_median;
    depth_double.convertTo(depth_float, CV_32FC1);
    Mat element(size, size, CV_32FC1, Scalar(1.0));
    morphologyEx(depth_float, depth_float_median, CV_MOP_CLOSE, element);
    depth_float_median.convertTo(depth_double_median, CV_64FC1);
    return depth_double_median;
}

void spherical_dibr::image_depth_forward_mapping(Mat& im, Mat& depth_double, Mat& rot_mat, Vec3d t_vec, Mat& im_out, Mat& depth_out_double, double focal, double ox, double oy)
{
    int im_width = im.cols;
    int im_height = im.rows;

	Mat srci(im_height, im_width, CV_32F);
	Mat srcj(im_height, im_width, CV_32F);
    float* srci_data = (float*)srci.data;
    float* srcj_data = (float*)srcj.data;

    im_out.create(im.rows, im.cols, im.type());
    depth_out_double.create(depth_double.rows, depth_double.cols, depth_double.type());
    Vec3w* im_data = (Vec3w*)im.data;
    Vec3w* im_out_data = (Vec3w*)im_out.data;
    double* depth_data = (double*)depth_double.data;
    double* depth_out_double_data = (double*)depth_out_double.data;

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < im_height; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            // forward warping
            Vec3d vec_pixel = tr_pixel(Vec3d(i, j, depth_data[i*im.cols + j])
                                    , t_vec
                                    , rot_mat
                                    , focal, ox, oy);
            int dist_i = vec_pixel[0];
            int dist_j = vec_pixel[1];
            dist_i = clip(dist_i, 0, im_height);
            dist_j = clip(dist_j, 0, im_width);

            srci_data[dist_i*im.cols + dist_j] = i;
            srcj_data[dist_i*im.cols + dist_j] = j;
            double dist_depth = vec_pixel[2];
            if((dist_i >= 0) && (dist_j >= 0) && (dist_i < im_height) && (dist_j < im_width))
            {
                if(depth_out_double_data[dist_i*im.cols + dist_j] == 0)
                    depth_out_double_data[dist_i*im.cols + dist_j] = dist_depth;
                else if(depth_out_double_data[dist_i*im.cols + dist_j] > dist_depth)
                    depth_out_double_data[dist_i*im.cols + dist_j] = dist_depth;
            }
        }
    }

    remap(im, im_out, srcj, srci, cv::INTER_LINEAR);
}

void spherical_dibr::image_depth_inverse_mapping(Mat& im, Mat& depth_out_double, Mat& rot_mat_inv, Vec3d t_vec_inv, Mat& im_out, double focal, double ox, double oy)
{
    int im_width = im.cols;
    int im_height = im.rows;
	
    Mat srci(im_height, im_width, CV_32F);
	Mat srcj(im_height, im_width, CV_32F);
    float* srci_data = (float*)srci.data;
    float* srcj_data = (float*)srcj.data;

    im_out.create(im.rows, im.cols, im.type());

    Vec3w* im_data = (Vec3w*)im.data;
    Vec3w* im_out_data = (Vec3w*)im_out.data;
    double* depth_out_double_data = (double*)depth_out_double.data;
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < im_height; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            // inverse warping
            Vec3d vec_pixel = rt_pixel(Vec3d(i, j, depth_out_double_data[i*im.cols + j])
                                       , t_vec_inv
                                       , rot_mat_inv
                                       , focal, ox, oy);
            int origin_i = vec_pixel[0];
            int origin_j = vec_pixel[1];
            origin_i = clip(origin_i, 0, im_height);
            origin_j = clip(origin_j, 0, im_width);
            srci_data[i*im.cols + j] = origin_i;
            srcj_data[i*im.cols + j] = origin_j;
            double dist_depth = vec_pixel[2];
        }
    }
    remap(im, im_out, srcj, srci, cv::INTER_LINEAR);
}

Mat spherical_dibr::invert_depth(Mat& depth_double, double min_dist, double max_dist)
{
    Mat depth_inverted(depth_double.rows, depth_double.cols, depth_double.type());
    double* depth_double_data = (double*)depth_double.data;
    double* depth_inverted_data = (double*)depth_inverted.data;
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < depth_double.rows; i++)
    {
        for(int j = 0; j < depth_double.cols; j++)
        {
            if(depth_double_data[i*depth_double.cols + j] > 1e-6)
                depth_inverted_data[i*depth_double.cols + j] = max_dist - depth_double_data[i*depth_double.cols + j];
            else
                depth_inverted_data[i*depth_double.cols + j] = 0;
        }
    }

    return depth_inverted;
}

Mat spherical_dibr::revert_depth(Mat& depth_inverted, double min_dist, double max_dist)
{
    Mat depth_reverted(depth_inverted.rows, depth_inverted.cols, depth_inverted.type());
    double* depth_inverted_data = (double*)depth_inverted.data;
    double* depth_reverted_data = (double*)depth_reverted.data;
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < depth_inverted.rows; i++)
    {
        for(int j = 0; j < depth_inverted.cols; j++)
        {
            if(depth_inverted_data[i*depth_inverted.cols + j] > 1e-6)
                depth_reverted_data[i*depth_inverted.cols + j] = max_dist - depth_inverted_data[i*depth_inverted.cols + j];
            else
                depth_reverted_data[i*depth_inverted.cols + j] = 0;
        }
    }
    return depth_reverted;
}

void spherical_dibr::render(cv::Mat& im, cv::Mat& depth_double, double min_dist, double max_dist
            , cv::Mat& rot_mat, cv::Vec3d t_vec, double focal, double ox, double oy)
{
    // Do forward mapping
    image_depth_forward_mapping(im, depth_double, rot_mat, t_vec, im_out_forward, depth_out_forward, focal, ox, oy);
    Mat depth_out_forward_inverted = invert_depth(depth_out_forward, min_dist, max_dist);

    // Filtering depth with median/morphological closing
    int element_size = 7;
    Mat depth_out_median_inverted = median_depth(depth_out_forward_inverted, element_size);
    Mat depth_out_closing_inverted = closing_depth(depth_out_forward_inverted, element_size);
    depth_out_median = revert_depth(depth_out_median_inverted, min_dist, max_dist);
    depth_out_closing = revert_depth(depth_out_closing_inverted, min_dist, max_dist);

    // Do inverse mapping
    Mat rot_mat_inv = rot_mat.t();
    Vec3d t_vec_inv = -t_vec;

    image_depth_inverse_mapping(im, depth_out_median, rot_mat_inv, t_vec_inv, im_out_inverse_median, focal, ox, oy);
    image_depth_inverse_mapping(im, depth_out_closing, rot_mat_inv, t_vec_inv, im_out_inverse_closing, focal, ox, oy);
}
