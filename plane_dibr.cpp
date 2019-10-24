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

// pixel (u,v) to (X, Y, Z)
Vec3d spherical_dibr::plane2cart(const Vec3d& plane_pixel, double fx, double fy, double ox, double oy)
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
    tmp[0] = (plane_pixel[1] - ox)*plane_pixel[2]/fx;
    tmp[1] = (plane_pixel[0] - oy)*plane_pixel[2]/fy;
    tmp[2] = plane_pixel[2];

    Vec3d vec;
    vec[0] = -tmp[2];
    vec[1] = -tmp[0];
    vec[2] = -tmp[1];

    return vec;
}

// (X,Y,Z) to pixel domain
Vec3d spherical_dibr::cart2plane(const Vec3d& cart, double fx, double fy, double ox, double oy)
{
    Vec3d tmp;
    tmp[0] = -cart[1];
    tmp[1] = -cart[2];
    tmp[2] = -cart[0];

    Vec3d pixel;
    pixel[0] = tmp[1]*fy/tmp[2] + oy;
    pixel[1] = tmp[0]*fx/tmp[2] + ox;
    pixel[2] = tmp[2];

    return pixel;
}


// Translation and Rotation 
Vec3d spherical_dibr::applyTR(const Vec3d& vec_cartesian, const Mat& rot_mat, const Vec3d t_vec)
{
#ifdef USE_PTR
    double* rot_mat_data = (double*)rot_mat.data;
#endif

    Vec3d tran;
    tran[0] = vec_cartesian[0] - t_vec[0];
    tran[1] = vec_cartesian[1] - t_vec[1];
    tran[2] = vec_cartesian[2] - t_vec[2];

    Vec3d rot;
#ifdef USE_PTR
    rot[0] = rot_mat_data[0]*tran[0] + rot_mat_data[1]*tran[1] + rot_mat_data[2]*tran[2];
    rot[1] = rot_mat_data[3]*tran[0] + rot_mat_data[4]*tran[1] + rot_mat_data[5]*tran[2];
    rot[2] = rot_mat_data[6]*tran[0] + rot_mat_data[7]*tran[1] + rot_mat_data[8]*tran[2];
#else
    rot[0] = rot_mat.at<double>(0,0)*tran[0] + rot_mat.at<double>(0,1)*tran[1] + rot_mat.at<double>(0,2)*tran[2];
    rot[1] = rot_mat.at<double>(1,0)*tran[0] + rot_mat.at<double>(1,1)*tran[1] + rot_mat.at<double>(1,2)*tran[2];
    rot[2] = rot_mat.at<double>(2,0)*tran[0] + rot_mat.at<double>(2,1)*tran[1] + rot_mat.at<double>(2,2)*tran[2];
#endif
    return rot;
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

// TODO: I think many OpenCV function use 32F ie. float as default, then  we donot need to use double
//       because that high accuracy is not needed for DIBR processing. Right?
Mat spherical_dibr::median_depth(Mat& depth_double, int size)
{
    Mat depth_float, depth_double_median, depth_float_median;
    depth_double.convertTo(depth_float, CV_32FC1);
    medianBlur(depth_float, depth_float_median, size);
#ifdef HOLE_FILLING
    medianBlur(mask, mask, size);  // To refelct the filtering to the mask 
#endif
    depth_float_median.convertTo(depth_double_median, CV_64FC1);
    return depth_double_median;
}

// TODO: I think many OpenCV function use 32F ie. float as default, then  we donot need to use double
//       because that high accuracy is not needed for DIBR processing. Right?
Mat spherical_dibr::closing_depth(Mat& depth_double, int size)
{
    Mat depth_float, depth_double_median, depth_float_median;
    depth_double.convertTo(depth_float, CV_32FC1);
    Mat element(size, size, CV_32FC1, Scalar(1.0));
    morphologyEx(depth_float, depth_float_median, CV_MOP_CLOSE, element);
#ifdef HOLE_FILLING
    morphologyEx(mask, mask, CV_MOP_CLOSE, element); // To refelct the filtering to the mask 
#endif
    depth_float_median.convertTo(depth_double_median, CV_64FC1);
    return depth_double_median;
}

// forwarding warping of depthmap 
void spherical_dibr::image_depth_forward_mapping(Mat& im, Mat& depth_double
                                                , Mat& rot_mat, Vec3d t_vec
                                                , Mat& im_out, Mat& depth_out_double
                                                , camera_info& cam_info, camera_info& vt_cam_info)
{
    int im_width = cam_info.width;
    int im_height = cam_info.height;

    int im_out_width = vt_cam_info.width;
    int im_out_height = vt_cam_info.height;
    // BUG FIX !!!
    im_out_width = im_width;
    im_out_height = im_height;
    //  END of BUG FIX

    cout << "in-size:" << im_width << "x" << im_height << endl;
    cout << "out-size:" << im_out_width << "x" << im_out_height << endl;

// This will make seg-fault, bus fault and so on, because wrong re-map .....??
//#define RGB_FWD_WARPING
#ifdef RGB_FWD_WARPING
    Mat srci(im_height, im_width, CV_32F);
    Mat srcj(im_height, im_width, CV_32F);
#endif

    im_out = Mat::zeros(im_out_height, im_out_width, im.type());
    depth_out_double = Mat::zeros(im_out_height, im_out_width, depth_double.type());

#ifdef HOLE_FILLING
    //cout << "mask allocating" << endl;
    mask = Mat::zeros(im_out_height, im_out_width, CV_8UC1);  // 1 : has value, 0: not
    //cout << mask.size()  << ":" << mask.type()  << endl;
#endif

#ifdef USE_PTR
    float* srci_data = (float*)srci.data;
    float* srcj_data = (float*)srcj.data;
    Vec3w* im_data = (Vec3w*)im.data;
    Vec3w* im_out_data = (Vec3w*)im_out.data;
    double* depth_data = (double*)depth_double.data;
    double* depth_out_double_data = (double*)depth_out_double.data;
#endif

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < im_height; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            // forward warping
#ifdef USE_PTR
            Vec3d in_vec(i, j, depth_data[i*im_width + j]);
#else
            Vec3d in_vec(i, j, depth_double.at<double>(i,j));
#endif
            Vec3d vec_cartesian = plane2cart(in_vec, cam_info.fx, cam_info.fy, cam_info.ox, cam_info.oy);
            Vec3d vec_cartesian_rot = applyTR(vec_cartesian, rot_mat, t_vec);
            Vec3d vec_pixel = cart2plane(vec_cartesian_rot, vt_cam_info.fx, vt_cam_info.fy, vt_cam_info.ox, vt_cam_info.oy);

            int dist_i = vec_pixel[0];
            int dist_j = vec_pixel[1];
            dist_i = clip(dist_i, 0, im_out_height);
            dist_j = clip(dist_j, 0, im_out_width);

#ifdef RGB_FWD_WARPING
#ifdef USE_PTR		    
            srci_data[dist_i*im_width + dist_j] = (float)i;
            srcj_data[dist_i*im_width + dist_j] = (float)j;
#else
            srci.at<float>(dist_i, dist_j) = (float)i;
            srcj.at<float>(dist_i, dist_j) = (float)j;
#endif
#endif

            double dist_depth = vec_pixel[2];
            if((dist_i >= 0) && (dist_j >= 0) && (dist_i < im_height) && (dist_j < im_width))
            {
#ifdef USE_PTR		    
                if(depth_out_double_data[dist_i*im_out_width + dist_j] == 0){  // TODO: how about to use mask value
                    depth_out_double_data[dist_i*im_out_width + dist_j] = dist_depth;
#ifdef HOLE_FILLING
		    mask.at<unsigned char>(dist_i, dist_j) = 255;   // now has value 
#endif
		} else if(depth_out_double_data[dist_i*im_out_width + dist_j] > dist_depth)
                    depth_out_double_data[dist_i*im_out_width + dist_j] = dist_depth;
#else
                if(depth_out_double.at<double>(dist_i,dist_j) < 1.0e-30){  
                //if(mask.at<unsigned char>(dist_i,dist_j) == 0){ // no value yet  
                    depth_out_double.at<double>(dist_i,dist_j) = dist_depth;
#ifdef HOLE_FILLING
		    mask.at<unsigned char>(dist_i, dist_j) = 255;   // now has value 
#endif
		} else if(depth_out_double.at<double>(dist_i,dist_j) > dist_depth)  // more closer one
                    depth_out_double.at<double>(dist_i, dist_j) = dist_depth;
#endif
            }
        }
    }

#ifdef RGB_FWD_WARPING
    remap(im, im_out, srcj, srci, cv::INTER_LINEAR);
#endif
}

void spherical_dibr::image_depth_inverse_mapping(Mat& im, Mat& depth_out_double
                                                , Mat& rot_mat_inv, Vec3d t_vec_inv
                                                , Mat& im_out
                                                , camera_info& cam_info, camera_info& vt_cam_info)
{
    int im_width = cam_info.width;
    int im_height = cam_info.height;

    int im_out_width = vt_cam_info.width;
    int im_out_height = vt_cam_info.height;
	
    Mat srci(im_height, im_width, CV_32F);
    Mat srcj(im_height, im_width, CV_32F);
#ifdef USE_PTR // TODO: Somehow  this pointer access make some problems 
    float* srci_data = (float*)srci.data;
    float* srcj_data = (float*)srcj.data;

    im_out = Mat::zeros(im_out_height, im_out_width, im.type());

    Vec3w* im_data = (Vec3w*)im.data;
   // Vec3w* im_out_data = (Vec3w*)im_out.data;
    double* depth_out_double_data = (double*)depth_out_double.data;
#endif 

    im_out = Mat::zeros(im_out_height, im_out_width, im.type());
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < im_out_height; i++)
    {
        for(int j = 0; j < im_out_width; j++)
        {
            // inverse warping
#if USE_PTR
            Vec3d in_vec(i, j, depth_out_double_data[i*im_out_width + j]);
#else
            Vec3d in_vec(i, j, depth_out_double.at<double>(i,j));
#endif
            Vec3d vec_cartesian = plane2cart(in_vec, vt_cam_info.fx, vt_cam_info.fy, vt_cam_info.ox, vt_cam_info.oy);
            Vec3d vec_cartesian_rot = applyRT(vec_cartesian, rot_mat_inv, t_vec_inv);
            Vec3d vec_pixel = cart2plane(vec_cartesian_rot, cam_info.fx, cam_info.fy, cam_info.ox, cam_info.oy);

            float origin_i = vec_pixel[0];
            float origin_j = vec_pixel[1];
            origin_i = clip(origin_i, 0.f, im_height*1.f);
            origin_j = clip(origin_j, 0.f, im_width*1.f);
#ifdef USE_PTR // TODO: Somehow  this pointer access make some problems 
            srci_data[i*im_out_width + j] = origin_i;
            srcj_data[i*im_out_width + j] = origin_j;
#else
            srci.at<float>(i,j) = origin_i;
            srcj.at<float>(i,j) = origin_j;
#endif 
        }
    }
    remap(im, im_out, srcj, srci, cv::INTER_LINEAR);
}

/* far(large)-near(small)  to far (small)-near(large) 
 * so that in filtering process, values are upgraded to FG values 
 * */
Mat spherical_dibr::invert_depth(Mat& depth_double, double min_dist, double max_dist)
{
    Mat depth_inverted =  Mat::zeros(depth_double.rows, depth_double.cols, depth_double.type());

#ifdef USE_PTR    
    double* depth_double_data = (double*)depth_double.data;
    double* depth_inverted_data = (double*)depth_inverted.data;
#endif
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < depth_double.rows; i++)
    {
        for(int j = 0; j < depth_double.cols; j++)
        {
#ifdef USE_PTR		
            if(depth_double_data[i*depth_double.cols + j] > 1e-6)
                depth_inverted_data[i*depth_double.cols + j] = max_dist - depth_double_data[i*depth_double.cols + j];
            else
                depth_inverted_data[i*depth_double.cols + j] = 0;
#else
            if(depth_double.at<double>(i,j) > 1e-6)
                depth_inverted.at<double>(i,j) = max_dist - depth_double.at<double>(i,j);
#endif
        }
    }

    return depth_inverted;
}

Mat spherical_dibr::revert_depth(Mat& depth_inverted, double min_dist, double max_dist)
{
    Mat depth_reverted = Mat::zeros(depth_inverted.rows, depth_inverted.cols, depth_inverted.type());
#ifdef USE_PTR		
    double* depth_inverted_data = (double*)depth_inverted.data;
    double* depth_reverted_data = (double*)depth_reverted.data;
#endif
    //
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < depth_inverted.rows; i++)
    {
        for(int j = 0; j < depth_inverted.cols; j++)
        {
#ifdef USE_PTR		
            if(depth_inverted_data[i*depth_inverted.cols + j] > 1e-6)
                depth_reverted_data[i*depth_inverted.cols + j] = max_dist - depth_inverted_data[i*depth_inverted.cols + j];
            else
                depth_reverted_data[i*depth_inverted.cols + j] = 0;
#else
     //   BUG FIXED:  when max dist, here you will have 0 and then you donot invert the value,  so that become 0 
     //       if(depth_inverted.at<double>(i,j)> 1e-6)
#ifdef HOLE_FILLING
              if(mask.at<unsigned char>(i,j) > 0)
#endif
                depth_reverted.at<double>(i,j) = max_dist - depth_inverted.at<double>(i,j);
#endif
        }
    }
    return depth_reverted;
}

//#define CLOSING_FILTER
#define MEDIAN_FILTER
static int  num = 0;
void spherical_dibr::render(cv::Mat& im, cv::Mat& depth_double
                            , cv::Mat& rot_mat, cv::Vec3d t_vec
                            , camera_info& cam_info, camera_info& vt_cam_info)
{
    // 1. Do forward depthmap warping
    image_depth_forward_mapping(im, depth_double
                                , rot_mat, t_vec
                                , im_out_forward, depth_out_forward
                                , cam_info, vt_cam_info);
    Mat depth_out_forward_inverted = invert_depth(depth_out_forward, cam_info.depth_min, cam_info.depth_max);

    // 2. Filtering depth with median/morphological closing
    int element_size = 7;
#ifdef MEDIAN_FILTER  
    Mat depth_out_median_inverted = median_depth(depth_out_forward_inverted, element_size);
    depth_out_median = revert_depth(depth_out_median_inverted, cam_info.depth_min, cam_info.depth_max);
#endif 

#ifdef CLOSING_FILTER
    Mat depth_out_closing_inverted = closing_depth(depth_out_forward_inverted, element_size);
    depth_out_closing = revert_depth(depth_out_closing_inverted, cam_info.depth_min, cam_info.depth_max);
#endif

    // 3. Do inverse mapping
    Mat rot_mat_inv = rot_mat.t();
    Vec3d t_vec_inv = -t_vec;

#ifdef MEDIAN_FILTER  
    image_depth_inverse_mapping(im, depth_out_median
                                , rot_mat_inv, t_vec_inv
                                , im_out_inverse_median
                                , cam_info, vt_cam_info);

    //cout << "GAP:" << im.isContinuous() << " vs " << im_out_inverse_median.isContinuous() << endl;
#endif

#ifdef CLOSING_FILTER
    image_depth_inverse_mapping(im, depth_out_closing
                                , rot_mat_inv, t_vec_inv
                                , im_out_inverse_closing
                                , cam_info, vt_cam_info);

    /*
    cout << "Type:" << im.type() << " vs " << im_out_inverse_closing.type() << endl;
    //Mat m16 = im_out_inverse_closing/255;
    Mat m16 = im/255;
    Mat m8;
    m16.convertTo(m8,CV_8UC3);
    if (num == 0){
    	imwrite( "closing_rendered0.png", m8);
    }else{
    	imwrite( "closing_rendered1.png", m8);
    }
    */
#endif 
//	num = 1;
}
