# Main
* 카메라 정보를 읽어와서 camera_info 구조체로 저장. 
```cpp
    // Read Camera information
    vector<camera_info> cam_info(cam_num);
    for(int i = 0; i < cam_num; i++)
    {
        string cams = "camera";
        cams = cams + to_string(1+i);
        cout << "Read camera: " << cams << endl;

        cam_info[i].cam_name = input_dir + reader.Get(cams, "imagename", "UNKNOWN");
        cam_info[i].depth_name = input_dir + reader.Get(cams, "depthname", "UNKNOWN");
        cam_info[i].fx = reader.GetReal(cams, "fx", -1);
        cam_info[i].fy = reader.GetReal(cams, "fy", -1);
        cam_info[i].ox = reader.GetReal(cams, "ox", -1);
        cam_info[i].oy = reader.GetReal(cams, "oy", -1);
        cam_info[i].rot = string_to_vec(reader.Get(cams, "rotation", "UNKNOWN"));
        cam_info[i].tran = string_to_vec(reader.Get(cams, "translation", "UNKNOWN"));
        cam_info[i].depth_min = reader.GetReal(cams, "depthmin", -1);
        cam_info[i].depth_max = reader.GetReal(cams, "depthmax", -1);
        cam_info[i].width = reader.GetInteger(cams, "width", -1);
        cam_info[i].height = reader.GetInteger(cams, "height", -1);
        cam_info[i].bit_depth_image = reader.GetInteger(cams, "bitdepthimage", -1);
        cam_info[i].bit_depth_depth = reader.GetInteger(cams, "bitdepthdepth", -1);
    }
    
    // Read Virtual View Point information
    Vec3d vt_rot, vt_tran;
    double vt_depth_min, vt_depth_max;
    vt_depth_min = reader.GetReal("virtualview", "depthmin", -1);
    vt_depth_max = reader.GetReal("virtualview", "depthmax", -1);
    vt_rot = string_to_vec(reader.Get("virtualview", "rotation", "UNKNOWN"));
    vt_tran = string_to_vec(reader.Get("virtualview", "translation", "UNKNOWN"));
```
* 저장된 카메라 정보를 이용해 각각의 카메라에서의 가상 뷰를 렌더링
```cpp
    // Start rendering
    for(int i = 0; i < cam_num; i++)
    {
        START_TIME(render_one_image);
        spherical_dibr spd;
        cout << "image " << i << " now do rendering" << endl;

        // Calculate R|t to render virtual view point
        Mat r = vt_rot_mat*rot_mat_inv[i];
        Vec3d t_tmp = vt_tran-cam_info[i].tran;
        double* rot_mat_data = (double*)rot_mat[i].data;
        Vec3d t;
        t[0] = rot_mat_data[0]*t_tmp[0] + rot_mat_data[1]*t_tmp[1] + rot_mat_data[2]*t_tmp[2];
        t[1] = rot_mat_data[3]*t_tmp[0] + rot_mat_data[4]*t_tmp[1] + rot_mat_data[5]*t_tmp[2];
        t[2] = rot_mat_data[6]*t_tmp[0] + rot_mat_data[7]*t_tmp[1] + rot_mat_data[8]*t_tmp[2];

        // Render virtual view point
        spd.render(im[i], depth_double[i], cam_info[i].depth_min, cam_info[i].depth_max, r, t, cam_info[i].fx, cam_info[i].ox, cam_info[i].oy);
        STOP_TIME(render_one_image);

        // Put result of each rendering results to vector buffer

        img_forward[i] = spd.im_out_forward;
        depth_forward[i] = spd.depth_out_forward;
        depth_map_result[i] = spd.depth_out_median;
        img_result[i] = spd.im_out_inverse_median;
        cam_dist[i] = sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
    }
```
* 렌더링된 뷰 들을 블랜딩
```cpp
    START_TIME(Blend_image);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            Vec3d pixel_val = 0;
            double dist_sum = 0;
            double threshold = 0.01; // consider below than threshold are occluded area
            int valid_count = 0;
            for(int c = 0; c < cam_num; c++)
            {
                if(depth_data[c][i*width + j] > threshold)
                {
                    valid_count++;
                    dist_sum += 1/cam_dist[c];
                }
            }
            for(int c = 0; c < cam_num; c++)
            {
                if(depth_data[c][i*width + j] > threshold)
                {
                    if(valid_count > 1)
                    {
                        pixel_val[0] += (1/cam_dist[c]/dist_sum)*im_data[c][i*width + j][0];
                        pixel_val[1] += (1/cam_dist[c]/dist_sum)*im_data[c][i*width + j][1];
                        pixel_val[2] += (1/cam_dist[c]/dist_sum)*im_data[c][i*width + j][2];
                    }
                    else if(valid_count == 1)
                    {
                        pixel_val[0] += im_data[c][i*width + j][0];
                        pixel_val[1] += im_data[c][i*width + j][1];
                        pixel_val[2] += im_data[c][i*width + j][2];
                    }
                }
            }
            blended_data[i*width + j][0] = pixel_val[0];
            blended_data[i*width + j][1] = pixel_val[1];
            blended_data[i*width + j][2] = pixel_val[2];
        }
    }
    STOP_TIME(Blend_image);
```

* 저장하고자 하는 영상들을 저장
```cpp
    // Save images
    cout << "Save images" << endl;
    vector<int> param;
    param.push_back(IMWRITE_PNG_COMPRESSION);
    param.push_back(0);
    for(int i = 0; i < cam_num; i++)
    {
        string forward_image_name = output_dir + "test_result";
        forward_image_name = forward_image_name + to_string(i);
        forward_image_name = forward_image_name + "_forward.png";
        cv::imwrite(forward_image_name, img_forward[i], param);

        string image_name = output_dir + "test_result";
        image_name = image_name + to_string(i);
        image_name = image_name + "_inverse.png";
        cv::imwrite(image_name, img_result[i], param);

        double min_pixel = 0, max_pixel = 65535;
        double min_dist = cam_info[i].depth_min, max_dist = cam_info[i].depth_max;
        int max_pixel_val = 255;
        string depth_forward_name = output_dir + "test_depth";
        depth_forward_name = depth_forward_name + to_string(i);
        depth_forward_name = depth_forward_name + "_forward.png";
        cv::imwrite(depth_forward_name, depth_forward[i]/max_dist*max_pixel_val, param);
        
        string depth_inverse_name = output_dir + "test_depth";
        depth_inverse_name = depth_inverse_name + to_string(i);
        depth_inverse_name = depth_inverse_name + ".png";
        cv::imwrite(depth_inverse_name, depth_map_result[i]/max_dist*max_pixel_val, param);

    }
    string blended_name = output_dir + "blend.png";
    cv::imwrite(blended_name, blended_img, param);
```
# 6DoF Reader
```cpp
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
```
* __read_yuv__  
YUV 파일을 읽어 Mat의 형태로 불러옴.
* __depth_double_yuv__
YUV 파일을 읽어 Mat의 형태로 불러옴. Depth정보를 실제 거리정보로 변환하여 불러옴
* __get_yuv_chan_10bit__, __get_yuv_chan_8bit__  
영상의 비트수를 고려하여 읽기 위함, 채널 별로 각각 불러옴
* __merge_yuv_chan__  
채널 별로 각각 불러온 이미지를 하나의 이미지로 합침

# Camera parameter .ini file
```ini
[option]
; number of camera
number = 2
; directory of inputs
input = input
; directory to save
output = save
```
* __number__  
랜더링에 필요한 원본 카메라 위치 수
* __input__, __output__  
입력 영상들의 위치 및 저장될 위치

```ini
; Config file for DIBR for Plane image 
[camera1]
; name of image file
imagename = image/PoznanFencing_1920x1080_cam0_10bps.yuv.1frame.yuv
; name of depth file
depthname = depth_10/Poznan_Fencing2_cam0_depth_1920x1080_10bps_cf420.yuv.1frame.yuv
; intrinsic parameters
fx = 1714.563022
fy = 1711.587928
ox = 925.077402
oy = 530.254549
; extrinsic parameters
rotation = 53.15421450159158 28.6913597092801 -15.301356405461455
translation = 7.354157256090722 6.503232071609891 -0.2810615662648424
; min/max value of depth in meter
depthmin = 3.5
depthmax = 7.0
; image format information
width = 1920
height = 1080
; bit depth of image and depthmap
bitdepthimage = 10
bitdepthdepth = 10
```

* __imagename__, __depthname__  
RGB 영상 및 깊이 영상의 파일명
* __fx__, __fy__, __ox__, __ox__  
카메라의 intrinsic 정보
* __rotation__, __translatio__  
카메라의 extrinsic 정보
* __depthmin__, __depthmax__  
깊이 정보의 최소/최대값
* __width__, __height__  
카메라의 해상도
* __bitdepthimage__, __bitdepthdepth__  
RGB 영상 및 깊이 영상의 bit depth

# Plane DIBR class

## camera_info class
```cpp
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
```
* 카메라 정보를 가지고 있는 클래스, 아직 완전히 적용해보지는 않았음 
* 내용은 .ini 파일 내용하고 거의 비슷함

## plane dibr class
```cpp
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
```
* class 명이 spherical 인건 spherical_dibr class 에서 구조를 가지고와서 수정해 사용해서 그런것
### rotation vector 와 rotation matrix간 변환
```cpp
    cv::Vec3d rot2eular(cv::Mat rot_mat);
    cv::Mat eular2rot(cv::Vec3d theta);
```
* xyz-eular 회전을 기본으로 함

### pixel 과 3차원 xyz 좌표계 변환
```cpp
    cv::Vec3d plane2cart(const cv::Vec3d& plane_pixel, double focal, double ox, double oy);
    cv::Vec3d cart2plane(const cv::Vec3d& cart, double focal, double ox, double oy);
```
* 픽셀 행/열값과 픽셀별 거리값을 Vec3d 형태로 전달, 3차원 좌표계로 픽셀의 실제 위치를 계산
* 그 역의 방향

### rotation 및 translation 적용 
```cpp
    cv::Vec3d applyTR(const cv::Vec3d& vec_cartesian, const cv::Mat& rot_mat, const cv::Vec3d t_vec);
    cv::Vec3d applyRT(const cv::Vec3d& vec_cartesian, const cv::Mat& rot_mat, const cv::Vec3d t_vec);
```
* forward mapping 시에는 Translation 적용 후 Rotation 적용
* inverse mapping 시에는 Rotation 적용 후 Translation 적용

### 픽셀을 3차원변환/R|t 적용, 다시 픽셀로 변환
```cpp
    cv::Vec3d tr_pixel(const cv::Vec3d& in_vec, const cv::Vec3d& t_vec, const cv::Mat& rot_mat, double focal, double ox, double oy);
    cv::Vec3d rt_pixel(const cv::Vec3d& in_vec, const cv::Vec3d& t_vec, const cv::Mat& rot_mat, double focal, double ox, double oy);
```
* 앞서 말한 함수들을 이용해 픽셀 입력을 주면 Rotation 및 translation이 적용된 픽셀 위치를 계산해주는 함수

### depth map filtering 함수
```cpp
    cv::Mat median_depth(cv::Mat& depth_double, int size);
    cv::Mat closing_depth(cv::Mat& depth_double, int size);
```
* filtering 방식은 현재 median 필터 및 closing 필터를 각각 따로따로 사용하고있음
* 원하는 필터를 선택해서 사용하려 하는데, 현재는 태스트 중이라 중간 결과를 둘다 저장하거나 원하는것만 저장하면서 테스트중

### forward mapping 및 inverse mapping 
```cpp
    void image_depth_forward_mapping(cv::Mat& im, cv::Mat& depth_double, cv::Mat& rot_mat, cv::Vec3d t_vec, cv::Mat& im_out, cv::Mat& depth_out_double, double focal, double ox, double oy);
    void image_depth_inverse_mapping(cv::Mat& im, cv::Mat& depth_out_double, cv::Mat& rot_mat_inv, cv::Vec3d t_vec_inv, cv::Mat& im_out, double focal, double ox, double oy);
```
* RGB 영상 및 depth 영상을 forward/inverse 방향으로 mapping 하는 함수
* 순서 상으로는, forward mapping -> filtering -> inverse mapping

### invert/revert depth
```cpp
    cv::Mat invert_depth(cv::Mat& depth_double, double min_dist, double max_dist);
    cv::Mat revert_depth(cv::Mat& depth_inverted, double min_dist, double max_dist);
```
* 단순히 depth map 필터링 중에 거리 정보를 가지고 있는 Mat의 최대/최소를 뒤바꾸는 함수
* invert: 픽샐값이 가까운쪽 < 먼쪽 이었던것을 가까운쪽 > 먼쪽이 되게끔 바꿔줌
* revert: 픽샐값이 가까운쪽 > 먼쪽 이었던것을 가까운쪽 < 먼쪽이 되게끔 바꿔줌
* 사실 둘이 하는건 똑같은대 그냥 구분해서 쓰고싶어서 두개로 나눔
* closing 필터 사용시에 '가까운쪽 > 먼쪽' 형태를 사용하는게 물체가 사라지거나 하는 현상이 줄어들어 추가한 것.

### rendering
```cpp
    void render(cv::Mat& im, cv::Mat& depth_double, double min_dist, double max_dist
            , cv::Mat& rot_mat, cv::Vec3d t_vec, double focal, double ox, double oy);
```
* 앞서 언급한 forward mapping -> filtering -> inverse mapping 순으로 관련 함수들을 호출하는 함수
* 저장은 class의 맴버 변수로 선언된 Mat 들에 저장
