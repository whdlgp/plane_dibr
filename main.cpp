#include "plane_dibr.hpp"
#include "debug_print.h"
#include "6dof_reader.hpp"

#include <iostream>
#include <sstream>
#include <vector>
#include "INIReader.h"

using namespace std;
using namespace cv;

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

vector<string> string_parse(string str, string tok)
{
    vector<string> token;
    size_t pos = 0;
    while ((pos = str.find(tok)) != std::string::npos) 
    {
        token.push_back(str.substr(0, pos));
        str.erase(0, pos + tok.length());
    }
    token.push_back(str);

    return token;
}

Vec3d string_to_vec(string str)
{
    Vec3d vec;
    vector<string> vec_str = string_parse(str, " ");
    vec[0] = stod(vec_str[0]);
    vec[1] = stod(vec_str[1]);
    vec[2] = stod(vec_str[2]);

    return vec;
}

int main()
{
    INIReader reader("camera_info.ini");

    if (reader.ParseError() < 0)
    {
        std::cout << "Can't load 'camera_info.ini'\n";
        return 1;
    }

    int cam_num = reader.GetInteger("option", "number", -1);
    string input_dir = reader.Get("option", "input", "UNKNOWN") + "/";
    string output_dir = reader.Get("option", "output", "UNKNOWN") + "/";

    std::cout << "Config loaded from 'camera_info.ini'\n"
              << "number of camera : " <<  cam_num << "\n" << endl;

    // Read Camera information
    vector<camera_info> cam_info(cam_num);
    for(int i = 0; i < cam_num; i++)
    {
        string cams = "camera";
        cams = cams + to_string(1+i);
        cout << "Read camera: " << cams << endl;

        cam_info[i].cam_name = input_dir + reader.Get(cams, "imagename", "UNKNOWN");
        cam_info[i].depth_name = input_dir + reader.Get(cams, "depthname", "UNKNOWN");
        cam_info[i].projection = reader.GetInteger(cams, "projection", -1);
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
    camera_info vt_cam_info;
    vt_cam_info.cam_name = output_dir + reader.Get("virtualview", "imagename", "UNKNOWN");
    vt_cam_info.depth_name = output_dir + reader.Get("virtualview", "depthname", "UNKNOWN");
    vt_cam_info.projection = reader.GetInteger("virtualview", "projection", -1);
    vt_cam_info.fx = reader.GetReal("virtualview", "fx", -1);
    vt_cam_info.fy = reader.GetReal("virtualview", "fy", -1);
    vt_cam_info.ox = reader.GetReal("virtualview", "ox", -1);
    vt_cam_info.oy = reader.GetReal("virtualview", "oy", -1);
    vt_cam_info.rot = string_to_vec(reader.Get("virtualview", "rotation", "UNKNOWN"));
    vt_cam_info.tran = string_to_vec(reader.Get("virtualview", "translation", "UNKNOWN"));
    vt_cam_info.depth_min = reader.GetReal("virtualview", "depthmin", -1);
    vt_cam_info.depth_max = reader.GetReal("virtualview", "depthmax", -1);
    vt_cam_info.width = reader.GetInteger("virtualview", "width", -1);
    vt_cam_info.height = reader.GetInteger("virtualview", "height", -1);
    vt_cam_info.bit_depth_image = reader.GetInteger("virtualview", "bitdepthimage", -1);
    vt_cam_info.bit_depth_depth = reader.GetInteger("virtualview", "bitdepthdepth", -1);

    six_dof_reader image_reader;
    vector<Mat> im(cam_num);
    vector<Mat> depth_double(cam_num);
    for(int i = 0; i < cam_num; i++)
    {
        im[i] = image_reader.read_yuv(cam_info[i].cam_name, cam_info[i].width, cam_info[i].height, cam_info[i].bit_depth_image);
        depth_double[i] = image_reader.depth_double_yuv(cam_info[i].depth_name, cam_info[i].width, cam_info[i].height
                                                        , cam_info[i].bit_depth_depth
                                                        , cam_info[i].depth_min
                                                        , cam_info[i].depth_max);
    }

    spherical_dibr sp_dibr;
    
    vector<Mat> img_forward(cam_num);
    vector<Mat> depth_forward(cam_num);
    vector<Mat> depth_map_result(cam_num);
    vector<Mat> img_result(cam_num);
    vector<double> cam_dist(cam_num);

    // Start rendering
    for(int i = 0; i < cam_num; i++)
    {
        START_TIME(render_one_image);
        spherical_dibr spd;
        cout << "image " << i << " now do rendering" << endl;

        // Calculate R|t to render virtual view point
        Mat cam_rot_mat = sp_dibr.eular2rot(Vec3f(RAD(cam_info[i].rot[0]), RAD(cam_info[i].rot[1]), RAD(cam_info[i].rot[2])));
        Mat rot_mat_inv = cam_rot_mat.t();
        Mat vt_rot_mat = sp_dibr.eular2rot(Vec3f(RAD(vt_cam_info.rot[0]), RAD(vt_cam_info.rot[1]), RAD(vt_cam_info.rot[2])));
        Mat r = vt_rot_mat*rot_mat_inv;
        Vec3d t_tmp = vt_cam_info.tran-cam_info[i].tran;
        double* rot_mat_data = (double*)cam_rot_mat.data;
        Vec3d t;
        t[0] = rot_mat_data[0]*t_tmp[0] + rot_mat_data[1]*t_tmp[1] + rot_mat_data[2]*t_tmp[2];
        t[1] = rot_mat_data[3]*t_tmp[0] + rot_mat_data[4]*t_tmp[1] + rot_mat_data[5]*t_tmp[2];
        t[2] = rot_mat_data[6]*t_tmp[0] + rot_mat_data[7]*t_tmp[1] + rot_mat_data[8]*t_tmp[2];

        // Render virtual view point
        spd.render(im[i], depth_double[i]
                   , r, t
                   , cam_info[i], vt_cam_info);
        STOP_TIME(render_one_image);

        // Put result of each rendering results to vector buffer

        img_forward[i] = spd.im_out_forward;
        depth_forward[i] = spd.depth_out_forward;
        depth_map_result[i] = spd.depth_out_median;
        img_result[i] = spd.im_out_inverse_median;
        cam_dist[i] = sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
    }

    // start blending
    int width = vt_cam_info.width;
    int height = vt_cam_info.height;
    Mat blended_img(height, width, CV_16UC3);
    vector<Vec3w*> im_data(cam_num);
    vector<double*> depth_data(cam_num);
    Vec3w* blended_data = (Vec3w*)blended_img.data;
    for(int i = 0; i < cam_num; i++)
    {
        im_data[i] = (Vec3w*)img_result[i].data;
        depth_data[i] = (double*)depth_map_result[i].data;
    }

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

    return 0;
}
