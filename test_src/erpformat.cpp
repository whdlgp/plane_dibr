#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "debug_print.h"
#include "plane_dibr.hpp"
#include "6dof_reader.hpp"
#include "INIReader.h"

using namespace std;
using namespace cv;


 //
 //
 //    z
 //   /
 //  /
 // o----- x
 // |
 // |
 // y
 //

   
// pixel coordinate to cartesian coordinate
static Vec3d pixel2cart(const Vec3d& uv, double fx, double fy, double ox, double oy)
{

    Vec3d v;
    v[0] = (uv[0] - ox)/fx; 
    v[1] = (uv[1] - oy)/fy;
    v[2] = 1.0;

    return v;
}

static Vec3d cart2plane(const Vec3d& cart, double focal, double ox, double oy)
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


// this is polar coordinate (radius and angles)
// x,y,z defined in image plane way 
static Vec3d cart2polar(const Vec3d& XYZ)
{
    Vec3d polar;
    double X = XYZ[0], Y = XYZ[1], Z = XYZ[2];

    double r = sqrt(X*X + Y*Y + Z*Z);
    polar[1] = acos(-Y/r);  // lat vertical angle   [0, pi]   - for  makeing -90 to 0 and +90 to 180 
    polar[0] = atan2(X,Z); // lon horizontal angle  [-pi pi]  
    
    return polar;
}

static Vec3d rad2cart(const Vec3d& vec_rad)
{
    Vec3d vec_cartesian;
    vec_cartesian[0] = vec_rad[2]*sin(vec_rad[0])*cos(vec_rad[1]);
    vec_cartesian[1] = vec_rad[2]*sin(vec_rad[0])*sin(vec_rad[1]);
    vec_cartesian[2] = vec_rad[2]*cos(vec_rad[0]);
    return vec_cartesian;
}

// Polar coordinate to ERP pixel position 
static Vec2d polar2pixel(const Vec3d& polar, int width, int height)
{
    Vec2d pixel; 
    double  lon = polar[0] + M_PI;
    double  lat = polar[1]; // + M_PI/2.0;

    pixel[0] = width*lon/(2.0*M_PI);
    pixel[1] = height*lat/M_PI;

    if(pixel[0] < 0){
	    pixel[0] = 0;
    }else if (pixel[0] >= width){
	    pixel[0] = width -1;
    }
    if(pixel[1] < 0){
	    pixel[1] = 0;
    }else if (pixel[1] >= height){
	    pixel[1] = height-1;
    }

    return pixel;
}

static Vec3d pixel2rad(const Vec3d& in_vec, int width, int height)
{
    return Vec3d(M_PI*in_vec[0]/height, 2*M_PI*in_vec[1]/width, in_vec[2]);
}




static vector<string> string_parse(string str, string tok)
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

static Vec3d string_to_vec(string str)
{
    Vec3d vec;
    vector<string> vec_str = string_parse(str, " ");
    vec[0] = stod(vec_str[0]);
    vec[1] = stod(vec_str[1]);
    vec[2] = stod(vec_str[2]);

    return vec;
}


// 0. Read Camera information
void get1stCameraInfo(INIReader reader, camera_info &cam_info)
{
    int cam_num = reader.GetInteger("option", "number", -1);
    string input_dir = reader.Get("option", "input", "UNKNOWN") + "/";
    string output_dir = reader.Get("option", "output", "UNKNOWN") + "/";

    // 0. Read Camera information
    string cams = string("camera") + to_string(1);
    cout << "Read 1st camera " << endl;

    cam_info.cam_name = input_dir + reader.Get(cams, "imagename", "UNKNOWN");
    cam_info.depth_name = input_dir + reader.Get(cams, "depthname", "UNKNOWN");
    cam_info.fx = reader.GetReal(cams, "fx", -1);
    cam_info.fy = reader.GetReal(cams, "fy", -1);
    cam_info.ox = reader.GetReal(cams, "ox", -1);
    cam_info.oy = reader.GetReal(cams, "oy", -1);
    cam_info.rot = string_to_vec(reader.Get(cams, "rotation", "UNKNOWN"));
    cam_info.tran = string_to_vec(reader.Get(cams, "translation", "UNKNOWN"));
    cam_info.depth_min = reader.GetReal(cams, "depthmin", -1);
    cam_info.depth_max = reader.GetReal(cams, "depthmax", -1);
    cam_info.width = reader.GetInteger(cams, "width", -1);
    cam_info.height = reader.GetInteger(cams, "height", -1);
    cam_info.bit_depth_image = reader.GetInteger(cams, "bitdepthimage", -1);
    cam_info.bit_depth_depth = reader.GetInteger(cams, "bitdepthdepth", -1);

}

//
// planar image to ERP image using forward warping 
// 
void convertPlanar2ERP_fwd(Mat &rgb_image, Mat &erp, camera_info &cam_info)
{

    int width = rgb_image.cols, height = rgb_image.rows;
    int erp_width = erp.cols, erp_height = erp.rows;

    Vec3b* erp_data_8 = (Vec3b*)erp.data;
    Vec3b* rgb_data_8 = (Vec3b*)rgb_image.data;
    Vec3w* erp_data_16 = (Vec3w*)erp.data;
    Vec3w* rgb_data_16 = (Vec3w*)rgb_image.data;
    Mat test(height, width, CV_8UC3);
    Vec3b* test_data = (Vec3b*)test.data;

    // using forward mapping   
    // @TODO: inverse warping 
    // @TODO: vector processing



    int keyin;
    #pragma omp parallel for
    for(int v = 0; v <  height; v++){
        for(int u = 0; u < width; u++){
	
#ifdef DEBUG
	    if(!(u%100) && !(v%100))
	    	cout <<  '(' << u << ',' << v << "):";
#endif
	    //
	    // 2.1 (u,v) to (x,y,f) to (X,Y,Z)
	    //              i, j = (y, x) why changed?
            Vec3d cart = pixel2cart(Vec3d((double)u, (double)v,  (double) 7.0),  // note that we donot need  depth 
				    cam_info.fx, cam_info.fy, cam_info.ox, cam_info.oy);
#ifdef DEBUG
	    if(!(u%100) && !(v%100))
	    	cout << cart << ' ';
#endif
	    // 2.2 (X,Y,Z) to (lat, lon) 
            Vec3d polar = cart2polar(cart);
#ifdef DEBUG
	    if(!(u%100) && !(v%100))
	    	cout << polar*180.0/M_PI << ' ';
#endif
	    // 2.3 (lat, lon) to (u'.v') 
            Vec2d erp_pixel = polar2pixel(polar, erp_width, erp_height);
#ifdef DEBUG
	    if(!(u%100) && !(v%100))
	    	cout << erp_pixel << ' ';

	    if(!(u%100) && !(v%100))
	    	cout << endl;
#endif
	    // 2.4 get the pixel-value 
            int dest_u = erp_pixel[0];
            int dest_v = erp_pixel[1];
            switch(cam_info.bit_depth_image){
                case 8:
                erp_data_8[dest_v*erp_width + dest_u] = rgb_data_8[v*width + u];
                break;

                case 10:
                erp_data_16[dest_v*erp_width + dest_u] = rgb_data_16[v*width + u];
                break;
                
                case 16:
                erp_data_16[dest_v*erp_width + dest_u] = rgb_data_16[v*width + u];
                break;
            }
        }
	   // cin >> keyin;
    }

}


/* test function of Planar and ERP format transform */

int main(int argc, char* argv[])
{

    // 0. in and out camera parameters 
    if(argc  < 2){
	   cout << "usage:" << argv[0]   <<  " camera_info_file" << endl;
	   return 0;
    }

    bool print_camera = true;
    if(argc  < 2){
	   cout << "usage:" << argv[0]   <<  " camera_info_file" << endl;
	   return 0;
    }

    INIReader reader(argv[1]);
    if (reader.ParseError() < 0){
        std::cout << "Can't load " << argv[1] << endl;
        return 0;
    }

    camera_info cam_info;
    get1stCameraInfo(reader, cam_info);
    if (print_camera){ // print camera params
     	cout << "- yuvfile:" << cam_info.cam_name << endl;
      	cout << "- depth:" << cam_info.depth_name << endl;
      	cout << "- size:"  << cam_info.width  << "," << cam_info.height  << endl;
       	cout << "- bits:"   << cam_info.bit_depth_image <<  "," << cam_info.bit_depth_depth  << endl;
       	cout << "- fx,y:"<<  cam_info.fx << "," << cam_info.fy << endl;
       	cout << "- ox,y:"<<  cam_info.ox << ", " << cam_info.oy << endl;
       	/*cout << "- cam_info[i].rot = string_to_vec(reader.Get(cams, "rotation", "UNKNOWN"));
       	cout << cam_info[i].tran = string_to_vec(reader.Get(cams, "translation", "UNKNOWN"));
	*/
       	cout << "- depth range:" << cam_info.depth_min  << "," << cam_info.depth_max << endl;
   }

    // for now, same size output image 
    int erp_width = cam_info.width;
    int erp_height = cam_info.height;

    six_dof_reader image_reader;
    Mat rgb_image;
    Mat depth_double;


#define VIDEO_TEST
#ifndef VIDEO_TEST
    int fn_max = 1;
#else
    int fn_max = 300; // temp
    string vid_out_fname = string("erp_fencing_cam") + to_string(0)  + string(".mp4");
    int fps = 25;
    VideoWriter video(vid_out_fname,CV_FOURCC('H','2','6','4'), fps, Size(erp_width,erp_height));
#endif

    Mat erp;
    Mat erp8; // CV_8UC3 for video writer


    for(int fn = 0; fn  < fn_max; fn++){

	 cout << "processing " << fn + 1 << "-th frame" << endl;

   	 rgb_image = image_reader.read_yuv(cam_info.cam_name,
			cam_info.width, cam_info.height,
			cam_info.bit_depth_image);
   	 if (rgb_image.size() == Size()){
		cout << "cannot read the yuv file " << endl;
		return  0;
    	}

#if 0   // not require depth for erp 
    	depth_double = image_reader.depth_double_yuv(cam_info.depth_name,
		       	cam_info.width, cam_info.height,
			cam_info.bit_depth_depth,
			cam_info.depth_min,
			cam_info.depth_max);
    	if (depth_double.size()  == Size()){
		cout << "cannot read depth file "  << endl;
		return  0;
    	}
#endif

    	// 2. convert planar to ERP  
    	erp = Mat::zeros(erp_height, erp_width, rgb_image.type());
   	convertPlanar2ERP_fwd(rgb_image, erp, cam_info);

	
#ifndef VIDEO_TEST
    	// 3. save the ERP image 
    	string save_name = "heejune";
    	save_name = save_name + "_erp_test.png";
    	cv::imwrite(save_name, erp);
#else

	if (cam_info.bit_depth_image  != 8){ //  16 bits to  8 bits
        	erp = erp/256;
        	erp.convertTo(erp8,CV_8UC3);
		video.write(erp8);
	}else
		video.write(erp);
#endif
    }

    //  close video file 
#ifdef VIDEO_TEST
    video.release();
#endif

    return 0;
}
