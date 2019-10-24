#include "6dof_reader.hpp"

using namespace std;
using namespace cv;

// convert multi-bit image to 16bit rgb
// yuv only support:
// 8 and 10bit image
// png only support:
// 16bit image

// for YUV
void six_dof_reader::get_yuv_chan_10bit(ifstream& file, Mat& y_mat, Mat& u_mat, Mat& v_mat, int width, int height)
{
	//1. allocate memory 
    y_mat.create(height, width, CV_16UC1);
    u_mat.create(height/2, width/2, CV_16UC1);
    v_mat.create(height/2, width/2, CV_16UC1);
#ifdef USE_PTR
    unsigned short* y_mat_data = (unsigned short*)y_mat.data;
    unsigned short* u_mat_data = (unsigned short*)u_mat.data;
    unsigned short* v_mat_data = (unsigned short*)v_mat.data;
#endif

    // 2. read 10bit LE YUV frame and convert to 16bit
    int y_size = height*width;
    int uv_size = height*width/4;
#ifdef USE_PTR
    for(int i = 0; i < y_size; i++)
#else
    for(int y = 0; y < height; y++)
    for(int x = 0; x < width; x++)
#endif
    {
        unsigned short data;
        unsigned char data_u;
        unsigned char data_l;

	// @TODO, load once and convert 
	// @TODO, why not 2 byte operation 
        file.read(reinterpret_cast<char*>(&data_l), 1);
        file.read(reinterpret_cast<char*>(&data_u), 1);
        data = ((data_u << 8) | data_l) << 6;

#ifdef USE_PTR
        y_mat_data[i] = data;
#else
        y_mat.at<unsigned short>(y,x) = data;
#endif
    }

#ifdef USE_PTR
    for(int i = 0; i < uv_size; i++)
#else
    for(int y = 0; y < height/2; y++)
    for(int x = 0; x < width/2; x++)
#endif
    {
        unsigned short data;
        unsigned char data_u;
        unsigned char data_l;
        file.read(reinterpret_cast<char*>(&data_l), 1);
        file.read(reinterpret_cast<char*>(&data_u), 1);
        data = ((data_u << 8) | data_l) << 6;

#ifdef USE_PTR
        u_mat_data[i] = data;
#else
        u_mat.at<unsigned short>(y,x) = data;
#endif
    }

#ifdef USE_PTR
    for(int i = 0; i < uv_size; i++)
#else
    for(int y = 0; y < height/2; y++)
    for(int x = 0; x < width/2; x++)
#endif
    {
        unsigned short data;
        unsigned char data_u;
        unsigned char data_l;
        file.read(reinterpret_cast<char*>(&data_l), 1);
        file.read(reinterpret_cast<char*>(&data_u), 1);
        data = ((data_u << 8) | data_l) << 6;

#ifdef USE_PTR
        v_mat_data[i] = data;
#else
        v_mat.at<unsigned short>(y,x) = data;
#endif
    }

    resize(u_mat, u_mat, Size(), 2, 2, CV_INTER_NN);
    resize(v_mat, v_mat, Size(), 2, 2, CV_INTER_NN);
}

void six_dof_reader::get_yuv_chan_8bit(ifstream& file, Mat& y_mat, Mat& u_mat, Mat& v_mat, int width, int height)
{


    // 1. memory allocation 
    y_mat.create(height, width, CV_16UC1);
    u_mat.create(height/2, width/2, CV_16UC1);
    v_mat.create(height/2, width/2, CV_16UC1);
    unsigned short* y_mat_data = (unsigned short*)y_mat.data;
    unsigned short* u_mat_data = (unsigned short*)u_mat.data;
    unsigned short* v_mat_data = (unsigned short*)v_mat.data;


    // 2. read 10bit LE YUV frame and convert to 16bit
    int y_size = height*width;
    int uv_size = height*width/4;
    for(int i = 0; i < y_size; i++)
    {
        unsigned char data;
        file.read(reinterpret_cast<char*>(&data), 1);

        y_mat_data[i] = data << 8;
    }

    for(int i = 0; i < uv_size; i++)
    {
        unsigned char data;
        file.read(reinterpret_cast<char*>(&data), 1);

        u_mat_data[i] = data << 8;
    }

    for(int i = 0; i < uv_size; i++)
    {
        unsigned char data;
        file.read(reinterpret_cast<char*>(&data), 1);

        v_mat_data[i] = data << 8;
    }

    // 3. matching the U, V size to Y
    // @TODO:  use Y.size() instead of scaling factor 
    resize(u_mat, u_mat, Size(), 2, 2, CV_INTER_NN);
    resize(v_mat, v_mat, Size(), 2, 2, CV_INTER_NN);
}

Mat six_dof_reader::merge_yuv_chan(Mat& y_mat, Mat& u_mat, Mat& v_mat)
{
    vector<Mat> yuv_chan;
    yuv_chan.push_back(y_mat);
    yuv_chan.push_back(u_mat);
    yuv_chan.push_back(v_mat);
    Mat yuv_mat;
    merge(yuv_chan, yuv_mat);

    return yuv_mat;
}

Mat six_dof_reader::read_yuv(string image_name, int width, int height, int bit_depth)
{
    Mat y_image, u_image, v_image, rgb_image, yuv_image;
    ifstream image(image_name, ios::in | ios::binary);
    if (!image.is_open())
	    return rgb_image;

    switch(bit_depth)
    {
        case 8:
        get_yuv_chan_8bit(image, y_image, u_image, v_image, width, height);
        yuv_image = merge_yuv_chan(y_image, u_image, v_image);
        cvtColor(yuv_image, rgb_image, COLOR_YUV2BGR);
        break;

        case 10:
        get_yuv_chan_10bit(image, y_image, u_image, v_image, width, height);
        yuv_image = merge_yuv_chan(y_image, u_image, v_image);
        cvtColor(yuv_image, rgb_image, COLOR_YUV2BGR);
        break;
    }

    image.close(); // BUG FIX

    return rgb_image;
}

// convert depth to real distance
Mat six_dof_reader::depth_double_yuv(string image_name, int width, int height, int bit_depth, double min, double max)
{
    Mat depth;

    // 1. read depth (integer YUV format)
    Mat tmp = read_yuv(image_name, width, height, bit_depth);
    extractChannel(tmp, depth, 0);
    

    // 2. convert to double distance 
    Mat depth_double(height, width, CV_64FC1);
    unsigned short* depth_data = (unsigned short*)depth.data;
    double* depth_double_data = (double*)depth_double.data;

    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            double v_max = 65535;        
            if(max >= 1000)
                depth_double_data[i*width + j] = v_max*(min/(depth_data[i*width + j]));
            else
                depth_double_data[i*width + j] = (min*max*v_max)/((max - min)*(depth_data[i*width + j]) + v_max*min);
        }
    }

    return depth_double;
}
