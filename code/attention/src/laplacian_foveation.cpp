#include "laplacian_foveation.hpp"
#include <limits>
LaplacianBlending::LaplacianBlending(const int & _width, const int & _height, const int _levels, const int _sigma_x, const int _sigma_y) : width(_width), height(_height), levels(_levels),sigma_x(_sigma_x), sigma_y(_sigma_y)
{

    image_lap_pyr.resize(levels);
    foveated_pyr.resize(levels);
    image_sizes.resize(levels);
    kernel_sizes.resize(levels);
    kernels.resize(levels);

    CreateFilterPyr(width, height, _sigma_x, _sigma_y);
}

LaplacianBlending::~LaplacianBlending() {
    std::vector<cv::Mat>().swap(image_lap_pyr);
    std::vector<cv::Mat>().swap(foveated_pyr);
    std::vector<cv::Mat>().swap(kernels);
    std::vector<cv::Mat>().swap(image_sizes);
    std::vector<cv::Mat>().swap(kernel_sizes);
}

void LaplacianBlending::BuildPyramids(const cv::Mat & image) {

    cv::Mat current_img=image;
    cv::Mat lap;

    for (int l=0; l<levels; l++) {

        cv::pyrDown(current_img, down);
        cv::pyrUp(down, up, current_img.size());
        lap=current_img-up;
        
        image_lap_pyr[l]=lap.clone();
        current_img = down;
    }
            
    image_smallest_level=up;           
}

void LaplacianBlending::ComputeRois(const cv::Mat &center, cv::Rect &kernel_roi_rect,
                                    const cv::Mat &kernel_size, const cv::Mat &image_size) {

    // Kernel center - image coordinate
    cv::Mat upper_left_kernel_corner = (kernel_size) / 2.0 - center;

    // encontrar roi no kernel
    // cv::Rect take (upper left corner, width, heigth)
    kernel_roi_rect=cv::Rect(upper_left_kernel_corner.at<int>(0,0),
                             upper_left_kernel_corner.at<int>(1,0),
                             image_size.at<int>(0,0),
                             image_size.at<int>(1,0));
}


cv::Mat LaplacianBlending::Foveate(const cv::Mat &image, const cv::Mat &center) {
    BuildPyramids(image);

    for(int i=levels-1; i>=0; --i) {  
	cv::Mat image_size(2,1,CV_32S);
	image_size.at<int>(0,0)=image_lap_pyr[i].cols;
	image_size.at<int>(1,0)=image_lap_pyr[i].rows;
	image_sizes[i]=image_size;

	cv::Mat kernel_size(2,1,CV_32S);    
	kernel_size.at<int>(0,0)=kernels[i].cols;
	kernel_size.at<int>(1,0)=kernels[i].rows;
	kernel_sizes[i]=kernel_size;
    }


    image_smallest_level.copyTo(foveated_image);

    for(int i=levels-1; i>=0; --i) {
        
        cv::Rect image_roi_rect;  
        cv::Rect kernel_roi_rect;         
        cv::Mat aux;

        cv::Mat result_roi;
        cv::Mat aux_pyr;
        
        if(i!=0)
            aux=center/(powf(2,i));
        else
            aux=center;

        ComputeRois(aux, kernel_roi_rect, kernel_sizes[i], image_sizes[i]);
            
        // Multiplicar
        image_lap_pyr[i].copyTo(aux_pyr);
        cv::multiply(aux_pyr, kernels[i](kernel_roi_rect), result_roi, 1.0, aux_pyr.type());
        result_roi.copyTo(aux_pyr);

        if(i==(levels-1))
            cv::add(foveated_image,aux_pyr,foveated_image);
        else {
            cv::pyrUp(foveated_image, foveated_image, Size(image_sizes[i].at<int>(0,0),image_sizes[i].at<int>(1,0)));
            cv::add(foveated_image,aux_pyr,foveated_image);                   
        }
    }
        
    return foveated_image;
}


cv::Mat LaplacianBlending::CreateFilter(int m, int n, int sigma_x, int sigma_y) {

    cv::Mat gkernel(m,n,CV_64FC3);

    double r, rx, ry;
    double s_x = 2.0*sigma_x*sigma_x;
    double s_y = 2.0*sigma_y*sigma_y;
    double xc = n*0.5;
    double yc = m*0.5;
    double max_value = -std::numeric_limits<double>::max();

    for (int x=0; x<n; ++x) {
        
        rx = ((x-xc)*(x-xc));
        
        for(int y=0; y<m; ++y) {

            ry=((y-yc)*(y-yc));

	    double expression=exp(-rx/s_x)*exp(-ry/s_y);

            // FOR 3 CHANNELS
            gkernel.at<Vec3d>(y,x)[0] = expression;
            gkernel.at<Vec3d>(y,x)[1] = expression;
            gkernel.at<Vec3d>(y,x)[2] = expression;

            if(gkernel.at<Vec3d>(y,x)[0]>max_value)
                max_value=gkernel.at<Vec3d>(y,x)[0];
        }
    }

    // normalize the Kernel
    for(int x=0; x<n; ++x) {
        for(int y=0; y<m; ++y) {

            // FOR 3 CHANNELS
            gkernel.at<Vec3d>(y,x)[0] /= max_value;
            gkernel.at<Vec3d>(y,x)[1] /= max_value;
            gkernel.at<Vec3d>(y,x)[2] /= max_value;
        }
    }

    return gkernel;
}

void LaplacianBlending::CreateFilterPyr(int width, int height, const int _sigma_x, const int _sigma_y) {
    // Foveate images
    int m=floor(4*height);
    int n=floor(4*width);
    sigma_x=_sigma_x;
    sigma_y=_sigma_y;


    cv::Mat gkernel=CreateFilter(m,n,_sigma_x,_sigma_y);
    kernels[0]=gkernel;

    for (int l=1; l<levels; ++l) {
        cv::Mat kernel_down;
        cv::pyrDown(kernels[l-1], kernel_down);
        kernels[l]=kernel_down;
    }

}
