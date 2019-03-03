#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class LaplacianBlending {

    public:
        LaplacianBlending(const int & _width, const int & _height, const int _levels, const int _sigma_x, const int _sigma_y);
        ~LaplacianBlending();


        cv::Mat Foveate(const cv::Mat &image, const cv::Mat &center);

	// Change
	void CreateFilterPyr(int width, int height, const int _sigma_x, const int _sigma_y);
    private:
        cv::Mat image;
        int levels;
        int width, height;
        std::vector<cv::Mat> kernels;
        std::vector<cv::Mat> image_lap_pyr;
        std::vector<cv::Mat> foveated_pyr;
        std::vector<cv::Mat> image_sizes;
        std::vector<cv::Mat> kernel_sizes;
        int sigma_x, sigma_y;
        
        cv::Mat image_smallest_level;
        cv::Mat down;
        cv::Mat up;           
        cv::Mat foveated_image; 


	// Private auxiliary methods
	cv::Mat CreateFilter(int m, int n, int sigma_x, int sigma_y);

        //void BuildPyramids();
	void BuildPyramids(const cv::Mat & image);
        void ComputeRois(const cv::Mat &center, cv::Rect &kernel_roi_rect,
                         const cv::Mat &kernel_size, const cv::Mat &image_size);
};