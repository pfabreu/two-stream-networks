#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <limits>
#include <time.h>

#include "laplacian_foveation.hpp"

using namespace std;


cv::Mat foveate(const cv::Mat &img, const int &size_map,
				const int &levels, const int &sigma,
				const cv::Mat &fixation_point);

std::vector<cv::Mat> FixationPoints (int img_size, int n_points, int random);

template<typename T>
string ToString(T t) {

	stringstream ss;
	ss << t;

	return ss.str();
}

////////////////////
//      MAIN      //
////////////////////


int main(int argc, char** argv){

	// Init
	::google::InitGoogleLogging(argv[0]);

	/* network params */
	const string absolute_path_folder = string(argv[1]);
	const string model_file 		  = absolute_path_folder + string(argv[2]);
	const string weight_file          = absolute_path_folder + string(argv[3]);
	const string mean_file            = absolute_path_folder + string(argv[4]);
	const string label_file           = absolute_path_folder + string(argv[5]);
	const string dataset_folder       = string(argv[6]);
	static int N                      = atoi(argv[7]);			// Define number of top predicted labels
	static string threshs_            = string(argv[8]);		// Segmentation threshold for mask
	static int size_map 			  = atoi(argv[9]);          // Size of the network input images (227,227)
	static int levels 				  = atoi(argv[10]);         // Number of kernel levels
	static string sigmas_             = string(argv[11]);       // Size of the fovea
	static string results_folder      = string(argv[12]);       // Folder to store results
	static int mode                   = atoi(argv[13]);         // Mode
	static bool debug                 = atoi(argv[14]);         // Set debug = 1 to see figures
	static int total_images           = atoi(argv[15]);         // Number of images
	if (strcmp(argv[16], "CPU") == 0)                           // Set Mode
		Caffe::set_mode(Caffe::CPU);
	else {
		Caffe::set_mode(Caffe::GPU);
		int device_id = atoi(argv[17]);
		Caffe::SetDevice(device_id);
				std::cout << "GPU MODE" << std::endl;
	}
	static int npoints                  = atoi(argv[18]);         // Number of fixation points
	static bool random                  = atoi(argv[19]);         // Set random = 1 to random fixation points

	std::cout << "Absolute Path Folder: " 	<< absolute_path_folder<< std::endl;
	std::cout << "Model File: " 			<< model_file << std::endl;
	std::cout << "Weights File: " 			<< weight_file << std::endl;
	std::cout << "Mean File: " 				<< mean_file<< std::endl;
	std::cout << "Labels File: " 			<< label_file<< std::endl;
	std::cout << "Dataset Folder: " 		<< dataset_folder<< std::endl;
	std::cout << "Results Folder: "			<< results_folder << std::endl;
	std::cout << "Top Classes: " 			<< N << std::endl;
	std::cout << "Levels: " 				<< levels << std::endl;
	std::cout << "Sigmas: " 				<< sigmas_ << std::endl;
	std::cout << "Threshs: "				<< threshs_<< std::endl;
	std::cout << "Size Map: " 				<< size_map << std::endl;
	std::cout << "Mode: " 					<< mode << std::endl;
	std::cout << "N Pts: "                  << npoints << std::endl;
	std::cout << "Random : "                << random << std::endl;
	std::cout << "Total_images: " 			<< total_images << std::endl;
	std::cout << "Debug: "                  << debug << std::endl;

	std::vector<float> threshs;
	std::stringstream ss_(threshs_);
	float iter_th;
	while (ss_ >> iter_th) {
		threshs.push_back(iter_th);
		if (ss_.peek() == ',')
			ss_.ignore();
	}

	std::vector<int> sigmas;
	std::stringstream ss(sigmas_);
	int iter_sig;
	while (ss >> iter_sig) {
		sigmas.push_back(iter_sig);
		if (ss.peek() == ',')
			ss.ignore();
	}

	// Load network, pre-processment, set mean and load labels
	Network Network(model_file, weight_file, mean_file, label_file);

	// Load List of images
	std::vector<cv::String> image_image_files ;
	image_image_files = Network.GetDir (dataset_folder, image_image_files);

	if(total_images>image_image_files.size())
		total_images=image_image_files.size();
	glob(dataset_folder, image_image_files);

	// store results
	ofstream feedforward_detection;
	ofstream feedback_detection;

	// File with 5 classes + scores + 5 bounding boxes
	// 1st classification and localization
	std::string feedforward_detection_str=results_folder+"feedfoward_detection_"+
														 "t"    + ToString(threshs.size()) +
														 "s"   + ToString(sigmas.size())  +
														 "p"   + ToString(npoints)+
														 "r"   + ToString(random)         +
														 "i"   + ToString(total_images)   + ".txt";                                        

	feedforward_detection.open (feedforward_detection_str.c_str(),ios::out);
	feedforward_detection<<"sigma;thres;pt_w;pt_h;class1;score1;x1;y1;w1;h1;class2;score2;x2;y2;w2;h2;class3;score3;x3;y3;w3;h3;class4;score4;x4;y4;w4;h4;class5;score5;x5;y5;w5;h5"<<std::endl;

	// File with 5 classes + scores + 5 bounding boxes
	//  Re classification and Re localization
	std::string feedback_detection_str=results_folder+"feedback_detection_"+
														 "t"    + ToString(threshs.size()) +
														 "s"   + ToString(sigmas.size())  +
														 "p"   + ToString(npoints)+
														 "r"   + ToString(random)         +
														 "i"   + ToString(total_images)   + ".txt";                                        

	feedback_detection.open (feedback_detection_str.c_str(),ios::out);
	feedback_detection<<"sigma;thres;pt_w;pt_h;class1;score1;x1;y1;w1;h1;class2;score2;x2;y2;w2;h2;class3;score3;x3;y3;w3;h3;class4;score4;x4;y4;w4;h4;class5;score5;x5;y5;w5;h5"<<std::endl;

	// Seed for random fixation points
	srand (time(NULL));
	std::vector<cv::Mat> fixedpts = FixationPoints(size_map,npoints,random);
 
   // Total number of iterations
	int total_iterations=total_images*threshs.size()*sigmas.size()*fixedpts.size();

	float thresh;
	int sigma;
	cv::Mat fixedpt;

	// For each threshold
	for (unsigned int thresh_index=0; thresh_index<threshs.size(); ++thresh_index){
		thresh=threshs[thresh_index];

		// For each sigma
		for (unsigned int sigma_index = 0;sigma_index < sigmas.size(); ++sigma_index){
			sigma=sigmas[sigma_index];

			// For each fixation point
			for (unsigned int fixedpt_index = 0; fixedpt_index < fixedpts.size(); ++fixedpt_index){
				fixedpt = fixedpts[fixedpt_index];
				
				// For each image
				for (unsigned int input=0; input<total_images; ++input){

					// Preprocess each image:
					//  Read image
					//  Resize to network size 
					//  Save it in img_orig    
					string file = image_image_files[input];
					cv::Mat img = cv::imread(file, 1);
					resize(img,img, Size(size_map,size_map)); 
					cv::Mat img_orig=img.clone();

					// Varibles
					std::vector<string> labels;
					std::vector<float> scores;
					std::vector<int> indexs;
					std::vector<Rect> bboxes1;
					std::vector<Rect> bboxes2;

					// Set a specific foveation point				
					//fixedpt.at<int>(0,0)=50;
					//fixedpt.at<int>(1,0)=170;

					// Atual iteration
					int iteration=input+1+fixedpt_index*total_images+
								  sigma_index*total_images*fixedpts.size()+
								  thresh_index*total_images*sigmas.size()*fixedpts.size();
					std::cout << "Th:" << thresh
							  << "\tSig:" << sigma
							  << "\tPt: (" << fixedpt.at<int>(0,0)<<","<<fixedpt.at<int>(1,0)<<')'
							  << "\tImg:" << input+1
							  << " of " << total_images
							  << "\tIter: " << iteration
							  << " of "  << total_iterations
							  << " ("<< 100.0*(iteration)/(total_iterations) << "%)"<< std::endl;

					// 1st Foveation
					img=foveate(img,size_map,levels,sigma,fixedpt);
					cv::Mat img_first_pass=img.clone();
					
					// Uncomment for Visualize Foveated Image
					//Network.VisualizeFoveation(fixedpt,img,sigma,fixedpt_index);

					// 1st Feedforward with foveated image
					//  Prediciton of TOP 5 classes
					ClassData first_pass_data = Network.Classify(img, N);
					
					std::cout << "----- 1st Feedfoward Pass ------" << std::endl;
					std::cout << first_pass_data << std::endl;

					
					// Store results
					feedforward_detection << std::fixed << std::setprecision(4) << sigma << ";" << thresh 
										  << ";" << fixedpt.at<int>(0,0) << ";" << fixedpt.at<int>(1,0) << ";";
					feedback_detection    << std::fixed << std::setprecision(4) << sigma << ";" << thresh 
										  << ";" << fixedpt.at<int>(0,0) << ";" << fixedpt.at<int>(1,0) << ";";

					cv::Mat img_=img_orig.clone();

					// For each predicted class labels
					for (int class_index = 0; class_index < N; ++class_index) {
						cv::Mat img_second_pass;

						/////////////////////////////////////////////
						//  Weakly Supervised Object Localization  //
						// Saliency Map + Segmentation Mask + BBox //
						/////////////////////////////////////////////
						
						Rect Min_Rect = Network.CalcBBox(class_index,img_, first_pass_data, thresh);
						img_=img_orig.clone();

						
						// Save all bounding boxes
						bboxes1.push_back(Min_Rect);

						// Store results
						if (class_index==N-1) {
							feedforward_detection << first_pass_data.label[class_index] << ";" 
												  << first_pass_data.score[class_index] << ";"
												  << Min_Rect.x << ";" << Min_Rect.y << ";"
												  << Min_Rect.width << ";" << Min_Rect.height;
							feedforward_detection << endl;
						}
						else {
							feedforward_detection << first_pass_data.label[class_index] << ";" 
												  << first_pass_data.score[class_index] << ";"
												  << Min_Rect.x << ";" << Min_Rect.y << ";"
												  << Min_Rect.width << ";" << Min_Rect.height << ";";
						}

						/////////////////////////////////////////////////////////
						//      Image Re-Classification with Attention         //
						// Foveated Image + Forward + Predict new class labels //
						/////////////////////////////////////////////////////////

						cv::Mat fixation_point(2,1,CV_32S);
						fixation_point.at<int>(0,0) = Min_Rect.y + Min_Rect.height/2;
						fixation_point.at<int>(1,0) = Min_Rect.x + Min_Rect.width/2;
						
						img_second_pass=foveate(img_orig,size_map,levels,sigma,fixation_point);
						//Network.VisualizeFoveation(fixation_point,img_second_pass,sigma,class_index+1);

						// 2nd Feedforward with foveated image
						//  Prediciton New top 5 of each predicted class
						ClassData feedback_data = Network.Classify(img_second_pass, N);
						
						std::cout << "----- 2nd Feedfoward Pass ------" << std::endl;
						std::cout << feedback_data << std::endl;

						// For each bounding box
						for(int m=0; m<N; ++m) {
							// Save all labels, scores and indexes
							labels.push_back(feedback_data.label[m]);
							scores.push_back(feedback_data.score[m]);
							indexs.push_back(feedback_data.index[m]);
						}

						if(debug) {
							Mat dst;
							cv::hconcat(img_orig,img_first_pass, dst); // horizontal
							cv::hconcat(dst, img_second_pass, dst); // horizontal
							//cv::vconcat(a, b, dst); // vertical
							namedWindow( "original image,    first pass,   second pass     class", WINDOW_AUTOSIZE ); // Create a window for display.
							imshow( "original image,    first pass,   second pass     class", dst );                  // Show our image inside it.
							waitKey(0);
						}
					}

					/////////////////////////////////////////////////////////
					//      Image Re-Localization with Attention           //
					// Saliency Map + Segmentation Mask + BBox             //
					/////////////////////////////////////////////////////////

					// Variables for top 5 ranked classes
					std::vector<string> top_final_labels;
					std::vector<float> top_final_scores;
					std::vector<int> top_final_index;
					
					// Sorting
					std::vector<int> sort_scores_index  = ArgMax(scores, N*N);

					// Finding top 5 diferent labels 
					int top = 0;
					while(top_final_labels.size() < 5){
						int idx = sort_scores_index[top];
						if((std::find(top_final_labels.begin(), top_final_labels.end(), labels[idx])) == (top_final_labels.end())) {
							top_final_labels.push_back(labels[idx]);
							top_final_scores.push_back(scores[idx]);
							top_final_index.push_back(indexs[idx]);             
						}
						top ++;
					}

					// Feedfoward - 2nd Prediciton of TOP N classes
					ClassData feedback_top_final_data = ClassData(top_final_labels,top_final_scores,top_final_index);
					
					std::cout << "----- Ranked ------" << std::endl;
					std::cout << feedback_top_final_data << std::endl;


					for (int class_index = 0; class_index < N; ++class_index) {				
					   
						Rect Min_Rect = Network.CalcBBox(class_index, img, feedback_top_final_data, thresh);
						
						// Save all bounding boxes
						bboxes2.push_back(Min_Rect);

						// store results
						if (class_index == N-1) {
							feedback_detection << feedback_top_final_data.label[class_index] << ";" << feedback_top_final_data.score[class_index] 
							<< ";" << Min_Rect.x << ";" << Min_Rect.y << ";" << Min_Rect.width << ";" << Min_Rect.height ;
							feedback_detection << endl;
						}
						else
							feedback_detection << feedback_top_final_data.label[class_index] << ";" << feedback_top_final_data.score[class_index] 
							<< ";" << Min_Rect.x << ";" << Min_Rect.y << ";" << Min_Rect.width << ";" << Min_Rect.height << ";";
					   
					}

					// Uncomment for Visualize Bounding Boxes 1st pass
					//Network.VisualizeBBox(bboxes1,N,img_orig,size_map,1);
					// Uncomment for Visualize Bounding Boxes 2nd pass
					//Network.VisualizeBBox(bboxes2,N,img,size_map,2);

				}
			}
		}
	}
	feedforward_detection.close();
	feedback_detection.close();
}



//////////////////////
// FOVEATE FUNTION  //
//////////////////////

cv::Mat foveate(const cv::Mat &img, const int &size_map,
				const int &levels, const int &sigma,
				const cv::Mat &fixation_point) {

	cv::Mat image;
	img.convertTo(image, CV_64F);

	// Construct Pyramid
	LaplacianBlending pyramid(image,levels, sigma);

	// Foveate
	cv::Mat foveated_image = pyramid.Foveate(fixation_point);

	foveated_image.convertTo(foveated_image,CV_8UC3);
	cv::resize(foveated_image,foveated_image,Size(size_map,size_map));

	return foveated_image;
}



//////////////////////////////
// FIXATION POINTS FUNTION  //
//////////////////////////////

std::vector<cv::Mat> FixationPoints (int img_size, int n_points, int random) {

	std::vector<cv::Mat> fixation_points;

	std::cout<<"Fixation Points: " << std::endl;

		int bins = sqrt(n_points);

		if (random) {
			for (int i = 0; i < n_points; i++) {
				cv::Mat fixation_point(2,1,CV_32S);
				fixation_point.at<int>(0,0) = img_size*0.05 + rand() % (int)(img_size-img_size*0.05);
				fixation_point.at<int>(1,0) = img_size*0.05 + rand() % (int)(img_size-img_size*0.05);
						std::cout<<"("<<fixation_point.at<int>(0,0)<<","<<fixation_point.at<int>(1,0)<< ')'<< std::endl;
					  fixation_points.push_back(fixation_point);
			}
		}
		else {
			for (int i = 0; i < bins; i++) {
				for (int j = 0; j < bins; j++) {
					cv::Mat fixation_point(2,1,CV_32S);
					fixation_point.at<int>(0,0) = img_size / bins * j + (img_size / bins /2);
					fixation_point.at<int>(1,0) = img_size / bins * i + (img_size / bins / 2);
					std::cout<<"("<<fixation_point.at<int>(0,0)<<","<<fixation_point.at<int>(1,0)<< ')'<< std::endl;
					fixation_points.push_back(fixation_point);
				}
			}
		}		
	return fixation_points;
}

