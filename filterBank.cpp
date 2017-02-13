// opencv/2.4.13.2
// fltk/1.3.4
#include <Fl/Fl.h>
#include <Fl/Fl_window.H>
#include <FL/FL_Button.H>
#include <FL/Fl_Input.H>
#include <FL/fl_ask.H>
#include <FL/Fl_File_Chooser.H>

#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo.hpp"

#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

Mat src;
Fl_Input*    filePath;
Fl_Input*    bil_d;
Fl_Input*    bil_sigmaColor;
Fl_Input*    bil_sigmaSpace;
Fl_Input*    strength;
Fl_Button*   load;
Fl_Button*   quit;
Fl_Button*   box;
Fl_Button*   gaussian;
Fl_Button*   median;
Fl_Button*   bilateral;
Fl_Button*   canny_edge;
Fl_Button*   nonlocal_mean;
Fl_Button*   kuwahara;
Fl_Button*   gabor;
Fl_Button*   laplacian;

//for Gaussian
int gaussian_ksize  = 0;
int gaussian_sigma = 0;
//int gaussian_sigmaY = 13;
//for canny_edge
Mat image, gray, edge, cedge;
//for kuwahara
double my_sum[4096][4096];
long double my_sqsum[4096][4096];

void load_click_callback(Fl_Widget* wid, void* data);
void quit_click_callback(Fl_Widget* wid, void* data);

void box_click_callback(Fl_Widget* wid, void* data);
void box_trackbar_callback(int ksize, void* data);
void box_filter();

void gaussian_click_callback(Fl_Widget* wid, void* data);
void gaussian_trackbar_callback(int, void* data);
void gaussian_filter();

void canny_edge_click_callback(Fl_Widget* wid, void* data);
void canny_edge_filter();
void canny_edge_trackbar_callback(int, void* data);

void median_click_callback(Fl_Widget* wid, void* data);
void median_filter();
void median_trackbar_callabck(int ksize, void * data);


void bilateral_click_callback(Fl_Widget* wid, void* data);
void bilateral_filter();


void nonlocal_mean_click_callback(Fl_Widget* wid, void* data);
void nonlocal_mean_filter();

void kuwahara_click_callback(Fl_Widget* wid, void* data);
void kuwahara_mean_filter();
void kuwahara_trackbar_callback(int ksize, void * data);
void kuwahara_produce(int kernel, Mat & output);
int get_mean(int h, int w, int neibourhood,int num);

void laplacian_click_callback(Fl_Widget* wid, void* data);
void laplacian_filter();
void laplacian_trackbar_callabck(int ksize, void * data);
void laplacian_produce(int ksize, Mat & output);



int main(int argc, char** args)
{
    char name [] = "Filter Banks";
    Fl_Window * filterBank = new Fl_Window(360, 380, name);
    filterBank -> begin();

    load = new Fl_Button(10,20,100,30,"Load");
    load -> callback(( Fl_Callback* ) load_click_callback);

    quit = new Fl_Button(250,20,100,30,"Quit");
    quit -> callback(( Fl_Callback* ) quit_click_callback, filterBank);

    box = new Fl_Button(10,80,100,30,"Box");
    box -> callback(( Fl_Callback* ) box_click_callback);
    box -> deactivate();

    gaussian = new Fl_Button(130,80,100,30,"Gaussian");
    gaussian -> callback(( Fl_Callback* ) gaussian_click_callback);
    gaussian -> deactivate();

    canny_edge = new Fl_Button(250, 80, 100, 30, "Candy Edge");
    canny_edge -> callback((Fl_Callback* ) canny_edge_click_callback);
    canny_edge -> deactivate();

    median = new Fl_Button(10, 130, 100, 30,"Median");
    median -> callback(median_click_callback);
    median -> deactivate();

    bil_d = new Fl_Input(130, 190, 100, 30, "Diameter: ");
    bil_d->deactivate();

    bil_sigmaColor = new Fl_Input(130, 230, 100, 30, "Color: ");
    bil_sigmaColor -> deactivate();

    bil_sigmaSpace = new Fl_Input(130, 270, 100, 30, "Space: ");
    bil_sigmaSpace -> deactivate();

    bilateral = new Fl_Button(250,230,100,30,"Bilateral");
    bilateral -> callback(bilateral_click_callback);
    bilateral -> deactivate();

    strength = new Fl_Input(130, 330, 100, 30, "Strength: ");
    strength -> deactivate();

    nonlocal_mean = new Fl_Button(250,330,100,30,"Nonlocal Mean");
    nonlocal_mean -> callback(nonlocal_mean_click_callback);
    nonlocal_mean -> deactivate();

    kuwahara = new Fl_Button(130,130,100,30,"Kuwahara");
    kuwahara -> callback(kuwahara_click_callback);
    kuwahara -> deactivate();

    laplacian = new Fl_Button(250,130,100,30, "Laplacian");
    laplacian -> callback(laplacian_click_callback);
    laplacian -> deactivate();

    filterBank -> end();
    //resizable(this);
    filterBank -> show();

    return Fl::run();
}

void load_click_callback(Fl_Widget* wid, void* data) {
    char* imagePath = fl_file_chooser("Select file", "Image Files (*.{bmp,gif,jpg,png})", "*.jpg");
    if(imagePath == NULL) return;
    string failure = string("Cannot read image file:") + string(imagePath);
    src = imread(imagePath);
    if(src.empty())
    {
        printf("Cannot read image file: %s\n", imagePath);
        return;
    }
    box -> activate();
    gaussian -> activate();
    canny_edge -> activate();
    median -> activate();
    bil_d -> activate();
    bil_sigmaColor -> activate();
    bil_sigmaSpace -> activate();
    bilateral -> activate();
    strength -> activate();
    nonlocal_mean -> activate();
    kuwahara -> activate();
    laplacian -> activate();
    imshow("source", src);
}

// Box Filter
// -----------------------------------------------------------------------
void box_click_callback(Fl_Widget* wid, void* data){
    box_filter();
}

void box_filter() {
    int kernel_size = 5;
    namedWindow("Box",1);
    createTrackbar("kernel size", "Box", &kernel_size, 100, box_trackbar_callback);
    box_trackbar_callback(kernel_size, 0);
	return;
}

void box_trackbar_callback(int ksize, void * data){
    Mat output;
    if(ksize % 2 == 0)
        ksize += 1;
    cout << "Box kernel size: " << ksize << endl;
    boxFilter(src, output, -1, Size(ksize,ksize));
    imshow("Box", output);
    return;
}
// -----------------------------------------------------------------------


// Quit buttion
// -----------------------------------------------------------------------
void quit_click_callback(Fl_Widget* wid, void* data){
    ((Fl_Window*)data)->hide();
}
// -----------------------------------------------------------------------



// Gaussian Filter
// -----------------------------------------------------------------------
void gaussian_click_callback(Fl_Widget* wid, void* data) {
    gaussian_filter();
    return;
}

void gaussian_filter() {
    // Create a window
    namedWindow("Gaussian", 1);

    // create a toolbar
    createTrackbar("ksize(odd)", "Gaussian", &gaussian_ksize, 100, gaussian_trackbar_callback);
	createTrackbar("sigma(X=Y)", "Gaussian", &gaussian_sigma, 100, gaussian_trackbar_callback);
	//createTrackbar("sigmaY", "Gaussian", &gaussian_sigmaY, 100, gaussian_trackbar_callback);
    // Show the image
    gaussian_trackbar_callback(gaussian_ksize, 0);
    gaussian_trackbar_callback(gaussian_sigma, 0);
    //gaussian_trackbar_callback(gaussian_sigmaX, 0);
    return;
}

void gaussian_trackbar_callback(int, void* data)
{
    Mat output;
	if(gaussian_ksize % 2 == 0)
        gaussian_ksize += 1;
    cout << "Gaussian kernel size: " << gaussian_ksize << " | sigma: " << gaussian_sigma << endl;
	GaussianBlur(src, output, Size(gaussian_ksize, gaussian_ksize), gaussian_sigma, gaussian_sigma);		imshow("Gaussian", output);
}
// -----------------------------------------------------------------------

// Canny Edge Filter
// -----------------------------------------------------------------------
void canny_edge_click_callback(Fl_Widget* wid, void* data){
    canny_edge_filter();
    return;
}

void canny_edge_filter() {
    int edgeThresh = 10;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    namedWindow("Canny Edge", 1);
    createTrackbar("Threshold", "Canny Edge", &edgeThresh, 100, canny_edge_trackbar_callback);
    canny_edge_trackbar_callback(edgeThresh, 0);
    return;
}

void canny_edge_trackbar_callback(int edgeThresh, void * data) {
    cout << "Canny edge threshold1: " << edgeThresh  << " | threshold2 " << edgeThresh * 3 << endl;
    blur(gray, edge, Size(3,3));
    // Run the edge detector on grayscale
    Canny(edge, edge, edgeThresh, edgeThresh*3, 3);
    cedge = Scalar::all(0);

    src.copyTo(cedge, edge);
    imshow("Canny Edge", cedge);
    return;
}
// -----------------------------------------------------------------------


// Median Filter
// -----------------------------------------------------------------------
void median_click_callback(Fl_Widget* wid, void* data) {
    median_filter();
    return;
}

void median_filter() {
    int ksize = 9;
    namedWindow("Median", 1);
    createTrackbar("kerel size(odd)", "Median", &ksize, 50, median_trackbar_callabck);
    median_trackbar_callabck(ksize, 0);
    return;
}

void median_trackbar_callabck(int ksize, void * data) {
    Mat output;
    if(ksize % 2 == 0)
        ksize += 1;
    cout << "Median kernel size: " << ksize << endl; 
    medianBlur(src, output, ksize);
    imshow("Median", output);
    return;
}
// -----------------------------------------------------------------------


// Bilateral Filter
// -----------------------------------------------------------------------
void bilateral_click_callback(Fl_Widget* wid, void* data) {
    bilateral_filter();
    return;
}

void bilateral_filter() {
    Mat dst;
    int d = strtol((bil_d->value()), NULL, 0);
    int sigmaColor = strtol((bil_sigmaColor -> value()), NULL, 0);
    int sigmaSpace = strtol((bil_sigmaSpace -> value()), NULL, 0);
    bilateralFilter(src, dst, d, sigmaColor, sigmaSpace);
    imshow("Bilateral", dst);
    return;
}
// -----------------------------------------------------------------------



// Non-local Mean Filter
// -----------------------------------------------------------------------
void nonlocal_mean_click_callback(Fl_Widget* wid, void* data) {
    nonlocal_mean_filter();
    return;
}

void nonlocal_mean_filter() {
    Mat output;
    int s = strtol((strength -> value()), NULL, 0);
    fastNlMeansDenoisingColored(src, output, (float) s);
    imshow("Non-local Mean", output);
    return;
}
// -----------------------------------------------------------------------

// Kuwahara Filter
// -----------------------------------------------------------------------
void kuwahara_click_callback(Fl_Widget* wid, void* data) {
    kuwahara_mean_filter();
    return;
}

void kuwahara_mean_filter() {
    int kernel_size = 5;
    namedWindow("Kuwahara");
    createTrackbar("kernel size", "Kuwahara", &kernel_size, 50, kuwahara_trackbar_callback);
    kuwahara_trackbar_callback(kernel_size, 0);
	return;
}

void kuwahara_trackbar_callback(int ksize, void * data) {
    Mat output;
    cout << "Kuwahara kernel size: " << ksize << endl;
    kuwahara_produce(ksize, output);
    imshow("Kuwahara", output);
    return;
}

void kuwahara_produce(int kernel, Mat & image_out) {
    Mat image_in;
    cvtColor(src, image_in, COLOR_BGR2GRAY);
    image_in.copyTo(image_out);
    const int height = image_in.rows;
    const int width = image_in.cols;
        if (kernel/2==0) {
            kernel += 1;
        }
        for (int h = 0; h < image_in.rows; h++) {
            uchar* data = image_in.ptr<uchar>(h);
            for (int w = 0; w < image_in.cols; w++) {
                   my_sum[h+1][w+1] = data[w] + my_sum[h][w+1] + my_sum[h+1][w] - my_sum[h][w];
            }
        }
        for (int h = 0; h < image_in.rows; h++) {
            uchar* data = image_in.ptr<uchar>(h);
            for (int w = 0; w < image_in.cols; w++) {
                 my_sqsum[h+1][w+1] = data[w] * data[w] + my_sqsum[h + 1][w] + my_sqsum[h][w + 1] - my_sqsum[h][w];
            }
        }
        int neibourhood = kernel / 2 + 1;
        float N = neibourhood*neibourhood;
        for (int h = neibourhood; h < height-neibourhood;h++) {
            for (int w = neibourhood; w < width-neibourhood;w++) {
                double L_T_sqsum = my_sqsum[h+1][w+1] - my_sqsum[h - neibourhood+1][w+1] - my_sqsum[h+1 ][w - neibourhood+1] + my_sqsum[h - neibourhood+1][w - neibourhood+1];
                double L_T_sum = my_sum[h+1][w+1] - my_sum[h - neibourhood+1][w+1] - my_sum[h+1][w - neibourhood+1] + my_sum[h - neibourhood+1][w - neibourhood+1];
                double left_top = L_T_sqsum - L_T_sum*L_T_sum / N;
                double R_T_sqsum = my_sqsum[h+1][w  + neibourhood] - my_sqsum[h+1][w] - my_sqsum[h - neibourhood+1][w + neibourhood] + my_sqsum[h - neibourhood+1][w];
                double R_T_sum = my_sum[h+1][w + neibourhood] - my_sum[h+1][w] - my_sum[h - neibourhood+1][w + neibourhood] + my_sum[h - neibourhood+1][w];
                double right_top = R_T_sqsum - R_T_sum*R_T_sum / N;
                double L_B_sqsum = my_sqsum[h  + neibourhood][w+1] - my_sqsum[h][w+1] - my_sqsum[h + neibourhood][w - neibourhood+1] + my_sqsum[h][w - neibourhood+1];
                double L_B_sum = my_sum[h + neibourhood][w+1] - my_sum[h][w+1] - my_sum[h + neibourhood][w - neibourhood+1] + my_sum[h][w - neibourhood+1];
                double left_bottom = L_B_sqsum - L_B_sum*L_B_sum / N;
                double R_B_sqsum = my_sqsum[h + neibourhood][w + neibourhood] - my_sqsum[h][w  + neibourhood] - my_sqsum[h + neibourhood][w ] + my_sqsum[h ][w ];
                double R_B_sum = my_sum[h + neibourhood][w + neibourhood] - my_sum[h][w + neibourhood] - my_sum[h + neibourhood][w ] + my_sum[h ][w ];
                double right_bottom = R_B_sqsum - R_B_sum*R_B_sum / N;
                double min_var = (left_top > right_top) ? right_top : left_top;
                min_var = (left_bottom > min_var) ? min_var : left_bottom;
                min_var = (right_bottom > min_var) ? min_var : right_bottom;
                uchar kuwahara_val = 0;
                if (min_var == left_top)  kuwahara_val = get_mean(h, w,neibourhood, 1);
                else if (min_var == right_top) kuwahara_val=get_mean( h, w, neibourhood, 2);
                else if (min_var == left_bottom) kuwahara_val = get_mean(h, w, neibourhood, 3);
                else if(min_var==right_bottom)  kuwahara_val = get_mean( h, w, neibourhood, 4);

                if (kuwahara_val > 255)  kuwahara_val = 255;
                else if (kuwahara_val < 0) kuwahara_val = 0;

                uchar* point = image_out.ptr<uchar>(h);
                point[w] = kuwahara_val;
            }
        }
        return;
}
int get_mean(int h, int w, int neibourhood,int num) {
    int val;
    switch (num)
    {
    case 1: val = (int)(my_sum[h+1][w+1] - my_sum[h - neibourhood+1][w+1] - my_sum[h+1][w - neibourhood+1] + my_sum[h - neibourhood+1][w - neibourhood+1]) / (neibourhood*neibourhood); break;
    case 2: val = (int)(my_sum[h+1][w + neibourhood] - my_sum[h+1][w] - my_sum[h - neibourhood+1][w + neibourhood] + my_sum[h - neibourhood+1][w]) / (neibourhood*neibourhood); break;
    case 3: val = (int)(my_sum[h + neibourhood][w+1] - my_sum[h][w+1] - my_sum[h + neibourhood][w - neibourhood+1] + my_sum[h][w - neibourhood+1]) / (neibourhood*neibourhood); break;
    case 4: val = (int)(my_sum[h + neibourhood][w + neibourhood] - my_sum[h + neibourhood][w] - my_sum[h][w + neibourhood] + my_sum[h][w ]) / (neibourhood*neibourhood); break;
    }
    return val;
}
// -----------------------------------------------------------------------



void laplacian_click_callback(Fl_Widget* wid, void* data) {
    laplacian_filter();
    return;
}
void laplacian_filter() {
    int ksize = 9;
    namedWindow("Laplacian", 1);
    createTrackbar("kernel size", "Laplacian", &ksize, 30, laplacian_trackbar_callabck);
    laplacian_trackbar_callabck(ksize, 0);
    return;
}
void laplacian_trackbar_callabck(int ksize, void * data) {
    Mat output;
    if(ksize %2 == 0)
        ksize += 1;
    cout << "Laplacian desired depth: " << ksize << endl;
   // gabor_produce(ksize, output);
    cv::Laplacian(src, output, CV_16S, ksize);
    imshow("Laplacian", output);
    return;
}

void laplacian_produce(int kernel_size, Mat & output) {
    if(kernel_size % 2 == 0)
            kernel_size += 1;
    Mat dest;
    Mat src_f;
    src.convertTo(src_f,CV_32F);
    double sig = 1, th = 0, lm = 1.0, gm = 0.02, ps = 0;
    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
    cv::filter2D(src_f, dest, CV_32F, kernel);
    dest.convertTo(output,CV_8U,1.0/255.0); // move to proper[0..255] range to show it
    return;
}
