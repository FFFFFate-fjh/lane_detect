#include<iostream>
#include<string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

Mat src, src_gray,src_heq;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";
int ROI_x_l=475;
int ROI_y_l=100;
int ROI_x_r=600;
int ROI_y_r=100;

void CannyThreshold(int, void*);
Mat lane_perspective(Mat src);
Mat sobel(Mat src);
Mat set_ROI(Mat src, int ROI_L, int ROI_R);
int autoCanny(Mat &srcmat, Mat &dstmat, float type);

void CannyThreshold(int, void*)
{

   Mat rotated = lane_perspective(src_heq);

   /// Apply Histogram Equalization
//   equalizeHist( rotated, rotated );

  /// Reduce noise with a kernel 3x3
//  blur( src_gray, detected_edges, Size(3,3) );
//   medianBlur(rotated,detected_edges,3);
//   GaussianBlur(rotated,detected_edges,Size(3,3),0,0);

   Mat ROI_img_left = set_ROI(rotated,ROI_x_l,ROI_y_l);
   Mat ROI_img_right = set_ROI(rotated,ROI_x_r,ROI_y_r);

   Mat grad_l, abs_grad_x_l,grad_r, abs_grad_x_r;
   medianBlur(ROI_img_left,grad_l,3);
   abs_grad_x_l = sobel(grad_l);
   medianBlur(ROI_img_right,grad_r,3);
   abs_grad_x_r = sobel(grad_r);

   Mat detected_edges_l, detected_edges_r;

/// Canny detector
//  autoDeepCanny(abs_grad_x_l, detected_edges_l);
   int thres_l, thres_r,thres_max;
    thres_l=autoCanny(grad_l, detected_edges_l, 1);
    thres_r=autoCanny(grad_r, detected_edges_r, 1);
    if(thres_max>100)
        thres_max= max(thres_l,thres_r);
    else
        thres_max=100;
    Canny( abs_grad_x_l, detected_edges_l, thres_max, thres_max*3, kernel_size );
    Canny( abs_grad_x_r, detected_edges_r, thres_max, thres_max*3, kernel_size );
//    Canny( abs_grad_x_l, detected_edges_l, lowThreshold, lowThreshold*ratio, kernel_size );
//    Canny( abs_grad_x_r, detected_edges_r, lowThreshold, lowThreshold*ratio, kernel_size );


  Mat hdst_l, hdst_r,roi_show;
  cvtColor(detected_edges_l, hdst_l, CV_GRAY2BGR);
  cvtColor(detected_edges_r, hdst_r, CV_GRAY2BGR);
  cvtColor(rotated, roi_show, CV_GRAY2BGR);
  vector<Vec4i> lines_l,lines_r;

  HoughLinesP(detected_edges_l, lines_l, 1, CV_PI/180, 75, 10, 20 );
  HoughLinesP(detected_edges_r, lines_r, 1, CV_PI/180, 75, 10, 20 );
  for( size_t i = 0; i < lines_l.size(); i++ )
  {
    Vec4i l = lines_l[i];
    float k = ((l[0]-l[2])*1.0)/(l[1]-l[3]);
    if(k<1 && k >-1)
    {
        line( hdst_l, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
        l[0]=l[0]+ROI_x_l;
        l[2]=l[2]+ROI_x_l;
        l[1]=l[1]+ROI_y_l;
        l[3]=l[3]+ROI_y_l;
        line( roi_show, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }
  }
  for( size_t i = 0; i < lines_r.size(); i++ )
  {
    Vec4i l = lines_r[i];
    float k = ((l[0]-l[2])*1.0)/(l[1]-l[3]);
    if(k<1 && k >-1)
    {
        line( hdst_r, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
        l[0]=l[0]+ROI_x_r;
        l[2]=l[2]+ROI_x_r;
        l[1]=l[1]+ROI_y_r;
        l[3]=l[3]+ROI_y_r;
        line( roi_show, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }
  }

  imshow("source_l", detected_edges_l);
  imshow("detected lines left", hdst_l);
  imshow("source_r", detected_edges_r);
  imshow("detected lines right", hdst_r);
  imshow("detected lines on rotated", roi_show);
 }


/** @function main */
int main( int argc, char** argv )
{
  /// Load an image
  src = imread( "./data/21/image_1/001000.png" );

  if( !src.data )
  { return -1; }

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  equalizeHist( src_gray, src_heq );

  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  /// Show the image
  CannyThreshold(0, 0);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
  }

Mat lane_perspective(Mat src)
{
    vector<Point> not_a_rect_shape;
    not_a_rect_shape.push_back(Point(400,233));
    not_a_rect_shape.push_back(Point(800,233));
    not_a_rect_shape.push_back(Point(894,369));
    not_a_rect_shape.push_back(Point(305,369));

   const Point* point = &not_a_rect_shape[0];
   int n = (int )not_a_rect_shape.size();
   Mat draw = src.clone();
   polylines(draw, &point, &n, 1, true, Scalar(0, 255, 0), 3, CV_AA);
   imshow( "draw", draw);

   //  topLeft, topRight, bottomRight, bottomLeft
  Point2f src_vertices[4];
  src_vertices[0] = not_a_rect_shape[0];
  src_vertices[1] = not_a_rect_shape[1];
  src_vertices[2] = not_a_rect_shape[2];
  src_vertices[3] = not_a_rect_shape[3];

  Point2f dst_vertices[4];
  dst_vertices[0] = Point(400,233);
  dst_vertices[1] = Point(800,233);
  dst_vertices[2] = Point(700,369);
  dst_vertices[3] = Point(515,369);

  Mat warpMatrix = getPerspectiveTransform(src_vertices, dst_vertices);
  Mat rotated;
  warpPerspective(src, rotated, warpMatrix, rotated.size(), INTER_LINEAR, BORDER_CONSTANT);

  // Display the image
   namedWindow( "warp perspective");
   imshow( "warp perspective",rotated);

   return rotated;
}

Mat sobel(Mat src)
{
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

  Sobel( src, grad_x, ddepth, 1, 0, 3, 1, 0, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
//   Sobel( detected_edges, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
//   convertScaleAbs( grad_y, abs_grad_y );

//   /// Total Gradient (approximate)
//   addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  return abs_grad_x;
}

Mat set_ROI(Mat src, int ROI_L, int ROI_R)
{
//    Rect roi(500, 200, 200, 169);
//    Rect roi(500, 200, 200, 169);
    Rect roi(ROI_L, ROI_R, 100, 269);
    Mat roi_of_image = src(roi);
    return roi_of_image;
}

int autoCanny(Mat &srcmat,  Mat &dstmat, float type)
{
    // Check input Mat type is CV_8UC1
    if (srcmat.type() != CV_8UC1 || dstmat.type() != CV_8UC1)
        CV_Error(CV_StsUnsupportedFormat, "");

    // CHeck output Mat size equal to input Mat size or not
    if (!CV_ARE_SIZES_EQ(&srcmat, &dstmat)) {
        dstmat.create(srcmat.size(), srcmat.type());
    }

    int total_piexl = 0;
    int nr = srcmat.rows;
    int nc = srcmat.cols;

    // If the input and output mat is store continuous in memory, then loop
    // the Mat just in one rows will be much more quickly.
    if (srcmat.isContinuous() && dstmat.isContinuous()) {
        nr = 1;
        nc = nc * srcmat.rows;
    }

    // Means gray level in image.
    for (int i = 0; i < nr; i++) {
        const uchar *src_data = srcmat.ptr<uchar>(i);
        for (int j = 0; j < nc; j++) {
            total_piexl += *src_data;
        }
    }

    total_piexl /= nr * nc; // means gray level

    // threshold 1 is 0.66 * means gray level
    // threshold 2 is 1.33 * means gray level
    // the following threshould value is calc from experience.
    int thres1 = int(type * total_piexl);
    int thres2 = int(type * 3 * total_piexl);
    cout<<"thres1  "<<thres1<<"  "<<"thres2  "<<thres2<<endl;
//    Canny(cannymat, dstmat, thres1, thres2, kernel_size);
    return thres1;
}
