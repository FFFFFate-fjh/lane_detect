/////////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
// modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
//   notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote
//   products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
// disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any
// direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
///////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "autoCanny.hpp"

using namespace cv;
using namespace std;


// AutoCanny C++ implementation, used OpenCV Canny function. Input must be gray
// image. By exmperience, type = 0.66 is a normal choice in most case. If we
// want more details, 0.35 is the best choice.
//
// @type : 0.3 ~ 0.66
void autoCanny(const Mat &srcmat, Mat &dstmat, float type) {
    
    // Check input Mat type is CV_8UC1
    if (srcmat.type() != CV_8UC1 || dstmat.type() != CV_8UC1)
        CV_Error(CV_StsUnsupportedFormat, "");
    
    // CHeck output Mat size equal to input Mat size or not
    if (!CV_ARE_SIZES_EQ(&srcmat, &dstmat)) {
        dstmat.create(srcmat.size(), srcmat.type());
    }
    
    Mat equalarr;
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
    int thres2 = int(type * 2 * total_piexl);
    
    Canny(srcmat, dstmat, thres1, thres2);
}


// AutoCanny C++ implementation, used OpenCV Canny function.
// It maybe a little slower but give more detail edges.
// @input: must be BGR image
void autoDeepCanny(const cv::Mat &srcmat, cv::Mat &dstmat) {
    
    // Check input Mat type is CV_8UC3
    //    if (srcmat.type() != CV_8UC3 || dstmat.type() != CV_8UC3)
    //        CV_Error(CV_StsUnsupportedFormat, "");
    
    // CHeck output Mat size equal to input Mat size or not
    if (!CV_ARE_SIZES_EQ(&srcmat, &dstmat)) {
        dstmat.create(srcmat.size(), CV_8UC1);
    }
    
    // Each channel use autoCanny once, in order to get the different type of
    // color channel details. After that combine 3 channel Canny Edge into one.
    Mat equalarr;
    int nr = srcmat.rows;
    int nc = srcmat.cols;
    int nl = srcmat.cols * srcmat.channels();
    
    Mat blue_mat, green_mat, red_mat;
    blue_mat.create(srcmat.size(), CV_8UC1);
    green_mat.create(srcmat.size(), CV_8UC1);
    red_mat.create(srcmat.size(), CV_8UC1);
    
    // If the input and output mat is store continuous in memory, then loop
    // the Mat just in one rows will be much more quickly.
    for (int i = 0; i < nr; i++) {
        const uchar *src_data = srcmat.ptr<uchar>(i);
        uchar *blue_data = blue_mat.ptr<uchar>(i);
        uchar *green_data = green_mat.ptr<uchar>(i);
        uchar *red_data = red_mat.ptr<uchar>(i);
        int k = 0;
    
        for (int j = 0; j < nl; j++) {
            int remainder = (j + 1) % 3;
            if (remainder == 2){
                blue_data[k] = src_data[j];
            }
            else if (remainder == 1){
                green_data[k] = src_data[j];
            }
            else if (remainder == 0){
                red_data[k] = src_data[j];
                k++;
            }
        }
    }
    
    float lower = 0.35;
    autoCanny(blue_mat, blue_mat, lower);
    autoCanny(green_mat, green_mat, lower);
    autoCanny(red_mat, red_mat, lower);
    
    for (int i = 0; i < nr; i++) {
        uchar *dst_data = dstmat.ptr<uchar>(i);
        uchar *blue_data = blue_mat.ptr<uchar>(i);
        uchar *green_data = green_mat.ptr<uchar>(i);
        uchar *red_data = red_mat.ptr<uchar>(i);
        
        for (int j = 0; j < nc; j++) {
            dst_data[j] = (blue_data[j] + green_data[j] + red_data[j]);
        }
    }
}
