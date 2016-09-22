/*
 * (C) Copyright 2010-2014 - Marcos Nieto
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser General Public License
 * (LGPL) version 2.1 which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/lgpl-2.1.html
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * Contributors:
 *     Marcos Nieto 
 *			 marcos dot nieto dot doncel at gmail dot com
 *			 Homepage: http://marcosnietoblog.wordpress.com
 */

#ifndef __ERRORNIETO_H__
#define __ERRORNIETO_H__

#include "cv.h"      
#include "highgui.h" 
#include "cxcore.h" 

/** This is the data structure passed to the Levenberg-Marquardt procedure */
struct data_struct
{
	cv::Mat &LSS;		// Line segments vectors (3xN)	
	cv::Mat &Lengths;	// Length of line segments (NxN) (diagonal)
	cv::Mat &midPoints;	// Mid points (c=(a+b)/2) (3xN)	

	cv::Mat &K;		// Camera calibration matrix (3x3)

	data_struct (cv::Mat &_LSS, cv::Mat &_Lengths, cv::Mat &_midPoints, cv::Mat &_K): 
		LSS(_LSS), Lengths(_Lengths), midPoints(_midPoints), K(_K)
	{
	}
};

/** This is the function that computes the distance between a given vanishing point and a line segment using the metric
	proposed by Marcos Nieto. Reference:
	M. Nieto and L. Salgado, "Non-linear optimization for robust estimation of vanishing points," in IEEE Proc. Int.
	Conf. on Image Processing (ICIP2010), pp. 49-52, 2010.
 */
float distanceNieto( cv::Mat &vanishingPoint, cv::Mat &lineSegment, float lengthLineSegment, cv::Mat &midPoint );
/** This function contains the procedure of estimating a vanishing point given a set of line segments using the method
	proposed by Marcos Nieto.*/
void evaluateNieto( const double *param, int m_dat, const void *data, double *fvec, int *info);


#endif // __ERRORNIETO_H__
