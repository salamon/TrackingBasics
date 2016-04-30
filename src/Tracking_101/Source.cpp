#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

//App Parameters
int threshold_value = 12;
int opening_size = 2;
int closing_size = 6;

//App Controls
RNG rng;
vector<Point> last_points;
vector<Scalar> colors;

double eucDist(Point p1, Point p2)
{
	double x = p1.x - p2.x;
	double y = p1.y - p2.y;
	double dist = pow(x, 2) + pow(y, 2);
	dist = sqrt(dist);

	return dist;
}

void genColors(int n)
{
	for (int k = 0; k < n; k++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		colors.push_back(color);
	}
	
	randShuffle(colors); //if random
}


int main(int argc, char** argv)
{
	//Display Windows
	namedWindow("Current Frame", WINDOW_AUTOSIZE); 
	namedWindow("Result", WINDOW_AUTOSIZE);
	namedWindow("Tracking", WINDOW_AUTOSIZE);
	
	//Read all Frames
	String path("../../ds/vhl/*.png"); 
	vector<String> files;
	vector<Mat> imgs;
	glob(path, files, true);
	for (size_t k = 0; k < files.size(); k++)
	{
		Mat im = imread(files[k]);
		if (im.empty()) continue; 
		imgs.push_back(im);
	}

	//Trackbars to adjust parameters (without callback to apply only in the next iteration)
	//createTrackbar("Threshold", "Current Frame", &threshold_value, 255);
	//createTrackbar("Open", "Current Frame", &opening_size, 10);
	//createTrackbar("Closing", "Current Frame", &closing_size, 10);

	//Initializing Controls
	genColors(50);


	Mat bg = imgs[0];
	Mat acum = bg.clone();
	for (int i = 1; i < imgs.size(); i++)
	{
		//for each frame
		Mat im = imgs[i];
		imshow("Current Frame", im);


		//background diff
		Mat bin, diff, fin;
		diff = abs(im - bg);
		cvtColor(diff, bin, CV_BGR2GRAY);
		threshold(bin, bin, threshold_value, 255, THRESH_BINARY);
	
		
		//morph ops
		Mat element_op = getStructuringElement(MORPH_CROSS, Size(opening_size + 1, opening_size + 1), Point(opening_size, opening_size));
		erode(bin, bin, element_op);
		dilate(bin, bin, element_op);
		
		Mat element_cl = getStructuringElement(MORPH_CROSS, Size(closing_size + 1, closing_size + 1), Point(closing_size, closing_size));
		dilate(bin, bin, element_cl);
		erode(bin, bin, element_cl);


		//blob detection
		vector<KeyPoint> keypoints;
		SimpleBlobDetector::Params params;
		params.minDistBetweenBlobs = 1.0f;
		params.filterByInertia = false;
		params.filterByCircularity = false;
		params.filterByConvexity = false;
		params.filterByColor = false;
		params.filterByArea = true;
		params.minArea = 200.0;
		params.maxArea = 4000.0;
		Ptr<FeatureDetector> blob_detector = SimpleBlobDetector::create(params);
		blob_detector->detect(bin, keypoints);
		//Mat kp = bin;
		//drawKeypoints(bin, keypoints, kp, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		

		//tracking
		double max_mov = 0.35;
		cvtColor(bin, fin, CV_GRAY2BGR);
		for (int k = 0; k < keypoints.size(); k++)
		{
			Point p = Point(keypoints[k].pt.x, keypoints[k].pt.y);
			Point kp_ini = Point(p.x - keypoints[k].size / 2, p.y - keypoints[k].size / 2);
			Point kp_end = Point(p.x + keypoints[k].size / 2, p.y + keypoints[k].size / 2);

			bool tracked = false;
			for (int z = 0; z < last_points.size(); z++)
			{
				if ((eucDist(p, last_points[z]) < (keypoints[k].size * max_mov)))
				{
					rectangle(fin, kp_ini, kp_end, colors[z], 2, 8, 0);
					circle(acum, p, 2, colors[z], 2, 8, 0);
					last_points[z] = p;
					tracked = true;
				}
			}

			if (!tracked) last_points.push_back(p);
		}


		//Show Res (press any key to move on)
		imshow("Result", fin);
		imshow("Tracking", acum);
		waitKey(0);
	}

	return 0;
}
