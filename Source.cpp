

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace cv;
using namespace cv::face;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, std::map<int, string>& labelsInfo, char separator = ';') {
	ifstream csv(filename.c_str());
	if (!csv) CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
	string line, path, classlabel, info;
	while (getline(csv, line)) {
		stringstream liness(line);
		path.clear(); classlabel.clear(); info.clear();
		getline(liness, path, separator);
		getline(liness, classlabel, separator);
		getline(liness, info, separator);
		if (!path.empty() && !classlabel.empty()) {
			cout << "Processing " << path << endl;
			int label = atoi(classlabel.c_str());
			if (!info.empty())
				labelsInfo.insert(std::make_pair(label, info));
			// 'path' can be file, dir or wildcard path
			String root(path.c_str());
			vector<String> files;
			glob(root, files, true);
			for (vector<String>::const_iterator f = files.begin(); f != files.end(); ++f) {
				cout << "\t" << *f << endl;
				Mat img = imread(*f, IMREAD_GRAYSCALE);
				static int w = -1, h = -1;
				static bool showSmallSizeWarning = true;
				if (w>0 && h>0 && (w != img.cols || h != img.rows)) cout << "\t* Warning: images should be of the same size!" << endl;
				if (showSmallSizeWarning && (img.cols<50 || img.rows<50)) {
					cout << "* Warning: for better results images should be not smaller than 50x50!" << endl;
					showSmallSizeWarning = false;
				}
				images.push_back(img);
				labels.push_back(label);
			}
		}
	}
}
static void read_csv2(const string& filename, vector<Mat>& testimages, vector<int>& testlabels, std::map<int, string>& testlabelsInfo, char separator = ';') {
	ifstream csv(filename.c_str());
	if (!csv) CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
	string line, path, classlabel, info;
	while (getline(csv, line)) {
		stringstream liness(line);
		path.clear(); classlabel.clear(); info.clear();
		getline(liness, path, separator);
		getline(liness, classlabel, separator);
		getline(liness, info, separator);
		if (!path.empty() && !classlabel.empty()) {
			cout << "Processing " << path << endl;
			int label = atoi(classlabel.c_str());
			if (!info.empty())
				testlabelsInfo.insert(std::make_pair(label, info));
			// 'path' can be file, dir or wildcard path
			String root(path.c_str());
			vector<String> files;
			glob(root, files, true);
			for (vector<String>::const_iterator f = files.begin(); f != files.end(); ++f) {
				cout << "\t" << *f << endl;
				Mat img = imread(*f, IMREAD_GRAYSCALE);
				static int w = -1, h = -1;
				static bool showSmallSizeWarning = true;
				if (w>0 && h>0 && (w != img.cols || h != img.rows)) cout << "\t* Warning: images should be of the same size!" << endl;
				if (showSmallSizeWarning && (img.cols<50 || img.rows<50)) {
					cout << "* Warning: for better results images should be not smaller than 50x50!" << endl;
					showSmallSizeWarning = false;
				}
				testimages.push_back(img);
				testlabels.push_back(label);
			}
		}
	}
}


int main(int argc, const char *argv[]) {
	float accuracy = 0, howmanytimes = 1;;
	
		Mat klatka;
		VideoCapture videocap;
		videocap.open(0);
		if (!videocap.isOpened())
		{
			printf("Error: Cannot open video stream from camera\n");
			return 1;
		} //czy kamerka siê otworzy³a
		while (1) {
			videocap.read(klatka);
			//videocap >> klatka;
			imshow("Display window", klatka);
			waitKey(1);
			//}
			system("pause");
			return 0;
		}
	
		// œcie¿ka  pliku csv
		string fn_csv = string(argv[1]);
		// wektory do przechowywania obrazów i etykiet
		vector<Mat> images;
		vector<int> labels;
		std::map<int, string> labelsInfo; // Ta funkacja uczy model etykiet
		try {
			read_csv(fn_csv, images, labels, labelsInfo);
		}
		catch (cv::Exception& e) {
			cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
			// nothing more we can do
			exit(1);
		}
		//je¿eli za ma³o obrazków  w pliku
		if (images.size() <= 1) {
			string error_message = "____";
			CV_Error(Error::StsError, error_message);
		}

		/****************************************************************************/
		string saveModelPath = "zapisdanych.txt";
		int nlabels = (int)labels.size();
		Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer();
		model->load(saveModelPath);
		for (int g = 0; g < nlabels; g++)
			model->setLabelInfo(g, labelsInfo[g]);
		model->train(images, labels);
		/****************************************************************************/
		map<int, string> testlabelsInfo;
		vector<Mat> testimages;
		vector<int> testlabels;
		string fn_csv2 = string(argv[2]);
		try {
			read_csv2(fn_csv2, testimages, testlabels, testlabelsInfo);
		}
		catch (Exception& e) {
			cerr << "blad przy otwieraniu pliku \"" << fn_csv2 << "\". powod " << e.msg << endl;
			// nothing more we can do
			exit(1);
		}
		//je¿eli za ma³o obrazków  w pliku
		if (testimages.size() <= 1) {
			string error_message = "____";
			CV_Error(Error::StsError, error_message);
		}
		/*// Ta funkacja uczy model etykiet*/
		for (int g = 0; g < nlabels; g++)
			model->setLabelInfo(g, testlabelsInfo[g]);

		for (int i = 0; i < testimages.size();i++) {
			howmanytimes++;
			Mat testSample = testimages[i];
			int testLabel = testlabels[i]; // etykieta zdjêcia testowego

	//imshow("Display window", testSample);


	//cout << "zapis trenowanego modelu do " << saveModelPath << endl;
			model->save(saveModelPath);



			// testowy obrazek:
			int predictedLabel = model->predict(testSample);
			if (predictedLabel == testLabel)
				accuracy++;

		}
		accuracy = 100 * accuracy / howmanytimes;
		cout << endl << endl << "dokladnosc rozpoznawania twarzy metoda eigenface to  " << accuracy << "%" << endl;


		system("pause");
		waitKey(0);
		return 0;
	}
