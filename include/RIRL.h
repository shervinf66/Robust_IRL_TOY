#ifndef RIRL_H
#define RIRL_H

#include "data.h"
#include "process.h"
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

class RIRL
{
private:
    vector<vector<int>> generateListOfAllSA(Data &data, Process &pr);
    void constructAllT(Data &data, Process &pr, int trajectoryLenght);
    double calcPrT(Data &data, Process &pr, vector<vector<int>> t);
    double calcPrTgivenW(Data & data, Process &pr, vector<vector<int>> t, int tIndex, vector<Sample> w);

public:
    RIRL();
    ~RIRL();
    vector<double> calcFeatureExpectationLeft(Data &data, Process &pr, vector<double> weights); // This function will return a feature expection for each feature
    vector<double> exponentiatedGradient(Data &data, Process &pr, vector<double> y,
                                         vector<double> w, double c, double err); // the two above together are M-Step
    vector<double> eStep(Data & data, Process &pr, vector<vector<Sample> > allW);// E-step
    void printNestedVector(vector<vector<int>> v); // for debugging
};

#endif // RIRL_H
