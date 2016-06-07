#ifndef RIRL_H
#define RIRL_H

#include "data.h"
#include "process.h"
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;
struct Node{
public:
    bool isFakeNode;
    bool isInitalState;
    int previousState;
    int previousAction;
    int state;
    int action;
};

class RIRL
{
private:
    vector<vector<int>> generateListOfAllSA(Data &data, Process &pr);
    void constructAllT(Data &data, Process &pr, int trajectoryLenght);
    double calcPrT(Data &data, Process &pr, vector<vector<int>> t);
    double calcPrTgivenW(Data & data, Process &pr, vector<vector<int>> t, int tIndex, vector<Sample> w);
    vector<Node> returnChildren(Data &data, Process &pr, Node node);

public:
    RIRL();
    ~RIRL();
    vector<double> calcFeatureExpectationLeft(Data &data, Process &pr, vector<double> weights); // This function will return a feature expection for each feature
    vector<double> exponentiatedGradient(Data &data, Process &pr, vector<double> y,
                                         vector<double> w, double c, double err); // the two above together are M-Step
    vector<double> eStep(Data & data, Process &pr, vector<vector<Sample> > allW);// E-step

    //normalizerVectorForPrTgivenW and featureExpectationVector must initialize to zero
    // before calling eStepRecursiveUtil in the main e_step function
    void eStepRecursiveUtil(Data &data, Process &pr, vector<Sample> w, Node node, int level
                                  , double prT, double prWgivenT, double normalizerForObsModel,
                                  double &normalizerVectorForPrTgivenW, vector<double> &featureVector
                            , vector<double> &featureExpectationVector); // E-step recursive call

    void initializePolicy(Data & data, Process &pr);
    void printNestedVector(vector<vector<int>> v); // for debugging
    void printVector(vector<double> v); // for debugging
};

#endif // RIRL_H
