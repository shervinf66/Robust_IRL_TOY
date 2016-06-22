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
    vector<int> previousState;
    int previousAction;
    vector<int> state;
    int action;
};

class RIRL
{
private:
    vector<Node> returnChildren(Data &data, Process &pr, Node node);
    void constructLearnerPolicy(Data &data, map<vector<int>,map<int,double>> Q_value);
    double obsNormlizer(Data &data, Sample s);
public:
    RIRL();
    ~RIRL();
    //M-step
    vector<double> calcFeatureExpectationLeft(Data &data, Process &pr, vector<double> weights); //This function will return a feature expection for each feature
    vector<double> exponentiatedGradient(Data &data, Process &pr, vector<double> y,
                                         vector<double> w, double c, double err); // the two above together are M-Step
    vector<double> gradientDescent(Data &data, Process &pr, vector<double> y,
                                         vector<double> w, double c, double err); // the two above together are M-Step

    //normalizerVectorForPrTgivenW and featureExpectationVector must initialize to zero
    // before calling eStepRecursiveUtil in the main e_step function
    void eStepRecursiveUtil(Data &data, Process &pr, vector<Sample> w, Node node, int level
                            , double prT, double prWgivenT, double normalizerForObsModel,
                            double &normalizerVectorForPrTgivenW, vector<double> &featureVector
                            , vector<double> &featureExpectationVector, int &tCounter, vector<vector<int> > &t, bool deterministicObs); // E-step recursive call

    vector<double> eStep(Data &data, Process &pr, vector<Sample> w, bool deterministicObs); // E-step main function

    void initializePolicy(Data & data, Process &pr, vector<double> weights);
    void printNestedVector(vector<vector<int>> v); // for debugging
    void printVector(vector<double> v); // for debugging
    double clalObsPrUsingClusteringObsModel(Data & data, Process &pr, Sample s);
};

#endif // RIRL_H
