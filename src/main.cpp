#include "data.h"
#include "process.h"
#include "RIRL.h"
#include "DETree.h"

#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    srand(1366); // For reproducable results

    Data data = Data();
    Process pr;
    RIRL rirl;

    vector<double> y;
    //    y.push_back(20);
    //    y.push_back(0);
    //    y.push_back(0);
    //    y.push_back(0);
    //    y.push_back(60);
    //    y.push_back(0);
    //    y.push_back(0);
    //    y.push_back(0);
    //    y.push_back(0);
    //    y.push_back(0);
    //    y.push_back(1);

    //    vector<double> w;
    //    for(int i = 0 ; i < data.getNumberOfFeatures() ; i++){
    //        double f = (double)rand() / RAND_MAX;
    //        f = -1.0 + f * (1.0 + 1.0);
    //        w.push_back(f);
    //    }

    //    y ={3.91108 , 0.359476 , 1.53784 , 0 , 4.59918};
    //    //    w = rirl.exponentiatedGradient(data,pr,y,w,1.0,0.01);
    //    w = rirl.gradientDescent(data,pr,y,w,1,1);
    //    rirl.printVector(w);
    //    pr.printPolicy(data.getLearnerPolicy());
    //    cout << "--------------------------------------------" << endl;
    //    y.at(0) = (0.8);
    //    y.at(1) = (0.5);
    //    w = rirl.exponentiatedGradient(data,pr,y,w,1,0.1);
    //    cout << w.at(0) << endl;
    //    cout << w.at(1) << endl;
    //    cout << "--------------------------------------------" << endl;
    //    y.at(0) = 2;
    //    y.at(1) = 2;
    //    w = rirl.exponentiatedGradient(data,pr,y,w,0.33,0.1);
    //    cout << w.at(0) << endl;
    //    cout << w.at(1) << endl;
    //    cout << "--------------------------------------------" << endl;

    // saving the deterministic obs model there is bug dont run it with
    // rirl at the same time save model once and then run the program
    //    pr.saveDeterministicObsModel(data);
    int lt = 10;

    // i may need to seprate the sample size form trajectory lenght
    pr.generateTrajectories(data, lt, false);
    //    for(int i = 0 ; i < data.getNumberOfSamples() ; i++){
    //        rirl.printNestedVector(data.getDiscreteTrajectories().at(i));
    //    }
    //    pr.printPolicy(data.getExpertPolicy());
    //    vector<vector<int>> expertT = {{0,0,1},{1,0,1},{2,0,1},{3,0,1},{4,0,0},{4,1,1},{3,1,1},{2,1,1},{1,1,1},{0,1,0},{0,0,1},{1,0,1},{2,0,1},{3,0,1},{4,0,0},{4,1,1},{3,1,1},{2,1,1},{1,1,1},{0,1,0}};

    //    vector<double> fv = pr.getFeatureVectorOfT(data,expertT);
    //    rirl.printVector(fv);
    cout << "Loading ObsModel!" << endl;
    // I will not use dtree I will simply remove noise and then use a deteministic obs model
    pr.loadObsModel(data);
    //    pr.loaddeterministicObsModel(data);

    cout << "ObsModel loaded!" << endl;

    cout << "Initialization!" << endl;


    // initalize weights
    vector<double> weights;
    for(int i = 0 ; i < data.getNumberOfFeatures() ; i++){
        double f = (double)rand() / RAND_MAX;
        weights.push_back(f);
    }
    //initialize policy
    rirl.initializePolicy(data,pr,weights);

    //    weights.push_back(1);
    //    weights.push_back(1);
    //    weights.push_back(1);
    //    weights.push_back(1);
    //    weights.push_back(1);
    //    weights.push_back(1);
    //    weights.push_back(1);
    //    weights.push_back(1);
    //    weights.push_back(1);
    //    weights.push_back(1);
    //    weights.push_back(1);
    //    weights.push_back(1);
    rirl.printVector(weights);
    cout << "Initialization done!" << endl;
    cout << "***********************************************************" << endl;

    //    vector<vector<Sample > > obsList = data.getObsList();
    vector<Sample > obsList = data.getFlatObsList();
    vector<double> expertFeatureVector;
    vector<double> PreviousExpertFeatureVector(data.getNumberOfFeatures(),0.0);
    int counter = 0;
    do{
        cout << "Iteration: " << counter << endl;
        expertFeatureVector = rirl.eStep(data,pr,obsList,false);
        //        expertFeatureVector.at(expertFeatureVector.size()-2)=0;
        cout << "expertFeatureVector: ";
        rirl.printVector(expertFeatureVector);
        //        weights = rirl.exponentiatedGradient(data,pr,expertFeatureVector,weights,1.0,0.02);
        weights = rirl.gradientDescent(data,pr,expertFeatureVector,weights,1.0,0.1);
        cout << "weights: ";
        rirl.printVector(weights);
        counter++;


        for (int i = 0 ; i < int(PreviousExpertFeatureVector.size()) ; i++){
            PreviousExpertFeatureVector.at(i) = PreviousExpertFeatureVector.at(i) - expertFeatureVector.at(i);
        }
        double diff = pr.l2norm(PreviousExpertFeatureVector);
        cout << "Change in expertFeatureVector: " << diff << endl;
        if(diff < 0.0001){
            break;
        }
        PreviousExpertFeatureVector = expertFeatureVector;
        pr.printPolicy(data.getLearnerPolicy());
        cout << "***********************************************************" << endl;
    }while(true);
    pr.printPolicy(data.getLearnerPolicy());
    cout << "Done!" << endl;

    return 0;
}
