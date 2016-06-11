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

    //        vector<double> y;
    //        y.push_back(0);
    //        y.push_back(1);

    //        vector<double> w;
    //        w.push_back(1.0);
    //        w.push_back(1.0);


    //        w = rirl.exponentiatedGradient(data,pr,y,w,0.33,0.1);
    //        cout << w.at(0) << endl;
    //        cout << w.at(1) << endl;
    //        cout << "--------------------------------------------" << endl;
    //        y.at(0) = (0.8);
    //        y.at(1) = (0.5);
    //        w = rirl.exponentiatedGradient(data,pr,y,w,1,0.1);
    //        cout << w.at(0) << endl;
    //        cout << w.at(1) << endl;
    //        cout << "--------------------------------------------" << endl;
    //        y.at(0) = 2;
    //        y.at(1) = 2;
    //        w = rirl.exponentiatedGradient(data,pr,y,w,0.33,0.1);
    //        cout << w.at(0) << endl;
    //        cout << w.at(1) << endl;
    //        cout << "--------------------------------------------" << endl;

    int nT = 1;
    pr.generateTrajectories(data, nT, false);
    for(int i = 0 ; i < nT ; i++){
        rirl.printNestedVector(data.getDiscreteTrajectories().at(i));
    }


    cout << "Loading ObsModel!" << endl;
    pr.loadObsModel(data);
    cout << "ObsModel loaded!" << endl;

    cout << "Initialization!" << endl;
    //initialize policy
    rirl.initializePolicy(data,pr);

    // initalize weights
    vector<double> weights;
    weights.push_back(1);
    weights.push_back(2);
    rirl.printVector(weights);
    cout << "Initialization done!" << endl;

    vector<vector<Sample > > obsList = data.getObsList();
    vector<double> expertFeatureVector;
    int counter = 0;
    do{
        cout << "Iteration: " << counter << endl;
        expertFeatureVector = rirl.eStep(data,pr,obsList);
        rirl.printVector(expertFeatureVector);
        weights = rirl.exponentiatedGradient(data,pr,expertFeatureVector,weights,1.0,0.000001);
        rirl.printVector(weights);
        counter++;
        cout << "***********************************************************" << endl;
        //        if(counter == 50){
        //            break;
        //        }
    }while(true);

    cout << "Done!" << endl;

    return 0;
}
