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
    //    y.push_back(1.0);
    //    y.push_back(0.0);

    //    vector<double> w_initial;
    //    w_initial.push_back(1.0);
    //    w_initial.push_back(1.0);

    //    vector<double> w;
    //    w = rirl.exponentiatedGradient(data,pr,y,w_initial,1.0,0.1);
    //    cout << w.at(0) << endl;
    //    cout << w.at(1) << endl;

    pr.generateTrajectories(data, 1, false);
    cout << data.getObsList().at(0).size() << endl;
    cout << "Loading ObsModel!" << endl;
    pr.loadObsModel(data);
    cout << "ObsModel loaded!" << endl;

    cout << "Initialization!" << endl;
    //initialize policy
    rirl.initializePolicy(data,pr);

    // initalize weights
    vector<double> weights(2,1.0);
    rirl.printVector(weights);
    cout << "Initialization done!" << endl;

    vector<vector<Sample > > obsList = data.getObsList();
    vector<double> expertFeatureVector;
    int counter = 0;
    do{
        expertFeatureVector = rirl.eStep(data,pr,obsList);
        rirl.printVector(expertFeatureVector);
        weights = rirl.exponentiatedGradient(data,pr,expertFeatureVector,weights,1.0,0.01);
        rirl.printVector(weights);
        counter++;
        cout << "***********************************************************" << endl;
        if(counter == 50){
            break;
        }
    }while(true);

    cout << "Done!" << endl;

    return 0;
}
