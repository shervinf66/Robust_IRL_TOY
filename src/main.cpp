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

    // saving the deterministic obs model there is bug dont run it with
    // rirl at the same time save model once and then run the program
    //        pr.saveDeterministicObsModel(data);

    // new approach (clustering) for obsModel
    pr.saveClusteringObsModel(data, 5000);


    int lt = 10;

    // i may need to seprate the sample size form trajectory lenght
    pr.generateTrajectories(data, lt, false);

    cout << "Loading ObsModel!" << endl;
    //    pr.loadObsModel(data);
    //    pr.loadDeterministicObsModel(data);
    // load the clustering obsModel
    pr.loadClusteringObsModel(data);
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
        //        pr.printPolicy(data.getLearnerPolicy());
        cout << "***********************************************************" << endl;
    }while(true);
    pr.printPolicy(data.getLearnerPolicy());
    cout << "Done!" << endl;

    return 0;
}
