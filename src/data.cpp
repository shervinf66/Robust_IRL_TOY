#include "data.h"

Data::Data()
{
    noise = 0.0;
     for (double i = 0 ; i < _numberOfRawStates; i++){
         listOfRawStates.push_back(i); //changed
         for (double j = 0 ; j < 2; j++){ //{ j = 0 ==> est } , { j = 1 ==> west }
             vector<int> tuple; //<rawstate,orientation>
             tuple.push_back(i);
             tuple.push_back(j);
             listOfStates.push_back(tuple);
         }
     }

    listOfActions = {0,1}; //changed // get rid of this action to prevent the some problems -1,
    expertWeights = {1.0,-2.0,-3.0,-4.0,1.5};//{1.0,-2.0,-3.0,-4.0,-5.0,-6.0,-7.0,-8.0,-9.0,-10.0,-2.0,-2.0,1.5};//{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1}; //new
    low->push_back(-1000.0);
    low->push_back(-1000.0);

    high->push_back(1000.0);
    high->push_back(1000.0);

    for (double i = 0 ; i < _numberOfSamples ; i++){
        timeChunk.push_back(i);
    }

    for (double i = 0 ; i < listOfRawStates.size() ; i++){
        vector<int> state = listOfStates.at(i);
        if (state.at(0) == 0 && state.at(1) == 0){ //initial state at 0 facing 0 (east)
            stateInitialPriorities[state] = 1.0;
        }else{
            stateInitialPriorities[state] = 0.0;
        }
    }
}

Data::~Data()
{

}



