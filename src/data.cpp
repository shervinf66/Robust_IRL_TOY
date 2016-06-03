#include "data.h"

Data::Data()
{
    listOfStates = {10,11,12,13,20,21,22,23,30,31,32,33};
    listOfActions = {-10,-1,0,1,10};
    low->push_back(-1000.0);
    low->push_back(-1000.0);

    high->push_back(1000.0);
    high->push_back(1000.0);

    for (double i = 0 ; i < _numberOfSamples ; i++){
        timeChunk.push_back(i);
    }

    for (double i = 0 ; i < listOfStates.size() ; i++){
        int state = listOfStates.at(i);
        if (state == 10){
            stateInitialPriorities[state] = 1.0;
        }else{
            stateInitialPriorities[state] = 0.0;
        }
    }
}

Data::~Data()
{

}



