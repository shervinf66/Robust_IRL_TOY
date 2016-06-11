#include "process.h"
#include "data.h"
#include "linear.h"
#include "Sample.h"
#include "DETree.h"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;

Process::Process()
{

}

Process::~Process()
{

}

vector<vector<double> > Process::getContinuousSamples(Data &data,vector<double> previousPoint,vector<double> currentPoint){
    int numberOfSamples = data.getNumberOfSamples();

    double startPointX = previousPoint.at(0);
    double startPointY = previousPoint.at(1);

    double endPointX = currentPoint.at(0);
    double endPointY = currentPoint.at(1);

    double deltaX = abs((startPointX - endPointX)) / ((double) (numberOfSamples));
    double deltaY = abs((startPointY - endPointY)) / ((double) (numberOfSamples));

    double xSample = startPointX;
    double ySample = startPointY;

    vector<vector<double> > samplesList;

    for (int i = 0 ; i < numberOfSamples ; i++){
        vector<double> sample;
        sample.push_back(xSample);
        sample.push_back(ySample);

        samplesList.push_back(sample);

        xSample = xSample + deltaX;

        if(deltaX != 0.0){
            ySample = ((startPointY - endPointY) / (startPointX - endPointX)) * xSample
                    + (startPointY - ((startPointY - endPointY) /
                                      (startPointX - endPointX)) * startPointX);
        }else if(deltaX == 0.0 && deltaY != 0.0){
            ySample = ySample + deltaY;
        }else{
            ySample = ySample;
        }
    }

    return samplesList;
}

double Process::getDistanceToPowerTwo(vector <double> sample){
    return (pow ((sample.at(0) - 15), 2) + pow ((sample.at(1) - 15), 2));
}

vector<double> Process::getObsSampleList(Data &data,vector<double> previousPoint,vector<double> currentPoint){
    vector< vector<double> > continuousSampleList = getContinuousSamples(data, previousPoint, currentPoint);

    vector<double> obsList;

    double sigma = data.getSigma();
    double mean = data.getMean();
    double obs;
    for (int i = 0 ; i < (int)continuousSampleList.size() ; i++){
        vector<double> sample = continuousSampleList.at(i);

        double r2 = getDistanceToPowerTwo(sample);
        double p = data.getP(); // presure in intensity formula

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        normal_distribution<double> distribution(mean,sigma);
        double noise = distribution(generator);

        //        cout << noise << endl;

        obs = (p / (4 * M_PI * r2)) + noise;

        obsList.push_back(obs);
    }

    return obsList;
}

void Process::buildAContinuousTrajectoryAndDiscreteTrajectory(Data &data, bool forTrainingObs){
    vector< vector<double> > aContinusTrajectory;
    vector< vector<int> > aDiscreteTrajectory;
    vector<int> initialStateActionPair;
    vector<double> initialPoint;
    vector<double> previousPoint;
    vector<double> ObsSampleList;

    int currentState;
    int initialAction;
    int action;
    int initialState = 10; // modified for debug change back 32 to 10

    if(forTrainingObs == true){
        initialAction = randomPolicy(data);
    }else{
        initialAction = rationalPolicy(initialState);
    }

    initialStateActionPair.push_back(initialState);
    initialStateActionPair.push_back(initialAction);

    aDiscreteTrajectory.push_back(initialStateActionPair);

    initialPoint = getRandomPointInState(data, initialState);
    previousPoint = initialPoint;

    aContinusTrajectory.push_back(initialPoint);


    currentState = initialState;
    action = initialAction;

    bool done = false;
    int lastPointIndex = 0;

    while(!done){
        vector<int> stateActionPair;
        vector<double> point;

        int nextState = transitionFunction(data, currentState, action);
        if(isBlockedlState(nextState)){
            continue;
        }else{
            if(isTerminalState(nextState)){
                action = 0;
                done = true;
            }else{
                if(forTrainingObs == true){
                    action = randomPolicy(data);
                }else{
                    action = rationalPolicy(nextState);
                }
            }
        }

        if(currentState == nextState){
            point = previousPoint;
        }else{
            point = getRandomPointInState(data, nextState);
        }

        stateActionPair.push_back(nextState);
        stateActionPair.push_back(action);
        aDiscreteTrajectory.push_back(stateActionPair);

        vector<double> obs;
        obs = getObsSampleList(data, aContinusTrajectory.at(lastPointIndex), point);
        ObsSampleList.insert(ObsSampleList.end(), obs.begin(), obs.end());
        lastPointIndex = lastPointIndex + 1;
        aContinusTrajectory.push_back(point);

        currentState = nextState;
        if(done){
            obs = getObsSampleList(data,point, point);
            ObsSampleList.insert(ObsSampleList.end(), obs.begin(), obs.end());
        }
    }
    data.addAContinuousTrajectory(aContinusTrajectory);
    data.addADiscreteTrajectory(aDiscreteTrajectory);
    data.updateNormalizerFactorForP((double)aDiscreteTrajectory.size() - 1);
    data.addObsSample(ObsSampleList);
}

int Process::calcIntensityChangeFlag(Data &data, double intensityDifference){
    int intensityChangeThreshold = data.getIntensityChangeThreshold();
    int intensityChangeFlag;

    if(intensityDifference > 0 && abs(intensityDifference) > intensityChangeThreshold){
        intensityChangeFlag = 1;
    }else if(intensityDifference < 0 && abs(intensityDifference) > intensityChangeThreshold){
        intensityChangeFlag = -1;
    }else{
        intensityChangeFlag = 0;
    }

    return intensityChangeFlag;
}

void Process::bildObsList(Data &data, bool forTrainingObs){
    int numberOfSamples = data.getNumberOfSamples();

    vector<vector<double> > obsSampleList = data.getObsSampleList();
    vector< vector<vector<int> > > discreteTrajectories = data.getDiscreteTrajectories();
    double normalizerForP = data.getNormalizerFactorForP();

    for (int i = 0 ; i < (int)obsSampleList.size() ; i++){
        vector<double> obsSampleForOneTrajectory = obsSampleList.at(i);
        vector<vector<int> > aDiscreteTrajectory = discreteTrajectories.at(i);

        // New way using regression. that is continuous
        vector<Sample > obsForOneTrajectory;
        for (int j = 0 ; j < (int)obsSampleForOneTrajectory.size() ; j = j + numberOfSamples){
            vector<double> IntensityChunk;
            for(int k = 0 ; k < numberOfSamples ; k++){
                IntensityChunk.push_back(obsSampleForOneTrajectory.at(j + k));
            }
            int stateAcionIndex = j / numberOfSamples;
            vector<double> timeChunk = data.getTimeChunk();
            Sample obsTuple;
            Maths::Regression::Linear linearRegression(numberOfSamples, timeChunk, IntensityChunk);
            double slope =linearRegression.getSlope();
            obsTuple.values.push_back(slope);
            double intercept =linearRegression.getIntercept();
            obsTuple.values.push_back(intercept);
            if(forTrainingObs){
                obsTuple.values.push_back(aDiscreteTrajectory.at(stateAcionIndex).at(0)); // adding state
                obsTuple.values.push_back(aDiscreteTrajectory.at(stateAcionIndex).at(1)); // adding action
            }
            //            else{
            //                obsTuple.values.push_back(-1); // adding dummy state
            //                obsTuple.values.push_back(-1); // adding dummy action
            //            }

            obsTuple.p = 1.0 / normalizerForP; //uniform
            obsForOneTrajectory.push_back(obsTuple);
            data.updateFlatObsList(obsTuple);
        }
        data.addObs(obsForOneTrajectory);
    }
}

int Process::classifyObs(Data &data, double avg){
    int cl = 0;
    int numberOfDisObs = data.getNumberOfDisObs();
    double start = data.getMinIntensity();
    double end = start + data.getObsStepSize();

    if (avg < data.getMinIntensity()){
        avg = data.getMinIntensity();
    }
    if (avg > data.getMaxIntensity()){
        avg = data.getMaxIntensity();
    }
    for (int i = 0 ; i <= numberOfDisObs ; i++){
        if (avg >= start && avg <= end){
            break;
        }else{
            cl++;
            start = start + data.getObsStepSize();
            end = end + data.getObsStepSize();
        }
    }
    return cl;
}

vector<double> Process::getRandomPointInState(Data &data, int state){
    vector<double> result;
    vector<double> centerBoxBoundaries = calcCenterBoxBoundaries(data, state);

    double x = getRandomDoubleNumber(centerBoxBoundaries.at(0), centerBoxBoundaries.at(1));
    double y = getRandomDoubleNumber(centerBoxBoundaries.at(2), centerBoxBoundaries.at(3));

    result.push_back(x);
    result.push_back(y);

    return result;
}

vector<int> Process::calcStateBoundaries(Data &data, int state){
    vector<int> result;

    int stepSize = data.getStepSize();
    int remainder = state % stepSize;
    int quotient = state / stepSize;

    int stateLowerX = remainder * stepSize;
    int stateUpperX = remainder  * stepSize + stepSize;
    int stateLowerY = quotient * stepSize - stepSize;
    int stateUpperY = quotient * stepSize;

    result.push_back(stateLowerX);
    result.push_back(stateUpperX);
    result.push_back(stateLowerY);
    result.push_back(stateUpperY);

    return result;
}

vector<double> Process::calcCenterBoxBoundaries(Data &data, int state){
    vector<double> result;
    vector<int> stateBoundaries = calcStateBoundaries(data,state);

    int stateLowerX = stateBoundaries.at(0);
    int stateUpperX = stateBoundaries.at(1);
    int stateLowerY = stateBoundaries.at(2);
    int stateUpperY = stateBoundaries.at(3);
    int centerBoxDim = data.getCenterBoxDim();

    double centerX = (double)(stateLowerX + stateUpperX) / 2.0;
    double centerY = (double)(stateLowerY + stateUpperY) / 2.0;

    double centerBoxLowerX = centerX - (double)(centerBoxDim) / 2.0;
    double centerBoxUpperX = centerX + (double)(centerBoxDim) / 2.0;
    double centerBoxLowerY = centerY - (double)(centerBoxDim) / 2.0;
    double centerBoxUpperY = centerY + (double)(centerBoxDim) / 2.0;

    result.push_back(centerBoxLowerX);
    result.push_back(centerBoxUpperX);
    result.push_back(centerBoxLowerY);
    result.push_back(centerBoxUpperY);

    return result;
}

int Process::rationalPolicy(int state){
    int rationalAction;

    if (state == 10){
        rationalAction = 1;
    }else if (state == 11){
        rationalAction = 1;
    }else if (state == 12){
        rationalAction = 10;
    }else if (state == 13){
        rationalAction = -1;
    }else if (state == 20){
        rationalAction = 10;
    }else if (state == 22){
        rationalAction = 10;
    }else if (state == 23){
        rationalAction = 0;
    }else if (state == 30){
        rationalAction = 1;
    }else if (state == 31){
        rationalAction = 1;
    }else if (state == 32){
        rationalAction = 1;
    }

    return rationalAction;
}

int Process::randomPolicy(Data &data){
    vector<int> listOfActions = data.getListOfActions();
    int randomAction = listOfActions.at(getRandomIntNumber(listOfActions.size()));
    return randomAction;
}
// somrthing is wrong here
double Process::probablityOfNextStateGivenCurrentStateAction(Data &data,int nextState, int currentState, int currentAction){
    double stochasticity = data.getStochasticity();
    int idealNextState = returnNextState(data,currentState,currentAction);
    double pr = 0.0;
    if(isBlockedlState(nextState)){
        pr = 0.0;
    }else if (nextState == idealNextState){
        pr = 1.0 - stochasticity;
    }else if(abs(nextState - currentState) == 10 || abs(nextState - currentState) == 1
             || abs(nextState - currentState) == 0){

        if(currentState == 10 || currentState == 11 || currentState == 13
                || currentState == 20 || currentState == 33 || currentState == 30 || currentState == 31) {

            pr = stochasticity / 3.0;
        } else { // if(currentState == 12 || currentState == 22 || currentState == 23 || currentState == 32)
            pr = stochasticity / 4.0;
        }

    }
    return pr;
}

int Process::returnNextState(Data &data,int currentState, int currentAction){
    int nextState;
    double stochasticity = 0.0;
    vector<int> listOfStates = data.getListOfStates();
    vector<int> listOfActions = data.getListOfActions();
    double stochasticityIndicator = getRandomDoubleNumber(0,1);

    if(stochasticityIndicator < stochasticity ){
        int randomAction = listOfActions.at(getRandomIntNumber(listOfActions.size()));
        while(randomAction == currentAction){
            randomAction = listOfActions.at(getRandomIntNumber(listOfActions.size()));
        }
        nextState = currentState + randomAction;
    }else{
        nextState = currentState + currentAction;
    }

    if((find(listOfStates.begin(), listOfStates.end(), nextState) != listOfStates.end())
            && !isBlockedlState(nextState)){
        return nextState;
    }
    return currentState;
}

int Process::transitionFunction(Data &data,int currentState, int currentAction){
    int nextState;
    double stochasticity = data.getStochasticity();
    vector<int> listOfStates = data.getListOfStates();
    vector<int> listOfActions = data.getListOfActions();
    double stochasticityIndicator = getRandomDoubleNumber(0,1);

    if(stochasticityIndicator < stochasticity ){
        int randomAction = listOfActions.at(getRandomIntNumber(listOfActions.size()));
        while(randomAction == currentAction){
            randomAction = listOfActions.at(getRandomIntNumber(listOfActions.size()));
        }
        nextState = currentState + randomAction;
    }else{
        nextState = currentState + currentAction;
    }

    if((find(listOfStates.begin(), listOfStates.end(), nextState) != listOfStates.end())
            && !isBlockedlState(nextState)){
        return nextState;
    }
    return currentState;
}

bool Process::isInTerminlState(vector<double> &point){
    bool flag = false;

    double x = point.at(0);
    double y = point.at(1);
    // can improve this part!
    if(((x >= 30 && x <= 40) && (y >= 20 && y <= 30)) ||
            ((x >= 30 && x <= 40) && (y >= 10 && y <= 20))){
        flag = true;
    }
    return flag;
}

bool Process::isTerminalState(int state){
    bool flag = false;

    if(state == 23 || state == 33){
        flag = true;
    }
    return flag;
}

bool Process::isBlockedlState(int state){
    bool flag = false;

    if(state == 21){
        flag = true;
    }
    return flag;
}

double Process::getRandomDoubleNumber(int lowerBound, int upperBound){
    double min = (double) lowerBound;
    double max = (double) upperBound;
    return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
}

int Process::getRandomIntNumber(int upperBound){

    return  rand() % upperBound;
}

vector<double> Process::rowNormalizer(vector<int> row){
    vector<double> result;
    int sum = 0;
    for (int i = 0 ; i < (int)row.size() ; i++){
        sum = sum + row.at(i);
    }

    if (sum != 0){
        for (int i = 0 ; i < (int)row.size() ; i++){
            result.push_back((double) row.at(i) / (double) sum);
        }
    }else{
        for (int i = 0 ; i < (int)row.size() ; i++){
            result.push_back(0);
        }
    }


    return result;
}

int Process::getStateIndex(Data &data, int state){
    int stateIndex;
    vector<int> stateList = data.getListOfStates();

    for (int i = 0 ; i < (int)stateList.size() ; i++){
        if (stateList.at(i) == state){
            stateIndex = i;
        }
    }
    return stateIndex;
}

int Process::getObsChangeFlagIndex(int ObsChangeFlag){
    int ObsChangeFlagIndex;

    if (ObsChangeFlag == -1){
        ObsChangeFlagIndex = 0;
    }else if (ObsChangeFlag == 0){
        ObsChangeFlagIndex = 1;
    }else{
        ObsChangeFlagIndex = 2;
    }

    return ObsChangeFlagIndex;
}

int Process::getActionIndex(Data &data, int action){
    int actionIndex;
    vector<int> actionList = data.getListOfActions();

    for (int i = 0 ; i < (int)actionList.size() ; i++){
        if (actionList.at(i) == action){
            actionIndex = i;
        }
    }
    return actionIndex;
}

void Process::trainObs(Data &data){
    vector<double> * low  = data.getLow();
    vector<double> * high = data.getHigh();
    DETree observationModel(data.getFlatObsList(), low, high);
    data.setObsModel(observationModel);
}

void Process::generateTrajectories(Data &data, int numberOfTrajectories, bool forTrainingObs){
    for (int i = 0 ; i < numberOfTrajectories ; i++){
        //        cout << "Trajectory " << i << " is Done!\n";
        buildAContinuousTrajectoryAndDiscreteTrajectory(data, forTrainingObs);
    }
    cout << "Trajectories generated!\n";
    cout << "bildObsList started!\n";
    bildObsList(data, forTrainingObs);
    cout << "bildObsListdone!\n";
    if(forTrainingObs){
        trainObs(data);
        cout << "Saving obsModel!\n";
        saveObsModel(data);
        cout << "obsModel saved!\n";
    }
}

vector<double> Process::getFeatures(int state, int action){
    vector<double> features;
    if (state == 23 && action == 0){
        features.push_back(1.0);
        features.push_back(0.0);
    }else if (state == 33 && action == 0){
        features.push_back(0.0);
        features.push_back(1.0);
    }else{
        features.push_back(0.0);
        features.push_back(0.0);
    }

    return features;
}

double Process::calcInnerProduct(vector<double> v1, vector<double> v2){

    //    double result = inner_product(v1.begin(), v1.end(), v2.begin(), 0);

    //    return result;
    double result = 0.0;
    if(int(v1.size()) != int(v2.size())){
        cout << "Something is wrong in Process::calcInnerProduct" << endl;
        return 0.0;
    }
    for (int i = 0 ; i < int(v1.size()) ; i++){
        result = result + v1.at(i) * v2.at(i);
    }
    return result;
}

vector<double> Process::multiply(double scalar, vector<double> v){
    vector<double> result;
    for (int i = 0 ; i < int(v.size()) ; i++){
        result.push_back(scalar * v.at(i));
    }
    return result;
}

vector<double> Process::divide(double scalar, vector<double> v){
    vector<double> result;
    for (int i = 0 ; i < int(v.size()) ; i++){
        result.push_back(v.at(i) / scalar);
    }
    return result;
}

vector<double> Process::add(vector<double> v1, vector<double> v2){
    vector<double> result;
    for (int i = 0 ; i < int(v1.size()) ; i++){
        result.push_back(v1.at(i) + v2.at(i));
    }
    return result;
}

double Process::l1norm(vector<double> v){
    double returnval = 0;
    for (int i = 0 ; i < int(v.size()) ; i++){
        returnval += abs(v.at(i));
    }
    return returnval;
}

double Process::l2norm(vector<double> v){
    return sqrt(calcInnerProduct(v,v));
}

void Process::loadObsModel(Data &data){
    vector<Sample> flatObsList;
    string fileAddress = data.getObsModelAddress();
    std::ifstream ifs(fileAddress);
    boost::archive::text_iarchive ia(ifs);
    ia >> flatObsList;
    //    string line;
    //    ifstream myfile (fileAddress.c_str());

    //    if (myfile.is_open()){
    //        Sample s;
    //        int counter = 0;
    //        while ( getline (myfile,line) ){
    //            if(counter != 4){ // if you use nonlinear regression that 4 must be more. parameters+2(state,action)+1(p)-1
    //                s.values.push_back(stod(line));
    //                counter++;
    //            }else{
    //                s.p = stod(line);
    //                flatObsList.push_back(s);
    //                s = Sample();
    //                counter = 0;
    //            }
    //        }

    //        myfile.close();
    //    }else{
    //        cout << "Unable to open file";
    //    }
    vector<double> * low  = data.getLow();
    vector<double> * high = data.getHigh();
    DETree observationModel(flatObsList, low, high);
    data.setObsModel(observationModel);
}


void Process::saveObsModel(Data &data){
    vector<Sample> flatObsList = data.getFlatObsList();
    ofstream myfile;
    string fileAddress = data.getObsModelAddress();
    myfile.open (fileAddress.c_str());
    //    for (int i = 0 ; i < int(flatObsList.size()) ; i++){
    //        cout << flatObsList.at(i).values.size() << endl;
    //        for (int j = 0 ; j < int(flatObsList.at(i).values.size()) ; j++){
    //            myfile << flatObsList.at(i).values.at(j)<< "\n";
    //        }
    //        myfile << flatObsList.at(i).p << "\n";
    //    }
    //    myfile.close();
    std::ofstream ofs(fileAddress.c_str());
    boost::archive::text_oarchive oa(ofs);
    // write class instance to archive
    oa << flatObsList;
}

bool  Process::areAdj(int state1, int state2){
    if(abs(state1 - state2) == 0 || abs(state1 - state2) == 10 || abs(state1 - state2) == 1){
        if(state1 != 21 && state2 != 21){// can do it using pr
            return true;
        }
        return false;
    }
    return false;
}

vector<double> Process::normalize(vector<double> v){
    double sum = 0;
    vector<double> result;
    for (int i = 0 ; i < int(v.size()) ; i++){
        sum = sum + v.at(i);
    }
    for (int i = 0 ; i < int(v.size()) ; i++){
        result.push_back(v.at(i) / sum);
    }
    return result;
}
