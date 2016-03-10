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
        double p = data.getP();

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
    int initialState = 10;

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
            if(isTerminlState(nextState)){
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

        vector<double> obs = getObsSampleList(data, aContinusTrajectory.at(lastPointIndex), point);
        ObsSampleList.insert(ObsSampleList.end(), obs.begin(), obs.end());
        lastPointIndex = lastPointIndex + 1;
        aContinusTrajectory.push_back(point);

        currentState = nextState;
    }
    data.addAContinuousTrajectory(aContinusTrajectory);
    data.addADiscreteTrajectory(aDiscreteTrajectory);
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

void Process::bildObsList(Data &data){
    int numberOfSamples = data.getNumberOfSamples();

    vector<vector<double> > obsSampleList = data.getObsSampleList();

    for (int i = 0 ; i < (int)obsSampleList.size() ; i++){
        vector<double> obsSampleForOneTrajectory = obsSampleList.at(i);

        // New way using regression that is continuous
        vector<Sample > obsForOneTrajectory;
        for (int j = 0 ; j < (int)obsSampleForOneTrajectory.size() ; j = j + numberOfSamples){
            vector<double> IntensityChunk;
            for(int k = 0 ; k < numberOfSamples ; k++){
                IntensityChunk.push_back(obsSampleForOneTrajectory.at(j + k));
            }
            vector<double> timeChunk = data.getTimeChunk();
            Sample obsTuple;
            Maths::Regression::Linear linearRegression(numberOfSamples, timeChunk, IntensityChunk);
            double slope =linearRegression.getSlope();
            obsTuple.values.push_back(slope);
            double intercept =linearRegression.getIntercept();
            obsTuple.values.push_back(intercept);
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

bool Process::isTerminlState(int state){
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

// this old and for the descrite model. not in use any more
/**
void Process::trainObs(Data &data){
    int numberOfDisObs = data.getNumberOfDisObs();
    int numberOfStates = data.getListOfStates().size();
    int numberOfAction = data.getListOfActions().size();
    int numberOfRow = numberOfStates * numberOfAction;
    int numberOfcolumn = numberOfDisObs * 3; // 3 is the number of flags -1 0 +1

    vector< vector<int> > obsCounts (numberOfRow  , vector<int> (numberOfcolumn , 0));

    vector<vector<vector<double> > > obsList = data.getObsList();
    vector< vector<vector<int> > > discreteTrajectories = data.getDiscreteTrajectories();

    for (int i = 0 ; i < (int)obsList.size() ; i++){
        for (int j = 0 ; j < (int)obsList.at(i).size() ; j++){
            vector<double> obsTuple = obsList.at(i).at(j);

            int obsClass = obsTuple.at(0);
            int ObsChangeFlag = obsTuple.at(1);

            int state = discreteTrajectories.at(i).at(j).at(0);
            int action = discreteTrajectories.at(i).at(j).at(1);
            int actionIndex = getActionIndex(data, action);
            int ObsChangeFlagIndex = getObsChangeFlagIndex(ObsChangeFlag);
            int stateIndex = getStateIndex(data, state);

            int rowIndex = stateIndex * numberOfAction + actionIndex;
            int columnIndex = obsClass * 3 + ObsChangeFlagIndex;
            obsCounts.at(rowIndex).at(columnIndex) += 1;
        }
    }
    // normalize first!
    vector< vector<double> > normalizedObsCounts = matrixNormalizer(obsCounts);
    data.setObsModel(normalizedObsCounts);
}

vector< vector<double> > Process::matrixNormalizer(vector< vector<int> > obsCounts){
    vector< vector<double> > result;
    for (int i = 0 ; i < (int)obsCounts.size() ; i++){
        vector<double> normalizedRow = rowNormalizer(obsCounts.at(i));
        result.push_back(normalizedRow);
    }
    return result;
}
**/

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
    int x = 0;
}

void Process::generateTrajectories(Data &data, int numberOfTrajectories, bool forTrainingObs){
    for (int i = 0 ; i < numberOfTrajectories ; i++){
        buildAContinuousTrajectoryAndDiscreteTrajectory(data, forTrainingObs);
    }
    bildObsList(data);
    if(forTrainingObs){
        trainObs(data);
    }
}

