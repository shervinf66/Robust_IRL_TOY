#ifndef PROCESS_H
#define PROCESS_H

#include "data.h"
#include <vector>
#include <string>

class Process
{
private:
    vector<vector<double> > matrixNormalizer(vector<vector<int> > obsCounts);
    vector<vector<double> > getContinuousSamples(Data &data,vector<double> previousPoint,vector<double> currentPoint);
    double getDistanceToPowerTwo(vector <double> sample);
    vector<double> rowNormalizer(vector<int> row);
    vector<double> getObsSampleList(Data &data, vector<double> previousPoint, vector<double> currentPoint);
    void buildAContinuousTrajectoryAndDiscreteTrajectory(Data &data, bool forTrainingObs);
    void bildObsList(Data &data, bool forTrainingObs);
    int classifyObs(Data &data, double avg);
    vector<double> getRandomPointInState(Data &data, int state);
    vector<int> calcStateBoundaries(Data &data, int state);
    vector<double> calcCenterBoxBoundaries(Data &data, int state);
    int rationalPolicy(int state);
    int randomPolicy(Data &data);
    int transitionFunction(Data &data,int currentState, int currentAction);
    bool isInTerminlState(vector<double> &point);
    double getRandomDoubleNumber(int lowerBound, int upperBound);
    int calcIntensityChangeFlag(Data &data, double intensityDifference);
    int getRandomIntNumber(int upperBound);
    void trainObs(Data &data);
    int getStateIndex(Data &data, int state);
    int getObsChangeFlagIndex(int ObsChangeFlag);
    int getActionIndex(Data &data, int action);

public:
    Process();
    ~Process();
    bool isTerminalState(int state);
    bool isBlockedlState(int state);
    vector<double> getFeatures(int state, int action);
    double calcInnerProduct(vector<double> v1, vector<double> v2);
    double probablityOfNextStateGivenCurrentStateAction(Data &data,int nextState,int currentState,int currentAction);
    void generateTrajectories(Data &data, int numberOfTrajectories, bool forTrainingObs);
    vector<double> multiply(double scalar, vector<double> v);
    vector<double> divide(double scalar, vector<double> v);
    vector<double> add(vector<double> v1, vector<double> v2);
    double l1norm(vector<double> v);
    double l2norm(vector<double> v);
    void loadObsModel(Data &data);
    void saveObsModel(Data &data);
    bool areAdj(int state1, int state2);
    vector<double> normalize(vector<double> v);
    int nextState(Data &data,int currentState, int currentAction);
};

#endif // PROCESS_H
