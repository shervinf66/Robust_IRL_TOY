#ifndef PROCESS_H
#define PROCESS_H

#include "data.h"
#include <vector>

class Process
{
private:
    vector<vector<double> > matrixNormalizer(vector<vector<int> > obsCounts);
    vector<vector<double> > getContinuousSamples(Data &data,vector<double> previousPoint,vector<double> currentPoint);
    double getDistanceToPowerTwo(vector <double> sample);
    vector<double> rowNormalizer(vector<int> row);
    vector<double> getObsSampleList(Data &data, vector<double> previousPoint, vector<double> currentPoint);
    void buildAContinuousTrajectoryAndDiscreteTrajectory(Data &data, bool forTrainingObs);
    void bildObsList(Data &data);
    int classifyObs(Data &data, double avg);
    vector<double> getRandomPointInState(Data &data, int state);
    vector<int> calcStateBoundaries(Data &data, int state);
    vector<double> calcCenterBoxBoundaries(Data &data, int state);
    int rationalPolicy(int state);
    int randomPolicy(Data &data);
    int transitionFunction(Data &data,int currentState, int currentAction);
    bool isInTerminlState(vector<double> &point);
    bool isTerminlState(int state);
    bool isBlockedlState(int state);
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
    void generateTrajectories(Data &data, int numberOfTrajectories, bool forTrainingObs);

};

#endif // PROCESS_H
