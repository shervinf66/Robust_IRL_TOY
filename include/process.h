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
    void bulidData(Data &data, int lenghtOfTrajectory, bool forTrainingObs);
    void updateFlatObsList(Data &data, vector<int> state, int action, vector<double> obs, bool forTrainingObs);
    int classifyObs(Data &data, double avg);
    vector<double> getCenterPointInState(int rawState);
    vector<int> calcStateBoundaries(Data &data, int state);
    vector<double> calcCenterBoxBoundaries(Data &data, int state);
    int rationalPolicy(int state);
    int randomPolicy(Data &data);
    vector<int> transitionFunction(Data &data, vector<int> currentState, int currentAction);
    bool isInTerminlState(vector<double> &point);
    double getRandomDoubleNumber(int lowerBound, int upperBound);
    int calcIntensityChangeFlag(Data &data, double intensityDifference);
    int getRandomIntNumber(int upperBound);
    void trainObs(Data &data);
    int getStateIndex(Data &data, int state);
    int getObsChangeFlagIndex(int ObsChangeFlag);
    int getActionIndex(Data &data, int action);
    void constructExpertPolicy(Data &data,map<vector<int>,map<int,double>> Q_value);

public:
    Process();
    ~Process();
    bool isTerminalState(int state);
    bool isBlockedlState(int state);
    vector<double> getFeatures(Data &data, vector<int> state, int action);
    double calcInnerProduct(vector<double> v1, vector<double> v2);
    double probablityOfNextStateGivenCurrentStateAction(Data &data, vector<int> returnNextState, vector<int> currentState, int currentAction);
    void generateTrajectories(Data &data, int lenghtOfT, bool forTrainingObs);
    vector<double> multiply(double scalar, vector<double> v);
    vector<double> divide(double scalar, vector<double> v);
    vector<double> add(vector<double> v1, vector<double> v2);
    double l1norm(vector<double> v);
    double l2norm(vector<double> v);
    void loadObsModel(Data &data);
    void saveObsModel(Data &data);
    bool areAdj(vector<int> currentState, vector<int> nextState);
    vector<double> normalize(vector<double> v);
    vector<int> returnNextState(Data &data, vector<int> currentState, int currentAction);
    double reward(Data &data, vector<int> state, int action, vector<double> weights);
    map<vector<int>, map<int, double> > qValueSoftMaxSolver(Data &data, double err, vector<double> weights);
    void printPolicy(map<vector<int>,map<int,double>> policy);
    vector<double> getFeatureVectorOfT(Data &data,vector<vector<int>> t);
    void saveDeterministicObsModel(Data &data);
    void loadDeterministicObsModel(Data &data);
    void saveClusteringObsModel(Data &data, int dataPerStateActionPair);
    void loadClusteringObsModel(Data &data);
};

#endif // PROCESS_H
