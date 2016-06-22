#include "process.h"
#include "data.h"
#include "linear.h"
#include "Sample.h"
#include "RIRL.h"
#include "DETree.h"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <limits>
#include <cfloat>
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

//each sample is a tuple <x,y>
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
    return (pow ((sample.at(0) - 10), 2) + pow ((sample.at(1) - 10), 2));
}

vector<double> Process::getObsSampleList(Data &data,vector<double> previousPoint,vector<double> currentPoint){
    //for debugging
    //    if(currentPoint == previousPoint){
    //        cout << "we are turnning!" << endl;
    //    }

    //list of <x,y> between two points
    vector< vector<double> > continuousSampleList = getContinuousSamples(data, previousPoint, currentPoint);

    vector<double> obsList;

    // parameters of the normal distribution
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
        //        double noise = 0.0; // for debugging noise free
        //        cout << (p / (4 * M_PI * r2)) << " + " << noise << endl;

        obs = (p / (4 * M_PI * r2)) + noise;

        obsList.push_back(obs);
    }

    return obsList;
}

//changed
// don't make continus trajectory any more just the discrete one. center to center move
void Process::bulidData(Data &data, int lenghtOfTrajectory, bool forTrainingObs){

    vector<double> initialPoint;
    vector<double> firstPoint;

    vector<int> firstState;
    int initialAction;
    int action;
    vector<int> initialState; //at 0 facing east
    initialState.push_back(0);
    initialState.push_back(0);

    if(forTrainingObs == true){
        initialAction = randomPolicy(data);
    }else{
        //call rational policy here
        map<vector<int>,map<int,double>> ExpertPolicy = data.getExpertPolicy();
        vector<int> listOfActions = data.getListOfActions();
        double max = -DBL_MAX;

        for(int j = 0 ; j < (int)(listOfActions.size()) ; j++ ){
            int a = listOfActions.at(j);
            if(max < ExpertPolicy[initialState][a]){
                max = ExpertPolicy[initialState][a];
                initialAction = a;
            }
        }
        //print the T for debugging
        cout << "[< " << initialState.at(0) << " , "
             << initialState.at(1) << " > , " <<
                initialAction << " ]" << endl;
    }
    initialPoint = getCenterPointInState(initialState.at(0));
    firstPoint = initialPoint;


    firstState = initialState;
    action = initialAction;

    int counter = 0;

    bool done = false;

    while(!done){
        vector<double> secondPoint;

        vector<int> secondState = transitionFunction(data, firstState, action);

        if(firstState == secondState){
            secondPoint = firstPoint;
        }else{
            secondPoint = getCenterPointInState(secondState.at(0));
        }

        vector<double> obs = getObsSampleList(data,firstPoint, secondPoint);
        updateFlatObsList(data,firstState,action,obs,forTrainingObs);

        firstState = secondState;
        firstPoint = secondPoint;
        counter++;

        if(counter == lenghtOfTrajectory){
            done = true;
        }else{
            if(forTrainingObs == true){
                action = randomPolicy(data);
            }else{
                //call rational policy here
                map<vector<int>,map<int,double>> ExpertPolicy = data.getExpertPolicy();
                vector<int> listOfActions = data.getListOfActions();
                double max = -DBL_MAX;

                for(int j = 0 ; j < (int)(listOfActions.size()) ; j++ ){
                    int a = listOfActions.at(j);
                    if(max < ExpertPolicy[firstState][a]){
                        max = ExpertPolicy[firstState][a];
                        action = a;
                    }
                }
                //print the T for debugging
                cout << "[< " << firstState.at(0) << " , "
                     << firstState.at(1) << " > , " <<
                        action << " ]" << endl;
            }
        }
    }
}

// copress that to above function
void Process::updateFlatObsList(Data &data, vector<int> state, int action, vector<double> obs, bool forTrainingObs){
    int numberOfSamples = data.getNumberOfSamples();

    vector<double> time_x = data.getTimeChunk();
    Sample obsTuple;
    Maths::Regression::Linear linearRegression(numberOfSamples, time_x, obs); //Intensity_y = Obs
    double slope =linearRegression.getSlope();
    obsTuple.values.push_back(slope);
    double intercept =linearRegression.getIntercept();
    obsTuple.values.push_back(intercept);
    if(forTrainingObs){
        obsTuple.values.push_back(state.at(0)); // adding state.position
        obsTuple.values.push_back(state.at(1)); // adding state.orientation
        obsTuple.values.push_back(action); // adding action
    }

    obsTuple.p = data.getNormalizerFactorForP(); //uniform
    data.updateFlatObsList(obsTuple);
}

//changed
//just consider the center
//rawstate = S_0 to S_n
vector<double> Process::getCenterPointInState(int rawState){
    vector<double> result;

    double x = (rawState + 1) * 20 +10;
    double y = 30;

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

//return a random action
//done
int Process::randomPolicy(Data &data){
    vector<int> listOfActions = data.getListOfActions();
    int randomAction = listOfActions.at(getRandomIntNumber(listOfActions.size()));
    return randomAction;
}

//I have to chagne it later
double Process::probablityOfNextStateGivenCurrentStateAction(Data &data,vector<int> nextState, vector<int> currentState, int currentAction){
    double stochasticity = data.getStochasticity();
    vector<int> idealNextState = returnNextState(data,currentState,currentAction);
    double pr = 0.0;
    if (nextState == idealNextState){
        pr = 1.0 - stochasticity;
    }else if(areAdj(currentState,nextState) &&
             currentState.at(0) != 0 &&
             currentState.at(0) != data.getNumberOfRawStates()-1){//if in the middle of hallway
        pr = stochasticity / 3.0; //confirm it with ken
    }else if(areAdj(currentState,nextState)){ //if at the end of the hallway
        pr = stochasticity / 2.0; //confirm it with ken
    }

    return pr;
}

//changed
vector<int> Process::returnNextState(Data &data,vector<int> currentState, int currentAction){
    vector<int> nextState;
    if(/*(currentState.at(1) == 0 && currentAction == -1) ||*/ //facing east going backward
            (currentState.at(1) == 1 && currentAction == 1)){ //facing west going forward
        int nextPosition = currentState.at(0) - 1;
        //check if the next position in the range of possible positions
        if(nextPosition < 0){
            nextPosition = 0;
        }else if(nextPosition > data.getNumberOfRawStates() - 1){
            nextPosition = data.getNumberOfRawStates() - 1;
        }
        nextState.push_back(nextPosition);
        nextState.push_back(currentState.at(1));
    }else if(currentAction == 0){ //turning
        int nextOrientation;
        if(currentState.at(1) == 0){
            nextOrientation = 1;
        }else{
            nextOrientation = 0;
        }
        nextState.push_back(currentState.at(0));
        nextState.push_back(nextOrientation);
    }else if((currentState.at(1) == 0 && currentAction == 1) /*|| //facing east going forward
                                                                            (currentState.at(1) == 1 && currentAction == -1)*/){ //facing west going backward
        int nextPosition = currentState.at(0) + 1;
        //check if the next position (rawState) in the range of possible positions
        if(nextPosition < 0){
            nextPosition = 0;
        }else if(nextPosition > data.getNumberOfRawStates() - 1){
            nextPosition = data.getNumberOfRawStates() - 1;
        }
        nextState.push_back(nextPosition);
        nextState.push_back(currentState.at(1));
    }else{
        cout << "missing something!" << endl;
    }


    return nextState;
}

//changed
vector<int> Process::transitionFunction(Data &data,vector<int> currentState, int currentAction){
    vector<int> nextState;
    double stochasticity = data.getStochasticity();
    vector<int> listOfActions = data.getListOfActions();
    double stochasticityIndicator = getRandomDoubleNumber(0,1);

    if(stochasticityIndicator < stochasticity ){
        int randomAction = listOfActions.at(getRandomIntNumber(listOfActions.size()));
        while(randomAction == currentAction){
            randomAction = listOfActions.at(getRandomIntNumber(listOfActions.size()));
        }
        //generate next state here
        nextState = returnNextState(data, currentState, randomAction);
    }else{
        //generate next state here
        nextState = returnNextState(data, currentState, currentAction);
    }

    return nextState;
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

void Process::trainObs(Data &data){
    vector<double> * low  = data.getLow();
    vector<double> * high = data.getHigh();
    DETree observationModel(data.getFlatObsList(), low, high);
    data.setObsModel(observationModel);
}

//changed
void Process::generateTrajectories(Data &data, int lenghtOfT, bool forTrainingObs){
    data.setSampleLength(lenghtOfT);
    //setting up expert's policy
    if(!forTrainingObs){
        map<vector<int>,map<int,double>> qValuesForExpert = qValueSoftMaxSolver(data, 0.001, data.getExpertWeights());
        constructExpertPolicy(data,qValuesForExpert);
    }
    cout << "Generating data!" << endl;
    data.setNormalizerFactorForP(lenghtOfT);
    bulidData(data, lenghtOfT, forTrainingObs);
    cout << "Data generated!" << endl;

    if(forTrainingObs){
        saveObsModel(data);
        cout << "Saving obsModel!\n";
        saveObsModel(data);
        cout << "obsModel saved!\n";
    }
}

//changed
vector<double> Process::getFeatures(Data &data, vector<int> state, int action){
    vector<double> features(data.getNumberOfFeatures(), 0.0);
    int rawState = state.at(0);
    int orientation = state.at(1);
    int activatedFeatureForTurningIndex;
    //    if(rawState == 9){
    //        cout << (data.getNumberOfRawStates() - 1) << endl;
    //    }
    if(action == 0){
        if (rawState < (data.getNumberOfRawStates() - 1) / 2 && orientation == 1){ //first half should face west
            activatedFeatureForTurningIndex = rawState;
            features.at(activatedFeatureForTurningIndex) = 1.0;
        }else if(rawState > (data.getNumberOfRawStates() - 1) / 2 && orientation == 0) { //second half should face east
            activatedFeatureForTurningIndex = data.getNumberOfRawStates() - rawState - 1;
            features.at(activatedFeatureForTurningIndex) = 1.0;
        }else if(rawState == (data.getNumberOfRawStates() - 1) / 2){
            activatedFeatureForTurningIndex = (data.getNumberOfRawStates() - 1) / 2;
            features.at(activatedFeatureForTurningIndex) = 1.0;
        }else{
            //            cout << "I missing some Feature!!" << endl;
            activatedFeatureForTurningIndex = features.size()-2;
            features.at(activatedFeatureForTurningIndex) = 1.0;
        }
    }else{
        //construct the feature vector here
        //last feature is for moving forward
        //I may change this part if i want to add moving backward!
        //I have to check if I am bangging the wall it should be no reward
        if(rawState == 0 || rawState == data.getNumberOfRawStates() - 1){
            //by not activating feature ==> no reward by going into the wall
            //            activatedFeatureForTurningIndex = features.size()-2;
            //            features.at(activatedFeatureForTurningIndex) = 1.0;
        }else{
            features.at(features.size()-1) = 1.0;
        }
    }

    return features;
}

//new
map<vector<int>,map<int,double>> Process::qValueSoftMaxSolver(Data &data, double err, vector<double> weights){

    map<vector<int>,double> V; //[State] //[State] ==> vector<int>
    map<vector<int>,map<int,double>> Q; //[State][Action] //[State] ==> vector<int> & [Action] ==> int
    vector<vector<int>> listOfStates = data.getListOfStates();
    vector<int> listOfActions = data.getListOfActions();

    for(int i = 0 ; i < (int)(listOfStates.size()) ; i++ ){ //foreach (State s ; model.S()) {
        vector<int> s = listOfStates.at(i);
        V[s] = 0.0;

        for(int j = 0 ; j < (int)(listOfActions.size()) ; j++ ){//foreach (a; model.A(s)) {
            int a = listOfActions.at(j);
            Q[s][a] = 0;
        }
    }
    //    V = V.rehash;
    //    Q = Q.rehash;
    double delta = 0;
    int iteration = 0; //size_t
    while (true) {
        delta = 0;

        for(int i = 0 ; i < (int)(listOfStates.size()) ; i++ ){ //foreach (State s ; model.S()) {
            vector<int> s = listOfStates.at(i);

            for(int j = 0 ; j < (int)(listOfActions.size()) ; j++ ){//foreach (a; model.A(s)) {
                int a = listOfActions.at(j);
                double r = reward(data,s,a,weights);//add reward function later
                //double[State] T = model.T(s, a);

                double expected_rewards = 0;
                for (int k = 0 ; k < int(listOfStates.size()) ; k++){ //foreach (s_prime, p; T){
                    vector<int> s_prime = listOfStates.at(k);
                    //p ==> probabilityOfTheNextState
                    double p  = probablityOfNextStateGivenCurrentStateAction(data,s_prime, s, a);
                    expected_rewards += p*V[s_prime];
                }

                Q[s][a] = r + data.getGamma() * expected_rewards;
            }
        }

        map<vector<int>,double> v;
        v = V;
        for(int i = 0 ; i < (int)(listOfStates.size()) ; i++ ){ //foreach (State s ; model.S()) {
            vector<int> s = listOfStates.at(i);
            double maxx = -DBL_MAX;
            for(int j = 0 ; j < (int)(listOfActions.size()) ; j++ ){//foreach (a; model.A(s)) {
                int a = listOfActions.at(j);
                maxx = max(maxx, Q[s][a]);
            }

            double e_sum = DBL_MIN;

            for(int j = 0 ; j < (int)(listOfActions.size()) ; j++ ){//foreach (a; model.A(s)) {
                int a = listOfActions.at(j);
                e_sum += exp(Q[s][a] - maxx);
            }

            v[s] = maxx - log(e_sum);
            delta = max(delta, abs(V[s] - v[s]));
        }


        V = v;
        //cout << "Current Iteration: " <<  delta << endl;

        if (delta < err || iteration > data.getMaxIter()) {
            map<vector<int>,map<int,double>> returnval; //[state][action]

            //doing this part to avoid overfelow problem when do e to the power in softmax!!
            for(int i = 0 ; i < (int)(listOfStates.size()) ; i++ ){ //foreach(state, actvalue; Q) {
                vector<int> state = listOfStates.at(i);

                for(int j = 0 ; j < (int)(listOfActions.size()) ; j++ ){//foreach(action, value; actvalue) {
                    int action = listOfActions.at(j);
                    double value = Q[state][action];
                    returnval[state][action] = value - v[state];
                }
            }
            //				writeln("Finished Q-Value");
            return returnval;
        }
        iteration ++;
    }
}

//new
//this function calculate the reward reciving state and action
double Process::reward(Data &data,vector<int> state, int action, vector<double> weights){
    double reward;
    vector<double> features = getFeatures(data,state,action);
    reward = calcInnerProduct(features,weights);
    return reward;
}

//construct the expert policy using the actual weights
//using softmax
void Process::constructExpertPolicy(Data &data, map<vector<int>,map<int,double>> Q_value){
    map<vector<int>,map<int,double>> policy; //[State][Action]
    vector<vector<int>> listOfStates = data.getListOfStates();
    vector<int> listOfActions = data.getListOfActions();

    for(int i = 0 ; i < (int)(listOfStates.size()) ; i++ ){ //foreach (State s ; model.S()) {
        vector<int> s = listOfStates.at(i);
        double normalizer = 0.0; //[State]

        for(int j = 0 ; j < (int)(listOfActions.size()) ; j++ ){//foreach (Action a; model.A(s)) {
            int a = listOfActions.at(j);
            normalizer = normalizer + exp(Q_value[s][a]);
        }

        for(int j = 0 ; j < (int)(listOfActions.size()) ; j++ ){//foreach (Action a; model.A(s)) {
            int a = listOfActions.at(j);
            policy[s][a] = exp(Q_value[s][a]) / normalizer;
        }
    }
    data.setExpertPolicy(policy);
}

double Process::calcInnerProduct(vector<double> v1, vector<double> v2){

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

void Process::printPolicy(map<vector<int>,map<int,double>> policy){
    //    std::map< std::string, std::map<std::string, std::string> > m;

    for (auto i : policy){
        for (auto j : i.second){
            cout << " < " << i.first.at(0) << " , " << i.first.at(1) << " > , " << j.first << " ==> Pr = " << j.second << endl;
        }
        cout << "--------------------------------------------" << endl;
    }

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
    std::ofstream ofs(fileAddress.c_str());
    boost::archive::text_oarchive oa(ofs);
    // write class instance to archive
    oa << flatObsList;
}

//changed
bool  Process::areAdj(vector<int> state1, vector<int> state2){
    int rawState1 = state1.at(0);
    int rawState2 = state2.at(0);
    if(abs(rawState1 - rawState2) == 0 ||
            (abs(rawState1 - rawState2) == 1 /*&& state1.at(1) == state2.at(1)*/)){ //|| abs(state1 - state2) == 10
        return true;
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

vector<double> Process::getFeatureVectorOfT(Data &data,vector<vector<int>> t){
    vector<double> fv(data.getNumberOfFeatures(),0.0);
    for(int i = 0 ; i < int(t.size()) ; i++){
        vector<int> state;
        state.push_back(t.at(i).at(0));
        state.push_back(t.at(i).at(1));
        int action = t.at(i).at(2);
        fv = add(fv, getFeatures(data,state,action));
    }
    return fv;
}

void Process::saveDeterministicObsModel(Data &data){
    vector<vector<int>> listOfStates = data.getListOfStates();
    vector<int> listOfActions = data.getListOfActions();

    for(int i = 0 ; i < int(listOfStates.size()) ; i++){
        vector<int> firstState = listOfStates.at(i);
        for(int j = 0 ; j < int(listOfActions.size()) ; j++){
            int action = listOfActions.at(j);
            vector<int> secondState = returnNextState(data,firstState,action);
            vector<double> firstPoint = getCenterPointInState(firstState.at(0));
            vector<double> secondPoint = getCenterPointInState(secondState.at(0));
            vector<double> obs = getObsSampleList(data,firstPoint, secondPoint);
            updateFlatObsList(data,firstState,action,obs,true);
        }
    }
    vector<Sample> flatObsList = data.getFlatObsList();
    ofstream myfile;
    string fileAddress = data.getObsModelAddress();
    myfile.open (fileAddress.c_str());
    std::ofstream ofs(fileAddress.c_str());
    boost::archive::text_oarchive oa(ofs);
    // write class instance to archive
    oa << flatObsList;
}

void Process::loaddeterministicObsModel(Data &data){
    vector<Sample> dObsModel;
    string fileAddress = data.getObsModelAddress();
    std::ifstream ifs(fileAddress);
    boost::archive::text_iarchive ia(ifs);
    ia >> dObsModel;
    data.setDeterministicObsModel(dObsModel);
}
