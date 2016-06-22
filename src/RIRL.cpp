#include "data.h"
#include "process.h"
#include "RIRL.h"

#include <vector>
#include <map>
#include <algorithm>
#include <float.h>
#include <chrono>

RIRL::RIRL()
{

}

RIRL::~RIRL()
{

}

void RIRL::constructLearnerPolicy(Data &data, map<vector<int>,map<int,double>> Q_value){
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
    data.updateLearnerPolicy(policy);
}

vector<double> RIRL::calcFeatureExpectationLeft(Data &data, Process &pr, vector<double> weights){

    vector<vector<int>> listOfStates = data.getListOfStates();
    vector<int> listOfActions = data.getListOfActions();
    int sampleLength = data.getSampleLength();
    map<vector<int>,map<int,double>> policy = data.getLearnerPolicy();

    //use softmax q value iteration to generate the policy
    map<vector<int>,map<int,double>> qValuesForLearner = pr.qValueSoftMaxSolver(data, 0.01, weights);
    constructLearnerPolicy(data,qValuesForLearner);

    //forward pass
    //initialize D_si,0
    vector< map < vector<int>, double> > d;// this a vector of maps that contains all D_sk for t=0 to t=N
    map < vector<int>, double> initials = data.getListOfStateInitialPriorities(); // initiall probability of the states
    d.push_back(initials);
    //recursive part
    for (int t = 1 ; t < sampleLength ; t++) {
        map < vector<int>, double> temp;
        for (int k = 0 ; k < int(listOfStates.size()) ; k++) {
            vector<int> state_k = listOfStates.at(k);
            temp[state_k] = 0.0;
            for (int i = 0 ; i < int(listOfStates.size()) ; i++) {
                vector<int> state_i = listOfStates.at(i);

                for (int j = 0 ; j < int(listOfActions.size()) ; j++) {
                    int action = listOfActions.at(j);
                    double probabilityOfTheNextState; // here we have to calc Pr(s_k|s_i,a_i,j)
                    probabilityOfTheNextState =
                            pr.probablityOfNextStateGivenCurrentStateAction(data,state_k, state_i, action);
                    temp[state_k] += d.at(t-1)[state_i] * policy[state_i][action] *  probabilityOfTheNextState;
                }

            }
        }
        d.push_back(temp);
    }
    // here to calculate feature excptation but first calculate DS
    vector < double> featureExpectation(weights.size(), 0.0); //double [] returnval = new double[ff.dim()]; returnval[] = 0;

    // calculate D_s (6)
    map < vector<int>, double> ds; //D_s[State] ==> stateVisitationFrequency
    for (int t = 1 ; t < sampleLength ; t++) {
        for (int i = 0 ; i < int(listOfStates.size()) ; i++) {
            vector<int> state_i = listOfStates.at(i);
            if (t == 0) {
                ds[state_i] = 0.0;
            }
            ds[state_i] += d.at(t)[state_i];
        }
    }

    for (int i = 0 ; i < int(listOfStates.size()) ; i++) {
        vector<int> state_i = listOfStates.at(i);

        for (int j = 0 ; j < int(listOfActions.size()) ; j++) {
            int action = listOfActions.at(j);
            vector <double> featureVector = pr.getFeatures(data,state_i,action);
            featureExpectation = pr.add(featureExpectation ,pr.multiply(ds[state_i] * policy[state_i][action] , featureVector));
        }
    }
    return featureExpectation;
}

vector<double> RIRL::exponentiatedGradient(Data &data, Process &pr, vector<double> y,//y is coming from e-step. it is the feature expectation
                                           vector<double> w, double c, double err){
    double y_norm = pr.l1norm(y);
    if (y_norm != 0){
        y = pr.divide(y_norm, y);
    }

    for (int i = 0 ; i < int(w.size()) ; i++) {
        w.at(i) = abs(w.at(i));
    }
    double w_norm = pr.l1norm(w);

    if (w_norm != 0){
        w = pr.divide(w_norm, w);
    }

    double diff;
    double lastdiff = DBL_MAX;

    do {

        vector<double> y_prime = calcFeatureExpectationLeft(data,pr,w);

        double y_prime_norm = pr.l1norm(y_prime);
        if (y_prime_norm != 0)
            y_prime = pr.divide(y_prime_norm, y_prime);

        vector<double> next_w;
        for (int i = 0 ; i < int(w.size()) ; i++) {
            next_w.push_back(w.at(i) * exp(-2*c*(y_prime.at(i) - y.at(i)))) ;
        }

        double norm = pr.l1norm(next_w);
        if (norm != 0)
            next_w = pr.divide(norm, next_w);

        vector<double>  test = w;
        //        test[] -= next_w[];
        for (int i = 0 ; i < int(test.size()) ; i++){
            test.at(i) = test.at(i) - next_w.at(i);
        }
        diff = pr.l2norm(test);

        w = next_w;
        c /= 1.05;
        /*			if (diff > lastdiff) {
            c /= 1.1;
        } else {
            c *= 1.05;
        }*/
        lastdiff = diff;
        cout << "Diff = " << diff << endl;
    } while (diff > err);

    return w;
}

vector<Node> RIRL::returnChildren(Data &data, Process &pr, Node node){
    vector<vector<int>> listOfStates = data.getListOfStates();
    vector<int> listOfActions = data.getListOfActions();
    vector<Node> result;

    if(node.isFakeNode){
        // generate a list of initial states
        for(int i = 0 ; i < 1 ; i++){//just consider 10 for initial state // modified for debug change back to for(int i = 0 ; i < 1 ; i++)
            for(int j = 0 ; j < int(listOfActions.size()) ; j++){
                Node child;
                child.isFakeNode = false;
                child.isInitalState = true;
                child.state = listOfStates.at(i);
                child.action = listOfActions.at(j);
                result.push_back(child);
            }
        }
    }else{
        // calc the next state using dtermistic transision function
        // then iterate through all action and make all children
        for(int j = 0 ; j < int(listOfActions.size()) ; j++){
            Node child;
            child.previousState = node.state;
            child.previousAction = node.action;
            child.isFakeNode = false;
            child.isInitalState = false;
            // should if the next state is a terminal state just add action do nothing
            vector<int> nextState = pr.returnNextState(data,node.state,node.action);
            child.state = nextState;
            child.action = listOfActions.at(j);
            result.push_back(child);
        }
    }
    return result;
}

vector<double> RIRL::eStep(Data &data, Process &pr, vector<Sample> w,bool deterministicObs){ // E-step main function
    // to retain the ultimate feature count
    vector<double> f(data.getNumberOfFeatures(),0.0);

    // go through all observation sequence

    //    for(int i = 0 ; i < int(allW.size()) ; i++){
    //        vector<Sample> w = allW.at(i);
    double normalizerVectorForPrTgivenW = 0.0;
    vector<double> featureExpectation(data.getNumberOfFeatures(),0.0);
    vector<double> featureVector(data.getNumberOfFeatures(),0.0);
    vector<vector<int>> t;
    Node node;
    node.isFakeNode = true;
    int tCounter = 0 ;// the tCounter show number of trajectories considered.
    eStepRecursiveUtil(data,pr,w,node,0,1,1,0,
                       normalizerVectorForPrTgivenW,
                       featureVector,featureExpectation,tCounter,t,deterministicObs);
    //normlize featureExpectation
    for (int j = 0 ; j < int(featureExpectation.size()) ; j++){
        featureExpectation.at(j) = featureExpectation.at(j) / normalizerVectorForPrTgivenW;
    }
    f = pr.add(f,featureExpectation);
    cout << "Number of T considered is: " << tCounter << endl;
    //    }
    return f;
}

void RIRL::eStepRecursiveUtil(Data &data, Process &pr, vector<Sample> w, Node node, int level
                              ,double prT, double prWgivenT,double normalizerForObsModel,
                              double &normalizerVectorForPrTgivenW, vector<double> &featureVector,
                              vector<double> &featureExpectationVector, int &tCounter,
                              vector<vector<int>> &t, bool deterministicObs){ // E-step recursive // counter total number of T considered // adding T to print trajectory at end of recursion call
    //    cout << level << endl;
    if(level == int(w.size())){
        double prWgivenTNormalized;
        if(deterministicObs){
            prWgivenTNormalized = prWgivenT;
        }else{
            // I am useing clusteringObsModel
            prWgivenTNormalized = prWgivenT; // normalizerForObsModel;
        }
        double prTgivenW = prWgivenTNormalized * prT;
        // accumulate normalizer after compilation of each T
        normalizerVectorForPrTgivenW = normalizerVectorForPrTgivenW + prTgivenW;
        //later i need to normalize this featureExpectationVector in the eStep function
        featureExpectationVector = pr.add(featureExpectationVector,
                                          pr.multiply(prTgivenW, featureVector));
        tCounter++;
        // print each trajectory with probablities here
        //        printNestedVector(t);
        //        cout << "Pr(T) = " << prT << endl;
        //        cout << endl;
        return;
    }
    vector<Node> children = returnChildren(data, pr, node);
    for(int i = 0 ; i < int(children.size()) ; i++){

        Node child = children.at(i);
        vector<double> tempFeatureVector = featureVector;
        vector<vector<int>> tempT = t;

        //updating common parameters before recursive call.
        int newLevel = level + 1;
        double newPrT;
        Sample s = w.at(level);
        s.values.push_back(child.state.at(0));
        s.values.push_back(child.state.at(1));
        s.values.push_back(child.action);
        double obsPr = 0.0;
        if(deterministicObs){
            vector<Sample> dObsModel = data.getDeterministicObsModel();
            for(int k = 0 ; k < int(dObsModel.size()) ; k++){
                vector<double> idealSvalues = dObsModel.at(k).values;
                if(s.values == idealSvalues){
                    obsPr = 1.0;
                    // cout << "The probability of the observation is 1" << endl;
                }
            }
        }else{
            obsPr = clalObsPrUsingClusteringObsModel(data, pr, s);
        }
        double newPrWgivenT = prWgivenT * obsPr;
        double newNormalizerForObsModel = normalizerForObsModel + newPrWgivenT;
        featureVector = pr.add(featureVector,pr.getFeatures(data,child.state,child.action));
        // update Trajectory
        vector<int> stateActionTuple;
        stateActionTuple.push_back(child.state.at(0));
        stateActionTuple.push_back(child.state.at(1));
        stateActionTuple.push_back(child.action);
        t.push_back(stateActionTuple);
        //check if in the initial state so the updating parameters will be different
        if(child.isInitalState){// updating parameters before recursive call. if we are in initial state.
            newPrT = prT * data.getStateInitialPriority(child.state);

        }else{ // updating parameters before recursive call. if we are not in initial state.
            newPrT = prT *
                    pr.probablityOfNextStateGivenCurrentStateAction(
                        data,child.state,
                        child.previousState,child.previousAction) *
                    data.getLearnerPolicy()[child.state][child.action];
        }
        // recursive call
        eStepRecursiveUtil(data,pr,w,child,newLevel,newPrT,newPrWgivenT,newNormalizerForObsModel,
                           normalizerVectorForPrTgivenW,featureVector,featureExpectationVector,tCounter,t,deterministicObs);
        // restore feature vector for going back to the previous level
        // normalizerVectorForPrTgivenW and featureExpectationVector must keep acumulate for all Ts
        // restore t
        featureVector = tempFeatureVector;
        t = tempT;
    }
}

double RIRL::obsNormlizer(Data & data, Sample s){
    double normalizer = 0;
    double samplePr = data.getObsModel().density_value(s,0.5);

    vector<Sample> flatObsList = data.getFlatObsList();
    for(int i = 0 ; i < int(flatObsList.size()) ; i++){
        Sample sample;
        sample.p = s.p;
        sample.values.push_back(flatObsList.at(i).values.at(0));
        sample.values.push_back(flatObsList.at(i).values.at(1));
        sample.values.push_back(s.values.at(2));
        sample.values.push_back(s.values.at(3));
        sample.values.push_back(s.values.at(4));
        double pr = data.getObsModel().density_value(sample,0.5);
        normalizer = normalizer + pr;
    }
    vector<double> v;
    for(int i = 0 ; i < int(flatObsList.size()) ; i++){
        Sample sample;
        sample.p = s.p;
        sample.values.push_back(flatObsList.at(i).values.at(0));
        sample.values.push_back(flatObsList.at(i).values.at(1));
        sample.values.push_back(s.values.at(2));
        sample.values.push_back(s.values.at(3));
        sample.values.push_back(s.values.at(4));
        v.push_back(data.getObsModel().density_value(sample,0.5)/normalizer);
    }
    printVector(v);
    samplePr = samplePr / normalizer;
    return samplePr;
}

void RIRL::initializePolicy(Data & data, Process &pr,vector<double> weights){

    // I was initiliazing the policy randomly or uniformly before now I will do
    // it based on random weights.
    map<vector<int>,map<int,double>> policy = data.getLearnerPolicy();

    //use softmax q value iteration to generate the policy
    map<vector<int>,map<int,double>> qValuesForLearner = pr.qValueSoftMaxSolver(data, 0.01, weights);
    constructLearnerPolicy(data,qValuesForLearner);

}

void RIRL::printNestedVector(vector<vector<int>> v){
    int counter = 0;
    for(int i = 0 ; i < int(v.size()) ; i++){
        cout << "<" << v.at(i).at(0) << " , " << v.at(i).at(1) << "> , ";
        counter ++;
    }
    cout << "The length is == " << counter << endl;

}

void RIRL::printVector(vector<double> v){
    for(int i = 0 ; i < int(v.size()) ; i++){
        if(i == 0){
            cout << "< "<< v.at(i) << " , ";

        }else if(i < int(v.size())-1){
            cout << v.at(i) << " , ";

        }else{
            cout << v.at(i) << " >" << endl;
        }
    }
}

vector<double> RIRL::gradientDescent(Data &data, Process &pr, vector<double> y,//y is coming from e-step. it is the feature expectation
                                     vector<double> w, double c, double err){


    double diff;
    double lastdiff = DBL_MAX;

    do {

        vector<double> y_prime = calcFeatureExpectationLeft(data,pr,w);

        vector<double> next_w;
        for (int i = 0 ; i < int(w.size()) ; i++) {
            next_w.push_back(w.at(i) - c * (y_prime.at(i) - y.at(i))) ;
        }


        vector<double>  test = w;
        //        test[] -= next_w[];
        for (int i = 0 ; i < int(test.size()) ; i++){
            test.at(i) = test.at(i) - next_w.at(i);
        }
        diff = pr.l2norm(test);

        w = next_w;
        c /= 1.05;
        /*			if (diff > lastdiff) {
            c /= 1.1;
        } else {
            c *= 1.05;
        }*/
        lastdiff = diff;

    } while (diff > err);
    cout << "error = " << diff << endl;
    return w;
}

double RIRL::clalObsPrUsingClusteringObsModel(Data & data, Process &pr, Sample s){
    int index;
    vector<Sample> flatObsList = data.getFlatObsList();
    vector<Sample> custeringObsModel = data.getClusteringObsModel();

    vector<int> state;
    state.push_back(s.values.at(2));
    state.push_back(s.values.at(3));
    int action = s.values.at(4);

    // find the corresponding centroid
    Sample centeroid;

    for(int i = 0 ; i < int(custeringObsModel.size()) ; i++){
        Sample tempCenteroid = custeringObsModel.at(i);
        if(state.at(0) == tempCenteroid.values.at(2) &&
                state.at(1) == tempCenteroid.values.at(3) &&
                action == tempCenteroid.values.at(4)){
            centeroid = tempCenteroid;
            break;
        }
    }

    // iterate through all observation and calculate Pr(w|s,a)
    vector<double> vectorOfPrs;
    //    double normalizer = 0.0;
    for(int i = 0 ; i < int(flatObsList.size()) ; i++){
        Sample obs = flatObsList.at(i);
        //calculate the distance
        if(s.values.at(0) == obs.values.at(0) &&
                s.values.at(1) == obs.values.at(1)){
            index = i;
        }
        double r = sqrt((centeroid.values.at(0) - obs.values.at(0)) * (centeroid.values.at(0) - obs.values.at(0))
                        +
                        (centeroid.values.at(1) - obs.values.at(1)) * (centeroid.values.at(1) - obs.values.at(1)));
        //        normalizer = normalizer + 1.0 / r;
        // pluse 1 in case of r is zero
        vectorOfPrs.push_back(1.0 / (r + 1));
    }

    double vectorOfPrs_norm = pr.l1norm(vectorOfPrs);

    if (vectorOfPrs_norm != 0){
        vectorOfPrs = pr.divide(vectorOfPrs_norm, vectorOfPrs);
    }

    //    printVector(vectorOfPrs);
    double result = vectorOfPrs.at(index);
    return result;
}
