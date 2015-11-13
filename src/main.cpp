#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <glog/logging.h>
#include "Sampler.h"
#include "Sample.h"
#include "MCFHMM.h"
#include "DETree.h"
#include "Observation.h"
#include "Timer.h"
using namespace std;
using namespace google;

void init_GLOG(int argc, char* argv[]);
void init_limits();
void init_observations(size_t);

vector<double> pi_low_limits;
vector<double> pi_high_limits;
vector<double> m_low_limits;
vector<double> m_high_limits;
vector<double> v_low_limits;
vector<double> v_high_limits;
vector<Observation> obs;

int main(int argc, char* argv[])
{
    init_GLOG(argc, argv);

    init_limits();
    init_observations(100);

    MCFHMM hmm;

    int N = 20;
    int max_iteration = 20;

    hmm.set_limits(&pi_low_limits, &pi_high_limits, &m_low_limits, &m_high_limits, &v_low_limits, &v_high_limits);

    // Learn the HMM structure
    {
        Timer tmr;
        double t1 = tmr.elapsed();
        hmm.learn_hmm(&obs, max_iteration, N);
        double t2 = tmr.elapsed();
        LOG(INFO) << "Learning Time: " << (t2 - t1) << " seconds";
    }

    init_observations(10);

    vector<vector<Sample> > forward = hmm.forward(&obs, 100);

    // BRANCH

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Initialization Part ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

void init_GLOG(int argc, char* argv[]){
    InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 0;
    FLAGS_log_dir = ".";
    FLAGS_minloglevel = 0;
    //FLAGS_logtostderr = true;
    //SetLogDestination(google::INFO, "./info");
}

void init_observations(size_t size){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    uniform_real_distribution<double> dist(-.1, 0.1);
    double x = 0;
    double y = 0;
    double th = 0;
    obs.clear();
    for (size_t i = 0; i < size / 2; i++){
        Observation temp1;
        temp1.values.push_back(x + dist(gen));
        temp1.values.push_back(y + dist(gen));
        temp1.values.push_back(th + dist(gen));
        obs.push_back(temp1);

        Observation temp2;
        temp2.values.push_back( x + 10.0 + dist(gen));
        temp2.values.push_back( y + 10.0 + dist(gen));
        temp2.values.push_back(th + 10.0 + dist(gen));
        obs.push_back(temp2);
    }
}

void init_limits(){
    pi_low_limits.push_back(0);
    pi_low_limits.push_back(0);
    pi_low_limits.push_back(0);

    pi_high_limits.push_back(10);
    pi_high_limits.push_back(10);
    pi_high_limits.push_back(2 * M_PI);

    m_low_limits.push_back(0);
    m_low_limits.push_back(0);
    m_low_limits.push_back(0);
    m_low_limits.push_back(0);
    m_low_limits.push_back(0);
    m_low_limits.push_back(0);

    m_high_limits.push_back(10);
    m_high_limits.push_back(10);
    m_high_limits.push_back(2 * M_PI);
    m_high_limits.push_back(10);
    m_high_limits.push_back(10);
    m_high_limits.push_back(2 * M_PI);

    v_low_limits.push_back(0);
    v_low_limits.push_back(0);
    v_low_limits.push_back(0);
    v_low_limits.push_back(0);
    v_low_limits.push_back(0);
    v_low_limits.push_back(0);

    v_high_limits.push_back(10);
    v_high_limits.push_back(10);
    v_high_limits.push_back(2 * M_PI);
    v_high_limits.push_back(10);
    v_high_limits.push_back(10);
    v_high_limits.push_back(2 * M_PI);
}
