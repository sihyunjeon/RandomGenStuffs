#!/usr/bin/env bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
if [ -f "MyProxy" ]; then
    echo "MyProxy exists, running rivet"
    mkdir hide_job
    cd hide_job
    cmsrel CMSSW_14_2_0
    cd CMSSW_14_2_0/src
    cmsenv
    cd ../../
    mv ../rivet.py ./
    mv ../inputs.txt ./
    cmsRun rivet.py
    mv out.yoda ../${JobBatchName}.yoda
else
    source /cvmfs/cms.cern.ch/rucio/setup.sh
    voms-proxy-init --voms cms
fi
