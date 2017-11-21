#!/usr/bin/env bash

# system parameters
K=100
C=101

# timestepping options
TSTYPE=euler
DT=0.01
TFIN=10.0

OUTFILE=${TSTYPE}_k_${K}_c_${C}_dt_${DT}_tfin_${TFIN}.out

RUNLINE="./fdo1 -ksp_type preonly -pc_type lu -ts_type ${TSTYPE} -ts_dt ${DT} -ts_final_time ${TFIN} -k ${K} -c ${C} -monitor_solution"
echo ${RUNLINE}
echo ${OUTFILE}

eval ${RUNLINE} > ${OUTFILE}
