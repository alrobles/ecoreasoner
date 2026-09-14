#!/usr/bin/env bash
#SBATCH --job-name=exp-test
#SBATCH --partition=sixhour
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/exp_test_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/exp_test_%j.err
set -x
echo "TAG=[${TAG:-UNSET}]"
echo "DATA_CACHE=[${DATA_CACHE:-UNSET}]"
echo "CURRICULUM=[${CURRICULUM:-UNSET}]"
echo "CUR_STAGES=[${CUR_STAGES:-UNSET}]"
echo "PAIRS_DIR=[${PAIRS_DIR:-UNSET}]"
env | grep -E 'TAG|DATA_CACHE|CURRICULUM|CUR_STAGES|PAIRS' | sort