#!/bin/bash
#SBATCH --job-name="$@"
#SBATCH --output=logs/%A.log
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
echo 	"Arguments:	$@"
echo -n	"Date:		"; date
echo 	"JobId:		$SLURM_JOBID"
echo	"Node:		$HOSTNAME"
echo	"Nodelist:	$SLURM_JOB_NODELIST"

source /home/krojerb/img-gen-project/ldm/bin/activate

python3 -u $@
