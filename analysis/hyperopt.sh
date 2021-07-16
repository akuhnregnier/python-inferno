#PBS -N hyperopt
#PBS -l select=1:ncpus=2:mem=24GB:ompthreads=2
#PBS -l walltime=24:00:00
#PBS -J 1-100
#PBS -e /rds/general/user/ahk114/ephemeral
#PBS -o /rds/general/user/ahk114/ephemeral

/rds/general/user/ahk114/home/.pyenv/versions/miniconda3-latest/envs/wildfires/bin/hyperopt-mongo-worker --mongo=146.0.189.20:1234/ba --poll-interval=1
