#PBS -N hyperopt
#PBS -l select=1:ncpus=2:mem=14GB:ompthreads=2
#PBS -l walltime=24:00:00
#PBS -J 1-100
#PBS -e /rds/general/user/ahk114/ephemeral
#PBS -o /rds/general/user/ahk114/ephemeral

/rds/general/user/ahk114/home/.pyenv/versions/miniconda3-latest/envs/wildfires/bin/hyperopt-mongo-worker --mongo=maritimus.webredirect.org:1234/ba --poll-interval=1