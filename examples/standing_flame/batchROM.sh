module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

dt=1e-09
nrsteps=40000
# updateFreq=1000
# useADEIM=ADEIM
use_FOM=0
outskip=10   
useLineSearch=0

for latentDims in 8 9 # 4 5 6 7 8 9 10 11
do
for initWindowSize in 12 15 # 12 15 25 50 100
do
for adaptWindowSize in $(( latentDims+1 )) 11 13 15 # 12 14 7 25 50 75 100 latentDims + 1
do
for adaptevery in 2 3 4 5 # 10 100
do
for useADEIM in AODEIM AFDEIM ADEIM
do
for numrescomp in 2048 # 1750 1024 # 512 256 128
do
for multiplier in 1 # 5 10 50
do

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --sampling_update_freq $(( multiplier*adaptevery )) --ADEIM_update $useADEIM --use_FOM $use_FOM --out_skip $outskip --adapt_every $adaptevery --num_residual_comp $numrescomp --use_line_search $useLineSearch" 


done
done
done
done
done
done
done
