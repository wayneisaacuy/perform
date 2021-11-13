module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

dt=1.0e-08
nrsteps=7000
use_FOM=1
outskip=10   
useLineSearch=1

for latentDims in 5 6 7 8 9 10 11
do
for initWindowSize in 12 15
do
for adaptWindowSize in $(( latentDims+1 )) 11 13 15
do
for adaptevery in 1 2 3 4
do
for useADEIM in POD
do
for numrescomp in 2048
do
for multiplier in 1
do

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/transient_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --sampling_update_freq $(( multiplier*adaptevery )) --ADEIM_update $useADEIM --use_FOM $use_FOM --out_skip $outskip --adapt_every $adaptevery --num_residual_comp $numrescomp --use_line_search $useLineSearch" 

done
done
done
done
done
done
done