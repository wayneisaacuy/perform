module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

dt=5.0e-08
nrsteps=14000
updateFreq=1000
useADEIM=0
use_FOM=1
outskip=10   

for latentDims in 5 6 7 8 9 10
do
for initWindowSize in 15 25 50 100 250
do
for adaptWindowSize in 11 12 13 14 15 25 50 75 100 250
do

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/transient_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --adapt_update_freq $updateFreq --ADEIM_update $useADEIM --use_FOM $use_FOM --out_skip $outskip"

done
done
done

