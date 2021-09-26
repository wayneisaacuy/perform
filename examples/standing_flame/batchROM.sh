module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

#dt=1e-10
dt=1e-08
outskip=1
#nrsteps=20000
nrsteps=4000
#latentDims=2
updateFreq=1000
useADEIM=0
use_FOM=2

for latentDims in 3 5 7 10 15 20 #2 3 4 5
do
for initWindowSize in 15 25 50 100 250 500 #5 15 25 50 100 # depending on the singular values sigma_n/sigma_1 1e-6
do
for adaptWindowSize in 15 25 50 75 100 250 500 #5 10 20 30 50 #5 10 20 30 50 75 300 500
do

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --adapt_update_freq $updateFreq --out_skip $outskip --ADEIM_update $useADEIM --use_FOM $use_FOM"

done
done
done

