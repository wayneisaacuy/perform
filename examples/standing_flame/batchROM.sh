module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

dt=1e-09
nrsteps=40000
#updateFreq=1000
#useADEIM=ADEIM
use_FOM=0
outskip=10   

for latentDims in 4 5 6 7 8 9 10 11
do
for initWindowSize in 12 15 25 50 100 250 500
do
for adaptWindowSize in 5 7 11 12 13 14 15 25 50 75 100 250 500
do
for adaptevery in 2 3 4 5
do
for useADEIM in ADEIM AODEIM
do
for numrescomp in 2048 1024 512 256 128 64 32
do

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --sampling_update_freq $(( 1*adaptevery )) --ADEIM_update $useADEIM --use_FOM $use_FOM --out_skip $outskip --adapt_every $adaptevery --num_residual_comp $numrescomp"
"../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --sampling_update_freq $(( 5*adaptevery )) --ADEIM_update $useADEIM --use_FOM $use_FOM --out_skip $outskip --adapt_every $adaptevery --num_residual_comp $numrescomp"
"../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --sampling_update_freq $(( 10*adaptevery )) --ADEIM_update $useADEIM --use_FOM $use_FOM --out_skip $outskip --adapt_every $adaptevery --num_residual_comp $numrescomp"
"../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --sampling_update_freq $(( 50*adaptevery )) --ADEIM_update $useADEIM --use_FOM $use_FOM --out_skip $outskip --adapt_every $adaptevery --num_residual_comp $numrescomp"
"../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --sampling_update_freq $(( 100*adaptevery )) --ADEIM_update $useADEIM --use_FOM $use_FOM --out_skip $outskip --adapt_every $adaptevery --num_residual_comp $numrescomp"

done
done
done
done
done
done
