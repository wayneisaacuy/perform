module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

dt=1e-8
nrsteps=4000
#latentDims=2
updateFreq=1000

for latentDims in 2 3
do
for initWindowSize in 5 15 25 50 100
do
for adaptWindowSize in 5 10 20 30 50
do

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --adapt_update_freq $updateFreq"

done
done
done

