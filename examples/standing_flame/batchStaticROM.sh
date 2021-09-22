module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

dt=1e-08
nrsteps=4000
snapskip=5

for latentDims in 3 5 7 10 15 20 25 50
do

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --adaptive 0 --init_window_size $nrsteps --initbasis_snap_skip $snapskip"

done


