module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

dt=1.0e-09
nrsteps=35000
snapskip=50
out_skip=10

for latentDims 5 6 7 8 9
do

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/transient_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --adaptive 0 --init_window_size $nrsteps --initbasis_snap_skip $snapskip --out_skip $out_skip"

done


