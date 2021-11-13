module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

dt=1.0e-8
nrsteps=7000

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/transient_flame --calc_rom 0 --dt $dt --nrsteps $nrsteps"
