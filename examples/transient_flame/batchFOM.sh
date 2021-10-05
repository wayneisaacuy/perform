module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

dt=5.0e-8
nrsteps=14000

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 0 --dt $dt --nrsteps $nrsteps"
