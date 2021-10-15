module load python/intel/3.8.6

export PATH="$HOME/uni/apro/mlib:$PATH"

dt=1e-09
nrsteps=40000
updateFreq=1000
useADEIM=ADEIM
use_FOM=0
 
latentDims=8
adaptevery=5
initWindowSize=12
adaptWindowSize=9

for learn_rate in 0.75 1.1 1.5 #2 5 10 25 50 100 #5e-01 1e-01
do

pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --sampling_update_freq $updateFreq --ADEIM_update $useADEIM --use_FOM $use_FOM --adapt_every $adaptevery --learn_rate $learn_rate"

done

#outskip=10  

#for updaterank in 2 3 4
#do

#latentDims=8
#adaptevery=5
#initWindowSize=12
#adaptWindowSize=9

#pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --adapt_update_freq $updateFreq --ADEIM_update $useADEIM --use_FOM $use_FOM --out_skip $outskip --adapt_every $adaptevery --update_rank $updaterank"

#latentDims=9
#adaptevery=2
#initWindowSize=15
#adaptWindowSize=13

#pySLURM.py "../../perform/driver.py /scratch/work/peherstorfer/wtu1/perform/examples/standing_flame --calc_rom 1 --dt $dt --nrsteps $nrsteps --latent_dims $latentDims --init_window_size $initWindowSize --adapt_window_size $adaptWindowSize --adapt_update_freq $updateFreq --ADEIM_update $useADEIM --use_FOM $use_FOM --out_skip $outskip --adapt_every $adaptevery --update_rank $updaterank"

#done