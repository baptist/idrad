# Indoor Person Identification with Radar Data.
Pytorch code for person identification with radar data for five different targets related to paper: <i>Indoor Person Identification Using a Low-Power FMCW Radar</i>.

Paper can be found at: https://ieeexplore.ieee.org/document/8333730



#### Data Set

Data set can be downloaded from: http://imec-int.com/idrad

Run command <i>python scripts/process_all.py --input \<root path\></i>

#### Train model

Run command <i>python estimate.py --targets target1 target2 target3 target4 target5 --name personid </i>

#### Test model

Run command <i>python estimate.py --params params/personid_bvalid.pt --test </i>