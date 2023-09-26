# FADsite
A novel fusion technology utilizing complex network and sequence information for FAD-binding site identification.

The protein id, sequence and labels are available in ./dataset. The PDB files of proteins are availavle in ./pdb. The codes for CNRBind are available in ./src. The demo and corresponding documentation files can be found in ./demo. See our paper for more details.

Testing each proteins takes approximately 2 minute, depending on the sequence length.

## Test the FADsite_seq
cd ./src/
python FADsite_seq.py  

## Test the FADsite on test4 (~6min)
cd ./src/
python FADsite_test4.py  

## Test the FADsite on test6 (~16min)
cd ./src/
python FADsite_test6.py  

Kang Xiao: xiaokangneuq@163.com
