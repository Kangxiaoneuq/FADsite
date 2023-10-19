## About RLBind

A novel fusion technology utilizing complex network and sequence information for FAD-binding site identification.

The benchmark datasets can be found in ./dataset, the codes for FADsite are available in ./PDB. And the PDB files of proteins are saved in ./results. Furthermore, the demo and corresponding documentation files can be found in ./demo. See our paper for more details.

Testing each proteins takes approximately 2 minute, depending on the sequence length.


### Testing the FADsite_seq

```bash
cd ./src/
python FADsite_seq.py  
```

## Test the FADsite on test4 (~6min)
```bash
cd ./src/
python FADsite_test4.py  
```

### Testing the FADsite on test6 (~16min)
```bash
cd ./src/
python FADsite_test6.py  
```
### contact
Kang Xiao: xiaokangneuq@163.com

