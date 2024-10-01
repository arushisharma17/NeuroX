```bash
bash-5.1$ note_4_demo
Install HypHC using legacy pytorch1.6.0-CUDA10.2-py37
Source: https://github.com/HazyResearch/HypHC
Last Update: 2024-09-04

#1. micromamba-install required packages torch1.6.0-cuda10.2-py37

#on COMPUTE node of V100 or A100.

$ cd /to/your/prj/
$ module load micromamba
$ micromamba create -p $PWD/micromamba/envs/hyphc -c conda-forge -c nvidia  cython==0.29.21 networkx==2.2 numpy==1.19.2 pytorch==1.6.0 tqdm==4.31.1 cudatoolkit=10.2 python=3.7 cudnn=7.6.5

$ eval "$(micromamba shell hook --shell bash)"
$ micromamba activate  $PWD/micromamba/envs/hyphc

#2. install HypHC
$ git clone https://github.com/HazyResearch/HypHC
$ cd HypHC

# Comment out to activate  python env as we used micromamba env.
$ sed -i 's/source/#source/' set_env.sh
$ cat set_env.sh
$ source set_env.sh
$ echo $HHC_HOME

$ cd $HHC_HOME/mst; python setup.py build_ext --inplace
$ cd $HHC_HOME/unionfind; python setup.py build_ext --inplace

#3 download sample datasets
$ cd $HHC_HOME   # where you git clone HypHC
$ source  set_env.sh
$ cat download_data.sh
$ sh download_data.sh

#4. quick run training on a100 ( 10-11m ) or v100 ( ~6 min)

# for a100: ~ 7min to convert/load legacy code to sm80 a100.
# will run 4min for traning.

$ which python     $ should be from your micromamba env.
$ python train.py
```
