if install pywr on Linux system, please do steps below:

1.
cd /mnt/iusers01/mace01/q14944pz/PyModules/pywr   # or wherever your repo is
nano install.sh    # or vim, etc.
# paste the script, save, exit

Save (Write Out) the file, Press:
Ctrl + O
Exit nano, Press:
Ctrl + X

2.
chmod +x install.sh


3.
conda activate pywrnew
conda install -c conda-forge glpk lpsolve55  # example, adjust as needed


4
../install.sh              # installs from current directory


