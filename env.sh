module load languages/intel/2018-u3
alias q='squeue -u $USER'
alias job='sbatch stencil.job'
alias out='less stencil.out'
alias tjob='sbatch test.job'
alias tout='less test.out'
alias tmake='mpiicc TEST.c -std=c99'
alias qcancel='scancel -u $USER'
alias n='nano stencil.job'
