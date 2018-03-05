#! /bin/sh
#SBATCH --job-name=neuralstyle
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8100
#SBATCH --mail-type=ALL
#SBATCH --workdir=/tudelft.net/...
#SBATCH --gres=gpu:pascal:1



python3 "runal (copy).py" "in/North_American_X-15.jpg" 



