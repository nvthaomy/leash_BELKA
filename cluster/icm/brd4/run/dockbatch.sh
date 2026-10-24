#!/bin/bash
#SBATCH --job-name=icm_brd4
#SBATCH --account=katritch_502
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=48:00:00
#SBATCH --partition=epyc-64
##SBATCH --gres-flags=disable-binding

# ======================
# Step 1a - MEL Docking
# ======================

ASSIGNED_CHUNK=0


# TODO These shpuld be set with replace
ProjectName="brd4_5xi4"  #"DOCK_7br3refinedMod004D"
SlurmJobName="W" #"G"
files=(../files/test_BRD4_1024bits_smiles_Me_chunk01.inx)
nproc=32
effort_1=2.
ScoreThreshold=100.0
N_jobs=1
echo $N_jobs

# sdfPath="."     #$(pwd)"/WorkingDir/EPOCH003/0_enumerated_DOCK_7br3refinedMod004D_chunks_20000_3000000/"  
# files=($sdfPath/*inx)


for ((ii=0; ii<=${#files[@]}-1; ii++))
do
   f="${files[$ii]}"
   inx_file=$f
   inx_name="${f##*/}"
   inx_name="${inx_name%.*}"

   echo Docking inx file: $inx_file

   /project/katritch_223/icm-3.9-3b/icmng -vlscluster _dockScan $ProjectName input=$inx_file effort=${effort_1} proc=${nproc} -noStructFilter scoreCutoff=${ScoreThreshold} name=${inx_name} output=${inx_name}_hitlist > ${SlurmJobName}_${ii}_${inx_name}.log 

   if grep -q "FINISHED at ligand" ${SlurmJobName}_${ii}_${inx_name}.log
      then
         echo  $(date) >  ${inx_name}.finished
   fi

   sleep 50

done
