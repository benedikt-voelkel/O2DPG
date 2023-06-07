#!/bin/bash

# make sure O2DPG + O2 is loaded
[ ! "${O2DPG_ROOT}" ] && echo "Error: This needs O2DPG loaded" && exit 1
[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 1

# the number of variations we want to do
NVARIATIONS=20
# set the number of TFs
NUMBER_OF_TFS=1

# low and high values for parameters to scan
# energy
eCM_low=900
eCM_high=14000
# number of (signal) events
ns_low=100
ns_high=2000
# number of workers
j_low=8
j_high=16


compute_ir()
{
  local events=$1
  # 26.7km / 300,000 (km/s) * 128 aka length(LHC) / c * orbits_per_TF
  local t_tf=0.00114
  local ir=$(echo "$events / ${t_tf}" | bc)
  local ir_low=$(echo "${ir} * 0.9" | bc)
  local ir_up=$(echo "${ir} * 1.1" | bc)
  ir_low=${ir_low%.*}
  ir_up=${ir_up%.*}
  ir=$(shuf -i ${ir_low}-${ir_up} -n 1)
  echo ${ir}
}

cwd="o2dpg_sim_resource_estimation"
mkdir ${cwd} 2>/dev/null
pushd ${cwd}

for ((i=0; i<NVARIATIONS; i++))
do
  # use shuf consistentliy; everything except for interaction rate could be done with RANDOM (range too small...)
  eCM=$(shuf -i ${eCM_low}-${eCM_high} -n 1)
  ns=$(shuf -i ${ns_low}-${ns_high} -n 1)
  interactionRate=$(compute_ir ${ns})
  j=$(shuf -i ${j_low}-${j_high} -n 1)
  # here, we just use RANDOM
  seed=$((RANDOM % (10000 - 1 + 1) + 1))
  echo ${eCM} ${ns} ${interactionRate} ${j} ${seed}
  continue
  echo "START VARIATION ${i}"
  echo "  eCM: ${eCM}"
  echo "  ns: ${ns}"
  echo "  interactionRate: ${interactionRate}"
  echo "  j: ${j}"
  # prepare directory, move into it and run
  this_dir=variation_eCM_${eCM}_ns_${ns}_interactionRate_${interactionRate}_j_${j}_seed_${seed}
  mkdir ${this_dir} 2>/dev/null
  pushd ${this_dir}
  # create our new workflow file
  ${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow.py -eCM ${eCM} -col pp -gen pythia8 -proc inel -tf ${NUMBER_OF_TFS} \
                                                          -ns ${ns} -e TGeant4                                \
                                                          -j ${j} -interactionRate ${interactionRate}         \
                                                          --include-qc                                        \
                                                          --include-analysis                                  \
                                                          -productionTag "alibi_O2DPG_pp_minbias_testbeam"    \
                                                          -run 301000 -seed ${seed}
  # run this workflow
  ${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json -tt aod --cpu-limit 28 > variation.log 2>&1
  # rename piepline metric file for convenience
  pipeline_file_out=$(find . -type f -name "pipeline_metric*")
  mv ${pipeline_file_out} pipeline_metric_eCM_${eCM}_ns_${ns}_interactionRate_${interactionRate}_j_${j}_seed_${seed}.log
  # back to parent directory
  popd
done

# collect all pipeline_metrics, also for convenience
collected_pipelines_dir=collected_pipeline_metrics
mkdir ${collected_pipelines_dir} 2>/dev/null
cp variation_*/pipeline_metric* ${collected_pipelines_dir}

# throw everything into a single pandas.DataFrame and store in JSON
out_df=df.json
${O2DPG_ROOT}/MC/utils/o2dpg_sim_metrics.py pandas-json -p ${collected_pipelines_dir}/* -o ${out_df}
# now comes the interesting part
out_fit_params=o2dpg_sim_resource_parameters.json
${O2DPG_ROOT}/MC/utils/resource_estimation/o2dpg_workflow_resource_fitting.py fit -dfs ${out_df} -o ${out_fit_params}
echo "FITS WRITTEN TO ${out_fit_params}"

# back to where the user started
popd


