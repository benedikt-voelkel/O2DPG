#!/usr/bin/env bash

# shellcheck disable=SC1091
source common/setenv.sh

# shellcheck disable=SC1091
source common/getCommonArgs.sh

PROXY_INSPEC_EOS="eos:***/INFORMATION"
PROXY_INSPEC="calib:FT0/CALIB_INFO/0;${PROXY_INSPEC_EOS}"
CALIB_TIME_OFFSET_CONFIG="FT0CalibParam.mNExtraSlots=0"

WORKFLOW="o2-dpl-raw-proxy ${ARGS_ALL} --proxy-name ft0-timeoffset-input-proxy --dataspec \"${PROXY_INSPEC}\" --network-interface ib0 --channel-config \"name=ft0-timeoffset-input-proxy,method=bind,type=pull,rateLogging=1,transport=zeromq\" | "
WORKFLOW+="o2-calibration-ft0-time-offset-calib --configKeyValues \"${CALIB_TIME_OFFSET_CONFIG}\" ${ARGS_ALL} --tf-per-slot 26400 | "
WORKFLOW+="o2-calibration-ccdb-populator-workflow ${ARGS_ALL} --configKeyValues \"$ARGS_ALL_CONFIG\" --ccdb-path=\"http://ccdb-test.cern.ch:8080\" --sspec-min 1 --sspec-max 1 | "
WORKFLOW+="o2-dpl-run ${ARGS_ALL} ${GLOBALDPLOPT}"

if [ $WORKFLOWMODE == "print" ]; then
  echo Workflow command:
  echo $WORKFLOW | sed "s/| */|\n/g"
else
  # Execute the command we have assembled
  WORKFLOW+=" --$WORKFLOWMODE ${WORKFLOWMODE_FILE}"
  eval $WORKFLOW
fi
