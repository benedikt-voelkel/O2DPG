#!/bin/bash

if [ "0$O2_ROOT" == "0" ]; then
    echo O2 environment not loaded 1>&2
    exit 1
fi

source $O2_ROOT/prodtests/full-system-test/workflow-setup.sh || { echo "workflow-setup.sh failed" 1>&2 && exit 1; }
