#!/bin/bash

# Repeat first pair of commands 5 times
#for i in {1..5}; do
#  echo "Run $i of Individual Edge"
#  rm -rf models/*
#  code/execute_batch_of_configs.sh configurations/13_paper1_updated/z_Individual_Edge/
#done


rm -rf models/*
code/execute_batch_of_configs.sh configurations/14_multiclass/z_Individual_Edge/

rm -rf models/*
code/execute_batch_of_configs.sh configurations/14_multiclass/z_Individual_MQTT/

rm -rf models/*
code/execute_batch_of_configs.sh configurations/14_multiclass/z_Individual_IoT/
