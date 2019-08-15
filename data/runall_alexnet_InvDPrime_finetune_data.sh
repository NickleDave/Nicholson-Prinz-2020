#!/usr/bin/env bash

# don't use 'searchnets all' because we don't want to remake datasets!!! Re-using previous datasets from other expts
# Just train then test accuracy of trained nets
searchnets train ./configs/searchnets_alexnet_InvDPrime_finetune_data_RVvGV.ini
searchnets test ./configs/searchnets_alexnet_InvDPrime_finetune_data_RVvGV.ini

searchnets train ./configs/searchnets_alexnet_InvDPrime_finetune_data_RVvRHGV.ini
searchnets test ./configs/searchnets_alexnet_InvDPrime_finetune_data_RVvRHGV.ini

searchnets train ./configs/searchnets_alexnet_InvDPrime_finetune_data_2_v_5.ini
searchnets test ./configs/searchnets_alexnet_InvDPrime_finetune_data_RVvRHGV.ini