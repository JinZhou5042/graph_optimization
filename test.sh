#!/bin/bash
trap "exit 130" SIGINT

bash part2_fault_tolerance.sh --workflow DV5 --repeat-id 4 && \
bash part2_fault_tolerance.sh --workflow DV5 --repeat-id 5 && \
bash part2_fault_tolerance.sh --workflow DV5 --repeat-id 6