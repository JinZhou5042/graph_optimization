vine_factory -T condor \
    -C factory_dv5.json \
    --scratch-dir /scratch365/jzhou24/factory_dv5 \
    --python-env dv5.tar.gz \
    --wrapper /users/jzhou24/graph_optimization/factories/gdb_wrapper.sh
