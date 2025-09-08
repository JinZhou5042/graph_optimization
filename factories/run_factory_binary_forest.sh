vine_factory -T condor \
    -C factory_binary_forest.json \
    --scratch-dir /scratch365/jzhou24/factory_binary_forest \
    --python-env binary_forest.tar.gz \
    --wrapper /users/jzhou24/graph_optimization/factories/gdb_wrapper.sh
