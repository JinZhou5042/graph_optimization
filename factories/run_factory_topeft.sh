vine_factory -T condor \
    -C factory_topeft.json \
    --scratch-dir /scratch365/jzhou24/factory_topeft \
    --python-env topeft.tar.gz \
    --wrapper /users/jzhou24/graph_optimization/factories/gdb_wrapper.sh \
    -d all -o factory.log
