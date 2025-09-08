vine_factory -T condor \
    -C factory_rstriphoton.json \
    --scratch-dir /scratch365/jzhou24/factory_rstriphoton \
    --python-env rstriphoton.tar.gz \
    --wrapper /users/jzhou24/graph_optimization/factories/gdb_wrapper.sh
