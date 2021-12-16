for j in `seq 0 8`; do for i in `seq 1 10`; do time python3 src/main.py --config ./src/config.yaml --target ${j} --set ${i} --ratio 1.0; done; done
