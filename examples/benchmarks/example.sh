# example bash file for executing `main.py` benchmarks
# device configurations and `nqubits` are based on DGX machine with
# 4x V100 32GB GPUs
FILENAME="dgx.dat" # log file name
export CUDA_VISIBLE_DEVICES="0"
for NQUBITS in 20 21 22 23 24 25 26 27 28 29 30
do
  python main.py --circuit qft --nqubits $NQUBITS --filename $FILENAME
  python main.py --circuit qft --nqubits $NQUBITS --filename $FILENAME --precision "single"
  python main.py --circuit variational --nqubits $NQUBITS --filename $FILENAME
  python main.py --circuit variational --nqubits $NQUBITS --filename $FILENAME --precision "single"
done
python main.py --circuit qft --nqubits 31 --filename $FILENAME --precision "single"
python main.py --circuit variational --nqubits 31 --filename $FILENAME --precision "single"
export CUDA_VISIBLE_DEVICES=""
for NQUBITS in 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
do
  python main.py --circuit qft --nqubits $NQUBITS --filename $FILENAME
  python main.py --circuit variational --nqubits $NQUBITS --filename $FILENAME
done

export CUDA_VISIBLE_DEVICES="0,1"
ACCEL="1/GPU:0,1/GPU:1"
python main.py --circuit qft --nqubits 31 --filename $FILENAME --device "/CPU:0" --accelerators $ACCEL
python main.py --circuit qft --nqubits 32 --filename $FILENAME --device "/CPU:0" --accelerators $ACCEL --precision "single"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
ACCEL="1/GPU:0,1/GPU:1,1/GPU:2,1/GPU:3"
python main.py --circuit qft --nqubits 32 --filename $FILENAME --device "/CPU:0" --accelerators $ACCEL
python main.py --circuit qft --nqubits 33 --filename $FILENAME --device "/CPU:0" --accelerators $ACCEL --precision "single"
ACCEL="2/GPU:0,2/GPU:1,2/GPU:2,2/GPU:3"
python main.py --circuit qft --nqubits 33 --filename $FILENAME --device "/CPU:0" --accelerators $ACCEL
python main.py --circuit qft --nqubits 34 --filename $FILENAME --device "/CPU:0" --accelerators $ACCEL --precision "single"
