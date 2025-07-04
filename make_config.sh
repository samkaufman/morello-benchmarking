#!/usr/bin/env bash
set -e

declare -a morello_matmul_sizes=("8")
declare -a matmul_oneoff_sizes=("64" "128")
declare -a matmul_chain_sizes=("${matmul_sizes[@]}")

# Define sub-ranges for each day of the week to smooth benchmarking load
# Total range: 128-2048 (1921 sizes), divided into 7 roughly equal parts (~274 sizes each)
day_of_week=$(date +%w)
case $day_of_week in
    0) # Sunday
        mapfile -t matmul_sizes < <(seq 128 401)
        ;;
    1) # Monday  
        mapfile -t matmul_sizes < <(seq 402 675)
        ;;
    2) # Tuesday
        mapfile -t matmul_sizes < <(seq 676 949)
        ;;
    3) # Wednesday
        mapfile -t matmul_sizes < <(seq 950 1223)
        ;;
    4) # Thursday
        mapfile -t matmul_sizes < <(seq 1224 1497)
        ;;
    5) # Friday
        mapfile -t matmul_sizes < <(seq 1498 1771)
        ;;
    6) # Saturday
        mapfile -t matmul_sizes < <(seq 1772 2048)
        ;;
esac


# Get the latest commit hash of the main branch
declare -r MORELLO_HASH=$(
    curl -s "https://api.github.com/repos/samkaufman/morello/branches/main" |
    jq -r '.commit.sha')

echo "max_work_time = 10800"  # 3 hours in seconds
echo ""

echo '[[jobs]]'
echo 'name = "gemma-decode-2b"'
echo "size = 1"
echo 'batch_size = 1'
echo "backend_name = \"gemma.cpp\""
echo "docker_path = \"./gemma.cpp/decode\""
echo "command = []"
echo ""

for i in "${matmul_oneoff_sizes[@]}"; do
echo '[[jobs]]'
echo 'name = "matmul-u8s8s16"'
echo "size = $i"
echo 'batch_size = 1'
echo "backend_name = \"morello-oneoff\""
echo "docker_path = \"./morello-oneoff-matmul-u8s8s16/$i\""
echo "command = []"
echo ""
done

# Temporarily disable Morello matmuls. Synthesis is too slow on HEAD.
# TODO: Re-enable.
#
# for i in "${morello_matmul_sizes[@]}"; do
# echo '[[jobs]]'
# echo 'name = "matmul"'
# echo "size = $i"
# echo 'batch_size = 1'
# echo "backend_name = \"morello\""
# echo "docker_path = \"./morello\""
# echo "docker_build_args = { MORELLO_VERSION = \"$MORELLO_HASH\" }"
# echo "command = [ \"matmul\", \"$i\" ]"
# echo ""
# done

for i in "${matmul_sizes[@]}"; do
echo '[[jobs]]'
echo 'name = "matmul-u8s8s16"'
echo "size = $i"
echo 'batch_size = 1'
echo 'backend_name = "aocl-4.2"'
echo 'docker_path = "./aocl"'
echo "command = [ \"u8s8s16\", \"$i\" ]"
echo ""

echo '[[jobs]]'
echo 'name = "matmul-f32"'
echo "size = $i"
echo 'batch_size = 1'
gflops_value=$(echo "scale=6; 2 * $i * $i * $i / 1000000000" | bc -l)
# Ensure proper decimal format (add leading 0 if missing)
if [[ $gflops_value == .* ]]; then
    gflops_value="0$gflops_value"
fi
echo "gflops = $gflops_value"
echo 'backend_name = "aocl-4.2"'
echo 'docker_path = "./aocl"'
echo "command = [ \"f32\", \"$i\" ]"
echo ""

for b in "tvm" "eigen"; do
    echo '[[jobs]]'
    echo 'name = "matmul-u32"'
    echo "size = $i"
    echo 'batch_size = 1'
    echo "backend_name = \"$b\""
    echo "docker_path = \"./$b\""
    echo "command = [ \"$i\" ]"
    echo ""
done
done

for i in "${matmul_chain_sizes[@]}"; do
echo '[[jobs]]'
echo 'name = "matmul-chain-f32"'
echo "size = $i"
echo 'batch_size = 1'
echo 'backend_name = "torch"'
echo 'docker_path = "./torch"'
echo "command = [ \"$i\" ]"
echo ""
done

echo '[reporters.google_sheets]'
echo 'key_file = "../morellosecrets/morello-339002-7f91851fee55.json"'
echo 'sheet_name = "Morello Performance Benchmarks"'
echo 'folder_name = "BETA"'