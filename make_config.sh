#!/usr/bin/env bash
set -e

declare -a morello_matmul_sizes=("8")
declare -a matmul_oneoff_sizes=("64" "128")
declare -a matmul_chain_sizes=("64" "128" "256" "512" "1024")

# Function to calculate GFLOPS for f32 matrix multiplication
# Formula: 2 * M * K * N / 1,000,000,000
calculate_gflops() {
    local b=$1
    local m=$2
    local k=$3
    local n=$4
    local gflops_value=$(echo "scale=6; 2 * $b * $m * $k * $n / 1000000000" | bc -l)
    # Ensure proper decimal format (add leading 0 if missing)
    if [[ $gflops_value == .* ]]; then
        gflops_value="0$gflops_value"
    fi
    echo "$gflops_value"
}

# mkn_combinations: all combinations of power-of-two sizes
declare -a power_of_two_sizes=(64 128 256 512 1024)
declare -a mkn_combinations=()

# First, add all power-of-two combinations
for m in "${power_of_two_sizes[@]}"; do
for k in "${power_of_two_sizes[@]}"; do
for n in "${power_of_two_sizes[@]}"; do
    mkn_combinations+=("$m,$k,$n")
done
done
done

# Then add non-power-of-two square sizes
day_of_week=$(date +%w)
declare -a powers_of_two=(128 256 512 1024 2048)
case $day_of_week in
    0) # Sunday
        mapfile -t day_specific_sizes < <(seq 128 13 401)
        ;;
    1) # Monday
        mapfile -t day_specific_sizes < <(seq 402 13 675)
        ;;
    2) # Tuesday
        mapfile -t day_specific_sizes < <(seq 676 13 949)
        ;;
    3) # Wednesday
        mapfile -t day_specific_sizes < <(seq 950 13 1223)
        ;;
    4) # Thursday
        mapfile -t day_specific_sizes < <(seq 1224 13 1497)
        ;;
    5) # Friday
        mapfile -t day_specific_sizes < <(seq 1498 13 1771)
        ;;
    6) # Saturday
        mapfile -t day_specific_sizes < <(seq 1772 13 2048)
        ;;
esac
mapfile -t weekly_square_sizes < <(printf "%s\n" "${day_specific_sizes[@]}" "${powers_of_two[@]}" | sort -un)
for size in "${weekly_square_sizes[@]}"; do
    mkn_combinations+=("$size,$size,$size")
done

# Use the latest commit hash of the main branch
MORELLO_HASH=$(
    curl -s "https://api.github.com/repos/samkaufman/morello/branches/main" |
    jq -r '.commit.sha')
declare -r MORELLO_HASH

echo "max_work_time = 10800"  # 3 hours in seconds
echo "order = \"random\""
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

for m in 64 128 256 512 1024 2048; do
for k in 64 128 256 512 1024 2048; do
for n in 64 128 256 512 1024 2048; do
echo '[[jobs]]'
echo "name = \"matmul-f32-${m}x${k}x${n}\""
echo "size = $n"
echo 'batch_size = 1'
gflops_value=$(calculate_gflops 1 "$m" "$k" "$n")
echo "gflops = $gflops_value"
echo "backend_name = \"morello\""
echo "docker_path = \"./morello\""
echo "docker_build_args = { MORELLO_VERSION = \"$MORELLO_HASH\" }"
echo "command = [ \"/run_matmul_x86_example.sh\", \"$m\", \"$k\", \"$n\" ]"
echo ""
done
done
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
# echo "command = [ \"/run_bench.sh\", \"matmul\", \"$i\" ]"
# echo ""
# done

# Add batch-parallel for 2048x2048x2048 only
for batch_size in 2 4 8 16; do
    echo '[[jobs]]'
    echo "name = \"matmul-batch-parallel-f32-${batch_size}x2048x2048x2048\""
    echo "size = 2048"
    echo "batch_size = $batch_size"
    gflops_value=$(calculate_gflops "$batch_size" 2048 2048 2048)
    echo "gflops = $gflops_value"
    echo 'backend_name = "intel-mkl"'
    echo 'docker_path = "./intel-mkl"'
    echo "command = [ \"batch-parallel-f32\", \"$batch_size\", \"2048\", \"2048\", \"2048\" ]"
    echo "num_cores = $batch_size"
    echo ""

    echo '[[jobs]]'
    echo "name = \"matmul-batch-parallel-u8s8s32-${batch_size}x2048x2048x2048\""
    echo "size = 2048"
    echo "batch_size = $batch_size"
    echo 'backend_name = "intel-mkl"'
    echo 'docker_path = "./intel-mkl"'
    echo "command = [ \"batch-parallel-u8s8s32\", \"$batch_size\", \"2048\", \"2048\", \"2048\" ]"
    echo "num_cores = $batch_size"
    echo ""

    echo '[[jobs]]'
    echo "name = \"matmul-batch-parallel-f32-${batch_size}x2048x2048x2048\""
    echo "size = 2048"
    echo "batch_size = $batch_size"
    gflops_value=$(calculate_gflops "$batch_size" 2048 2048 2048)
    echo "gflops = $gflops_value"
    echo "backend_name = \"morello\""
    echo "docker_path = \"./morello\""
    echo "docker_build_args = { MORELLO_VERSION = \"$MORELLO_HASH\" }"
    echo "command = [ \"/run_matmul_batch_parallel_x86_example.sh\", \"$batch_size\", \"2048\", \"2048\", \"2048\" ]"
    echo "num_cores = $batch_size"
    echo ""
done

for mkn in "${mkn_combinations[@]}"; do
IFS=',' read -r m k n <<< "$mkn"
    
for aocl_version in "4.1" "4.2"; do
echo '[[jobs]]'
echo "name = \"matmul-u8s8s16-${m}x${k}x${n}\""
echo "size = $m"
echo 'batch_size = 1'
echo "backend_name = \"aocl-$aocl_version\""
echo 'docker_path = "./aocl"'
echo "docker_build_args = { AOCL_VERSION = \"$aocl_version\" }"
echo "command = [ \"u8s8s16\", \"$m\", \"$k\", \"$n\" ]"
echo ""

echo '[[jobs]]'
echo "name = \"matmul-f32-${m}x${k}x${n}\""
echo "size = $m"
echo 'batch_size = 1'
gflops_value=$(calculate_gflops 1 "$m" "$k" "$n")
echo "gflops = $gflops_value"
echo "backend_name = \"aocl-$aocl_version\""
echo 'docker_path = "./aocl"'
echo "docker_build_args = { AOCL_VERSION = \"$aocl_version\" }"
echo "command = [ \"f32\", \"$m\", \"$k\", \"$n\" ]"
echo ""
done

echo '[[jobs]]'
echo "name = \"matmul-u8s8s32-${m}x${k}x${n}\""
echo "size = $m"
echo 'batch_size = 1'
echo 'backend_name = "intel-mkl"'
echo 'docker_path = "./intel-mkl"'
echo "command = [ \"u8s8s32\", \"$m\", \"$k\", \"$n\" ]"
echo ""

echo '[[jobs]]'
echo "name = \"matmul-f32-${m}x${k}x${n}\""
echo "size = $m"
echo 'batch_size = 1'
gflops_value=$(calculate_gflops 1 "$m" "$k" "$n")
echo "gflops = $gflops_value"
echo 'backend_name = "intel-mkl"'
echo 'docker_path = "./intel-mkl"'
echo "command = [ \"f32\", \"$m\", \"$k\", \"$n\" ]"
echo ""

for b in "tvm" "eigen"; do
    # only square matrix jobs for tvm/eigen
    # TODO: Support non-square shapes.
    if [ "$m" -eq "$k" ] && [ "$k" -eq "$n" ]; then
        echo '[[jobs]]'
        echo "name = \"matmul-u32-${m}x${k}x${n}\""
        echo "size = $m"
        echo 'batch_size = 1'
        echo "backend_name = \"$b\""
        echo "docker_path = \"./$b\""
        echo "command = [ \"$m\" ]"
        echo ""
    fi
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
echo "key_file = \"${MORELLO_SECRETS_KEY_PATH}\""
echo 'sheet_name = "Morello Performance Benchmarks"'
echo 'folder_name = "BETA"'