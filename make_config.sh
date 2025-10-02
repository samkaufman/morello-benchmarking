#!/usr/bin/env bash
set -e

# Parse command line arguments
USE_AVX512=false
DISABLE_TIMEOUT=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --avx512)
            USE_AVX512=true
            shift
            ;;
        --no-timeout)
            DISABLE_TIMEOUT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--avx512] [--no-timeout]"
            exit 1
            ;;
    esac
done

PHYSICAL_CORES=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l | tr -d '[:space:]')
if ! [[ "$PHYSICAL_CORES" =~ ^[0-9]+$ ]] || [ "$PHYSICAL_CORES" -lt 4 ]; then
    echo "Error: detected physical cores ('$PHYSICAL_CORES') is less than 4 or invalid." >&2
    exit 1
fi

declare -a SMALL_PARALLEL_FACTORS=(1 "$(( PHYSICAL_CORES / 2 ))" "$PHYSICAL_CORES")

declare -a matmul_oneoff_sizes=("64" "128")
declare -a matmul_chain_sizes=("64" "128" "256" "512" "1024")
declare -a power_of_two_sizes=(64 128 256 512 1024)

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

# mkn_u8s8s16_combinations: all combinations of power-of-two sizes
declare -a mkn_u8s8s16_combinations=()
for m in "${power_of_two_sizes[@]}"; do
for k in "${power_of_two_sizes[@]}"; do
for n in "${power_of_two_sizes[@]}"; do
    mkn_u8s8s16_combinations+=("$m,$k,$n")
done
done
done

# Then add non-power-of-two square sizes
day_of_week=$(date +%w)
declare -a powers_of_two=(128 256 512 1024 2048 4096)
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
    mkn_u8s8s16_combinations+=("$size,$size,$size")
done

# Use the latest commit hash of the main branch
MORELLO_HASH=$(
    curl -s "https://api.github.com/repos/samkaufman/morello/branches/main" |
    jq -r '.commit.sha')
declare -r MORELLO_HASH

if [ "$DISABLE_TIMEOUT" != true ]; then
    echo "max_work_time = 10800"  # 3 hours in seconds
fi
echo "order = \"random-new-first\""
echo ""

# Args: batch m k n
emit_f32_backend_trio() {
    local b="$1" m="$2" k="$3" n="$4"
    local gflops_value job_name
    gflops_value=$(calculate_gflops "$b" "$m" "$k" "$n")
    job_name="matmul-batch-parallel-f32-${b}x${m}x${k}x${n}"
    for backend in intel-mkl aocl-4.2 openblas; do
        echo '[[jobs]]'
        echo "name = \"${job_name}\""
        echo "size = $n"
        echo "batch_size = $b"
        echo "gflops = $gflops_value"
        echo "backend_name = \"$backend\""
        case $backend in
            intel-mkl) echo 'docker_path = "./intel-mkl"' ;;
            aocl-4.2) echo 'docker_path = "./aocl"' ;;
            openblas) echo 'docker_path = "./openblas"' ;;
        esac
        echo "command = [ \"batch-parallel-f32\", \"$b\", \"$m\", \"$k\", \"$n\" ]"
        echo "num_cores = $b"
        echo ""
    done
}

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
# echo "command = [ \"/run_bench.sh\", \"matmul\", \"$i\" ]"
# echo ""
# done

# Add batch-parallel for most square shapes
for batch_size in "${SMALL_PARALLEL_FACTORS[@]}"; do
# iterate over multiples of 96 (48*2) (100..4000) plus all powers of two 128..4096
mapfile -t seq_sizes < <(seq 100 96 4000)
mapfile -t sizes < <(printf "%s\n" "${seq_sizes[@]}" "${powers_of_two[@]}" | sort -un)
for n in "${sizes[@]}"; do
    emit_f32_backend_trio "$batch_size" "$n" "$n" "$n"

    echo '[[jobs]]'
    echo "name = \"matmul-batch-parallel-f32-${batch_size}x${n}x${n}x${n}\""
    echo "size = $n"
    echo "batch_size = \"$batch_size\""
    gflops_value=$(calculate_gflops "$batch_size" "$n" "$n" "$n")
    echo "gflops = $gflops_value"
    echo "backend_name = \"morello\""
    echo "docker_path = \"./morello\""
    echo "docker_build_args = { MORELLO_VERSION = \"$MORELLO_HASH\" }"
    if [ "$USE_AVX512" = true ]; then
        echo "command = [ \"/run_matmul_x86_parameterized_example.sh\", \"--avx512\", \"$batch_size\", \"$n\", \"$n\", \"$n\" ]"
    else
        echo "command = [ \"/run_matmul_x86_parameterized_example.sh\", \"$batch_size\", \"$n\", \"$n\", \"$n\" ]"
    fi
    echo ""

    echo '[[jobs]]'
    echo "name = \"matmul-batch-parallel-u8s8s32-${batch_size}x${n}x${n}x${n}\""
    echo "size = $n"
    echo "batch_size = $batch_size"
    echo 'backend_name = "intel-mkl"'
    echo 'docker_path = "./intel-mkl"'
    echo "command = [ \"batch-parallel-u8s8s32\", \"$batch_size\", \"$n\", \"$n\", \"$n\" ]"
    echo "num_cores = $batch_size"
    echo ""
done
done

# Do 2048x2048x2048 at other parallelism factors
for batch_size in $(seq 2 "$PHYSICAL_CORES" | sed -e "/^$(( PHYSICAL_CORES / 2 ))$/d" -e "/^$PHYSICAL_CORES$/d"); do
    emit_f32_backend_trio "$batch_size" "2048" "2048" "2048"

    echo '[[jobs]]'
    echo "name = \"matmul-batch-parallel-f32-${batch_size}x2048x2048x2048\""
    echo "size = 2048"
    echo "batch_size = \"$batch_size\""
    gflops_value=$(calculate_gflops "$batch_size" "2048" "2048" "2048")
    echo "gflops = $gflops_value"
    echo "backend_name = \"morello\""
    echo "docker_path = \"./morello\""
    echo "docker_build_args = { MORELLO_VERSION = \"$MORELLO_HASH\" }"
    if [ "$USE_AVX512" = true ]; then
        echo "command = [ \"/run_matmul_x86_parameterized_example.sh\", \"--avx512\", \"$batch_size\", \"2048\", \"2048\", \"2048\" ]"
    else
        echo "command = [ \"/run_matmul_x86_parameterized_example.sh\", \"$batch_size\", \"2048\", \"2048\", \"2048\" ]"
    fi
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
done

for mkn in "${mkn_u8s8s16_combinations[@]}"; do
IFS=',' read -r m k n <<< "$mkn"
    
echo '[[jobs]]'
echo "name = \"matmul-u8s8s16-${m}x${k}x${n}\""
echo "size = $m"
echo 'batch_size = 1'
echo 'backend_name = "aocl-4.2"'
echo 'docker_path = "./aocl"'
echo "command = [ \"u8s8s16\", \"$m\", \"$k\", \"$n\" ]"
echo ""

echo '[[jobs]]'
echo "name = \"matmul-u8s8s32-${m}x${k}x${n}\""
echo "size = $m"
echo 'batch_size = 1'
echo 'backend_name = "intel-mkl"'
echo 'docker_path = "./intel-mkl"'
echo "command = [ \"u8s8s32\", \"$m\", \"$k\", \"$n\" ]"
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