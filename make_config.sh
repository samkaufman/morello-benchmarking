#!/usr/bin/env bash
set -e

declare -a morello_matmul_sizes=("8")
declare -a matmul_oneoff_sizes=("64" "128")
declare -a matmul_sizes=("8" "64" "128" "256" "512" "1024" "2048")

# Get the latest commit hash of the main branch
declare -r MORELLO_HASH=$(
    curl -s "https://api.github.com/repos/samkaufman/morello/branches/main" |
    jq -r '.commit.sha')

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

for i in "${morello_matmul_sizes[@]}"; do
echo '[[jobs]]'
echo 'name = "matmul"'
echo "size = $i"
echo 'batch_size = 1'
echo "backend_name = \"morello\""
echo "docker_path = \"./morello\""
echo "docker_build_args = { MORELLO_VERSION = \"$MORELLO_HASH\" }"
echo "command = [ \"matmul\", \"$i\" ]"
echo ""
done

for i in "${matmul_sizes[@]}"; do
echo '[[jobs]]'
echo 'name = "matmul-u8s8s16"'
echo "size = $i"
echo 'batch_size = 1'
echo 'backend_name = "aocl"'
echo 'docker_path = "./aocl"'
echo "command = [ \"$i\" ]"
echo ""

for b in "tvm" "eigen"; do
    echo '[[jobs]]'
    echo 'name = "matmul"'
    echo "size = $i"
    echo 'batch_size = 1'
    echo "backend_name = \"$b\""
    echo "docker_path = \"./$b\""
    echo "command = [ \"$i\" ]"
    echo ""
done
done

echo '[reporters.google_sheets]'
echo 'key_file = "../morellosecrets/morello-339002-7f91851fee55.json"'
echo 'sheet_name = "Morello Performance Benchmarks"'
echo 'folder_name = "BETA"'