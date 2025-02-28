# See: https://github.com/riscv-collab/riscv-gnu-toolchain/issues/1669#issuecomment-2681539793

cd third_party/riscv-gnu-toolchain

sed -i '/shallow = true/d' .gitmodules
sed -i 's/--depth 1//g' Makefile.in
