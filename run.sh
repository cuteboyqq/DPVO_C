LIB_PATH=$(realpath ./app/lib/libeazyai/build/)

if [[ ":$LD_LIBRARY_PATH:" != *":$LIB_PATH:"* ]]; then
	    export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH
fi

cd build

./wnc-app  \
	-m 2 \
	-d 0 \
	--in_dir videos