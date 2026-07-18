#!/bin/bash
cd $1

export JAX_ENABLE_X64=True
uv run python $2 $3 $4 $5 $6 $7 $8 $9

exit