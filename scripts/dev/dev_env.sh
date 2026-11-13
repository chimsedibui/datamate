# Dev settings for runs on the NTQ API (source this). Python = conda env workdir.
export PATH=/home/khanh/miniconda3/envs/workdir/bin:$PATH
export LLM_CONTEXT=${LLM_CONTEXT:-65536} BUILD_MAX_S=${BUILD_MAX_S:-100000} BUILD_MAX_TOKENS=${BUILD_MAX_TOKENS:-24000}
export BUILD_EXAMPLES=${BUILD_EXAMPLES:-1}
