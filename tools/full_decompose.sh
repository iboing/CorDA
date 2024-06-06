CUDA_VISIBLE_DEVICES=0 python -u build_corda.py --model_id="meta-llama/Llama-2-7b-hf" \
    --cov_aware --r $1 --mode full_decompose \
    --use_cache --calib_dataset "wikitext2" --calib_loader_size 256 \

# CO-SVD in CorDA: --cov_aware
# ASVD: --act_aware
# Plain SVD: none