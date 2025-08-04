nohup python llm_generate.py \
    --model Qwen3:32B \
    --input code_ref/ref-wogfh-200-codebleu.jsonl \
    --output code_gen/Qwen3-32B-wogfh-200-codebleu-rag-top10-selfcorrect-iter1-static-wcorrect-iter5.jsonl \
    --use_rag \
    --rag_topk 10 \
    --use_selfcorrect \
    --llm_selfcorrect_iteration 1 \
    --use_static \
    --llm_static_iteration 5 \
    --use_static_w_selfcorrect \
    > logs/Qwen3-32B-wogfh-200-codebleu-rag-top10-selfcorrect-iter1-static-wcorrect-iter5.log 2>&1 &

nohup python llm_generate.py \
    --model Deepseek-32B:latest \
    --input code_ref/ref-gfh-46-passk.jsonl \
    --output code_gen/Deepseek-32B-gfh-46-passk-rag-top10-compile-iter10-static-iter5.jsonl \
    --use_rag \
    --rag_topk 10 \
    --use_compile \
    --llm_compile_iteration 10 \
    --use_static \
    --llm_static_iteration 5 \
    --use_static_w_compile \
    > logs/Deepseek-32B-gfh-46-passk-rag-top10-compile-iter10-static-iter5.log 2>&1 &

