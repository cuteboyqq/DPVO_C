MODEL_CONFIG_PATH=dpvo_update.yaml

cd AshaCam/config
eazyai_cvt -cy $MODEL_CONFIG_PATH -ds "-c act-force-fp16,coeff-force-fp16"
# eazyai_cvt -cy $MODEL_CONFIG_PATH
